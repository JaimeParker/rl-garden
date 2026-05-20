from __future__ import annotations

import numpy as np
import torch

from rl_garden.envs.robotwin import RoboTwinEnv, RoboTwinEnvConfig
import rl_garden.envs.robotwin.adapter as robotwin_adapter
from rl_garden.envs.robotwin.adapter import RoboTwinTaskAdapter, StepResult
from rl_garden.envs.robotwin.rewards import build_task_reward, supported_reward_tasks


class FakeExecutor:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.step_count = 0

    def reset(self, env_indices=None, env_seeds=None):
        del env_indices
        seeds = env_seeds if env_seeds is not None else list(range(self.num_envs))
        return [self._obs(i, seeds[i]) for i in range(self.num_envs)]

    def step(self, actions):
        assert actions.shape == (self.num_envs, 14)
        self.step_count += 1
        done = self.step_count == 2
        return [
            StepResult(
                obs=self._obs(i),
                reward=float(i + 1),
                terminated=done and i == 0,
                truncated=done and i == 1,
                info={"success": done and i == 0, "instruction": f"task {i}"},
            )
            for i in range(self.num_envs)
        ]

    def close(self):
        return None

    @staticmethod
    def _obs(i, seed=None):
        return {
            "rgb": np.full((8, 8, 3), i, dtype=np.uint8),
            "rgb_left_wrist": None,
            "rgb_right_wrist": None,
            "state": np.ones(14, dtype=np.float32) * i,
            "instruction": f"task {i}",
            "_env_seed": seed,
        }


class _Robot:
    def get_left_arm_jointState(self):
        return [0.0] * 7

    def get_right_arm_jointState(self):
        return [0.5] * 7


class _Pose:
    def __init__(self, p):
        self.p = np.array(p, dtype=np.float32)


class _Actor:
    def __init__(self, p):
        self._pose = _Pose(p)

    def get_pose(self):
        return self._pose


class _StackBowlsTask:
    def __init__(self):
        self.bowl1 = _Actor([0.0, -0.1, 0.76])
        self.bowl2 = _Actor([0.0, -0.1, 0.79])
        self.bowl3 = _Actor([0.0, -0.1, 0.82])


def test_robotwin_env_reset_step_and_auto_reset_contract():
    cfg = RoboTwinEnvConfig(num_envs=2, device="cpu", image_size=(8, 8), max_episode_steps=10)
    env = RoboTwinEnv(cfg, executor=FakeExecutor(num_envs=2))
    obs, infos = env.reset(seed=0)
    assert set(obs) == {"rgb", "rgb_left_wrist", "rgb_right_wrist", "state"}
    assert obs["rgb"].shape == (2, 8, 8, 3)
    assert obs["rgb"].device.type == "cpu"
    assert infos["instructions"] == ["task 0", "task 1"]
    assert infos["env_seed"].shape == (2,)

    actions = torch.zeros(2, 14)
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert rewards.tolist() == [1.0, 2.0]
    assert not terms.any()
    assert not truncs.any()
    assert "episode" in infos

    obs, rewards, terms, truncs, infos = env.step(actions)
    assert terms.tolist() == [True, False]
    assert truncs.tolist() == [False, True]
    assert "final_observation" in infos
    assert infos["_final_info"].tolist() == [True, True]
    assert infos["final_info"]["episode"]["return"].shape == (2,)
    env.close()


class UnStableError(Exception):
    pass


class _RetryTask:
    setup_seeds = []
    close_calls = []

    def setup_demo(self, *, now_ep_num, seed, **kwargs):
        del now_ep_num, kwargs
        self.setup_seeds.append(seed)
        if seed < 12:
            raise UnStableError(f"unstable seed {seed}")

    def close_env(self, clear_cache=True):
        self.close_calls.append(clear_cache)

    def get_obs(self):
        return {
            "observation": {"head_camera": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}},
            "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        }

    def get_instruction(self):
        return "retry task"


def test_robotwin_adapter_retries_unstable_reset_seeds(monkeypatch):
    task = _RetryTask()
    task.setup_seeds = []
    task.close_calls = []
    monkeypatch.setattr(robotwin_adapter, "make_task", lambda *args, **kwargs: task)

    cfg = RoboTwinEnvConfig(device="cpu", reward_mode="sparse")
    adapter = RoboTwinTaskAdapter(
        0,
        cfg,
        {
            "left_robot_file": "/tmp/left",
            "right_robot_file": "/tmp/right",
            "left_embodiment_config": {},
            "right_embodiment_config": {},
        },
        env_seed=10,
    )

    obs = adapter.reset()

    assert task.setup_seeds == [10, 11, 12]
    assert task.close_calls == [True, True]
    assert adapter.env_seed == 12
    assert obs["_env_seed"] == 12


def test_robotwin_env_syncs_actual_reset_seed_from_executor():
    class SeedChangingExecutor(FakeExecutor):
        def reset(self, env_indices=None, env_seeds=None):
            obs = super().reset(env_indices=env_indices, env_seeds=env_seeds)
            for item in obs:
                item["_env_seed"] = 12345
            return obs

    cfg = RoboTwinEnvConfig(num_envs=2, device="cpu", image_size=(8, 8))
    env = RoboTwinEnv(cfg, executor=SeedChangingExecutor(num_envs=2))

    _, infos = env.reset(seed=0)

    assert env.reset_state_ids.tolist() == [12345, 12345]
    assert infos["env_seed"].tolist() == [12345, 12345]
    env.close()


def test_delta_joint_pos_action_conversion():
    cfg = RoboTwinEnvConfig(device="cpu", joint_delta_scale=0.1, gripper_delta_scale=0.25)
    adapter = RoboTwinTaskAdapter(0, cfg, {}, env_seed=0)
    adapter.task = type("Task", (), {"robot": _Robot()})()
    raw = adapter._to_robotwin_action(np.ones(14, dtype=np.float32))
    np.testing.assert_allclose(raw[:6], np.full(6, 0.1))
    assert raw[6] == 0.25
    np.testing.assert_allclose(raw[7:13], np.full(6, 0.6))
    assert raw[13] == 0.75


def test_reward_registry_covers_rlinf_robotwin_env_configs():
    assert supported_reward_tasks() == (
        "adjust_bottle",
        "beat_block_hammer",
        "click_bell",
        "handover_block",
        "lift_pot",
        "move_can_pot",
        "pick_dual_bottles",
        "place_container_plate",
        "place_empty_cup",
        "place_shoe",
        "stack_bowls_three",
    )


def test_stack_bowls_three_dense_reward_factory_builds():
    task = _StackBowlsTask()
    reward = build_task_reward("stack_bowls_three", task)
    reward.update()
    assert reward.compute_reward() > 0.0
    assert task.reward is reward
