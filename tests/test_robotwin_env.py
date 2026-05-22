from __future__ import annotations

import numpy as np
import torch

from examples import train_ppo_robotwin_rgbd
from rl_garden.envs.robotwin import RoboTwinEnv, RoboTwinEnvConfig
import rl_garden.envs.robotwin.adapter as robotwin_adapter
from rl_garden.envs.robotwin.adapter import RoboTwinTaskAdapter, StepResult
from rl_garden.envs.robotwin.executor_shard import ShardedRoboTwinExecutor
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


class _FakeShard:
    instances = []

    def __init__(self, *, ctx, shard_id, cfg, task_args, env_seeds):
        del ctx, task_args
        self.shard_id = shard_id
        self.cfg = cfg
        self.env_seeds = list(env_seeds)
        self.closed = False
        self.obs = [self._obs(i, seed) for i, seed in enumerate(self.env_seeds)]
        self.pending = None
        self.step_actions = None
        self.start_step_called = False
        _FakeShard.instances.append(self)

    def start_reset(self, env_indices, env_seeds):
        self.pending = (env_indices, env_seeds)

    def finish_reset(self):
        env_indices, env_seeds = self.pending
        if env_indices is None:
            seeds = list(range(self.cfg.num_envs)) if env_seeds is None else env_seeds
            self.obs = [self._obs(i, seed) for i, seed in enumerate(seeds)]
        else:
            for offset, local_idx in enumerate(env_indices):
                seed = None if env_seeds is None else env_seeds[offset]
                self.obs[local_idx] = self._obs(local_idx, seed)
        self.pending = None
        return self.obs

    def start_step(self, actions):
        self.start_step_called = True
        self.step_actions = np.array(actions, copy=True)

    def finish_step(self):
        return [
            StepResult(
                obs=self._obs(i),
                reward=float(self.shard_id * 10 + i),
                terminated=False,
                truncated=False,
                info={"success": False, "instruction": f"shard {self.shard_id} env {i}"},
            )
            for i in range(self.cfg.num_envs)
        ]

    def start_get_obs(self):
        return None

    def finish_get_obs(self):
        return self.obs

    def close(self, clear_cache=True):
        del clear_cache
        self.closed = True

    def _obs(self, local_idx, seed=None):
        value = self.shard_id * 10 + local_idx
        return {
            "rgb": np.full((8, 8, 3), value, dtype=np.uint8),
            "rgb_left_wrist": None,
            "rgb_right_wrist": None,
            "state": np.ones(14, dtype=np.float32) * value,
            "instruction": f"shard {self.shard_id} env {local_idx}",
            "_env_seed": seed,
        }


class _Robot:
    def get_left_arm_jointState(self):
        return [0.0] * 7

    def get_right_arm_jointState(self):
        return [0.5] * 7

    def get_left_gripper_val(self):
        return 0.25

    def get_right_gripper_val(self):
        return 0.75


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


class _VideoTask:
    def __init__(self):
        self.eval_video_path = None
        self.ffmpeg = None
        self.closed_video = False
        self.closed_env = False

    def setup_demo(self, **kwargs):
        self.eval_video_path = kwargs.get("eval_video_save_dir")
        self.step_lim = kwargs["step_lim"]
        self.take_action_cnt = 0
        self.run_steps = 0
        self.reward_step = 0
        self.eval_success = False

    def get_obs(self):
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        return {
            "observation": {
                "head_camera": {"rgb": rgb},
                "left_camera": {"rgb": rgb},
                "right_camera": {"rgb": rgb},
            },
            "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
        }

    def get_instruction(self):
        return None

    def _set_eval_video_ffmpeg(self, ffmpeg):
        self.ffmpeg = ffmpeg
        self.eval_video_ffmpeg = ffmpeg

    def _del_eval_video_ffmpeg(self):
        self.closed_video = True
        del self.eval_video_ffmpeg

    def close_env(self, clear_cache=True):
        del clear_cache
        self.closed_env = True


class _FakeFFmpeg:
    def __init__(self):
        self.stdin = object()


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


def test_sharded_robotwin_executor_routes_reset_and_step_in_global_order():
    _FakeShard.instances = []
    cfg = RoboTwinEnvConfig(
        num_envs=5,
        device="cpu",
        image_size=(8, 8),
        executor_type="shard",
        shard_size=2,
    )
    executor = ShardedRoboTwinExecutor(
        cfg,
        task_args={"task_name": "fake"},
        env_seeds=[10, 11, 12, 13, 14],
        shard_factory=_FakeShard,
    )

    assert [shard.cfg.num_envs for shard in _FakeShard.instances] == [2, 2, 1]
    assert all(shard.cfg.executor_type == "thread" for shard in _FakeShard.instances)
    assert all(shard.cfg.parallel_topp is True for shard in _FakeShard.instances)
    assert all(shard.cfg.ctrl_concurrency == 1 for shard in _FakeShard.instances)

    obs = executor.reset(env_seeds=[20, 21, 22, 23, 24])
    assert [item["_env_seed"] for item in obs] == [20, 21, 22, 23, 24]
    assert [float(item["state"][0]) for item in obs] == [0.0, 1.0, 10.0, 11.0, 20.0]

    obs = executor.reset(env_indices=[1, 3, 4], env_seeds=[101, 103, 104])
    assert [item["_env_seed"] for item in obs] == [20, 101, 22, 103, 104]

    actions = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
    results = executor.step(actions)
    assert [res.reward for res in results] == [0.0, 1.0, 10.0, 11.0, 20.0]
    np.testing.assert_array_equal(_FakeShard.instances[0].step_actions, actions[:2])
    np.testing.assert_array_equal(_FakeShard.instances[1].step_actions, actions[2:4])
    np.testing.assert_array_equal(_FakeShard.instances[2].step_actions, actions[4:5])

    executor.close()
    assert all(shard.closed for shard in _FakeShard.instances)


def test_delta_joint_pos_action_conversion():
    cfg = RoboTwinEnvConfig(device="cpu", joint_delta_scale=0.1, gripper_delta_scale=0.25)
    adapter = RoboTwinTaskAdapter(0, cfg, {}, env_seed=0)
    adapter.task = type("Task", (), {"robot": _Robot()})()
    raw = adapter._to_robotwin_action(np.ones(14, dtype=np.float32))
    np.testing.assert_allclose(raw[:6], np.full(6, 0.1))
    assert raw[6] == 0.25
    np.testing.assert_allclose(raw[7:13], np.full(6, 0.6))
    assert raw[13] == 0.75


def test_ee_delta_pose_action_space_and_conversion():
    cfg = RoboTwinEnvConfig(
        control_mode="ee_delta_pose",
        device="cpu",
        image_size=(8, 8),
        ee_delta_pos_scale=0.01,
        ee_delta_rot_scale=0.2,
        gripper_delta_scale=0.1,
    )
    env = RoboTwinEnv(cfg, executor=FakeExecutor(num_envs=1))
    assert env.single_action_space.shape == (14,)
    assert np.all(env.single_action_space.low == -1.0)
    assert np.all(env.single_action_space.high == 1.0)
    env.close()

    adapter = RoboTwinTaskAdapter(0, cfg, {}, env_seed=0)
    adapter.task = type("Task", (), {"robot": _Robot()})()
    action = np.zeros(14, dtype=np.float32)
    action[0] = 1.0
    action[5] = 1.0
    action[6] = 1.0
    action[7] = -1.0
    action[13] = -1.0
    raw = adapter._to_robotwin_action(action)
    assert adapter._robotwin_action_type() == "ee"
    assert raw.shape == (16,)
    np.testing.assert_allclose(raw[0:3], [0.01, 0.0, 0.0])
    np.testing.assert_allclose(
        raw[3:7], [np.cos(0.1), 0.0, 0.0, np.sin(0.1)], atol=1e-6
    )
    np.testing.assert_allclose(raw[7], 0.35)
    np.testing.assert_allclose(raw[8:11], [-0.01, 0.0, 0.0])
    np.testing.assert_allclose(raw[11:15], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(raw[15], 0.65)


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


def test_robotwin_ppo_eval_env_wires_video_dir(monkeypatch, tmp_path):
    captured = []

    def fake_make_robotwin_env(cfg):
        captured.append(cfg)
        return cfg

    monkeypatch.setattr(train_ppo_robotwin_rgbd, "make_robotwin_env", fake_make_robotwin_env)
    args = train_ppo_robotwin_rgbd.Args(log_type="none")
    train_ppo_robotwin_rgbd._make_env(
        args,
        num_envs=1,
        is_eval=True,
        eval_record_dir=str(tmp_path),
    )
    train_ppo_robotwin_rgbd._make_env(
        args,
        num_envs=1,
        is_eval=False,
        eval_record_dir=str(tmp_path),
    )

    eval_cfg = captured[0].task_config
    train_cfg = captured[1].task_config
    assert eval_cfg["eval_video_log"] is True
    assert eval_cfg["eval_video_save_dir"] == str(tmp_path)
    assert train_cfg["eval_video_log"] is False
    assert "eval_video_save_dir" not in train_cfg


def test_robotwin_adapter_starts_and_stops_eval_video(monkeypatch, tmp_path):
    task = _VideoTask()
    popen_calls = []

    def fake_make_task(task_name, robotwin_root=None):
        del task_name, robotwin_root
        return task

    def fake_popen(args, stdin):
        popen_calls.append((args, stdin))
        return _FakeFFmpeg()

    monkeypatch.setattr(robotwin_adapter, "make_task", fake_make_task)
    monkeypatch.setattr(robotwin_adapter, "_ffmpeg_executable", lambda: "ffmpeg")
    monkeypatch.setattr(robotwin_adapter.subprocess, "Popen", fake_popen)

    cfg = RoboTwinEnvConfig(
        task_name="place_shoe",
        reward_mode="sparse",
        device="cpu",
        image_size=(8, 8),
        task_config={
            "step_lim": 2,
            "eval_video_log": True,
            "eval_video_save_dir": str(tmp_path),
            "left_robot_file": "left.yml",
            "right_robot_file": "right.yml",
        },
    )
    adapter = RoboTwinTaskAdapter(0, cfg, cfg.task_config, env_seed=123)
    obs = adapter.reset()

    assert obs["rgb"].shape == (8, 12, 3)
    assert task.ffmpeg is not None
    assert popen_calls
    args, stdin = popen_calls[0]
    assert stdin is robotwin_adapter.subprocess.PIPE
    assert "12x8" in args
    assert str(tmp_path / "episode_env0_seed123_0.mp4") in args

    adapter.close()
    assert task.closed_video is True
    assert task.closed_env is True
