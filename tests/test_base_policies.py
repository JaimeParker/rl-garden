from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.policies.base_policies import (
    RoboTwinACTEEPoseBasePolicy,
    SACBasePolicy,
    ZeroBasePolicy,
)
from rl_garden.policies.base_policies.base import BasePolicyOutput
from rl_garden.policies.base_policies.factory import make_base_policy


class DummyVecEnv:
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box) -> None:
        self.num_envs = 2
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(action_space.low, (self.num_envs,) + action_space.shape),
            high=np.broadcast_to(action_space.high, (self.num_envs,) + action_space.shape),
            dtype=action_space.dtype,
        )


def _state_env() -> DummyVecEnv:
    return DummyVecEnv(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )


def _rgb_env() -> DummyVecEnv:
    return DummyVecEnv(
        spaces.Dict(
            {
                "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )


def _sac_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 16,
        "batch_size": 4,
        "training_freq": 4,
        "learning_starts": 4,
        "eval_freq": 0,
        "net_arch": [16],
    }


def test_zero_base_policy_returns_env_space_zero_actions() -> None:
    env = _state_env()
    provider = ZeroBasePolicy(
        env.single_observation_space,
        env.single_action_space,
        device="cpu",
    )

    output = provider.select_action(torch.ones(env.num_envs, 4))

    assert output.actions.shape == (env.num_envs, 2)
    assert output.actions.device.type == "cpu"
    assert torch.equal(output.actions, torch.zeros(env.num_envs, 2))


def test_sac_base_policy_loads_state_checkpoint(tmp_path) -> None:
    env = _state_env()
    agent = SAC(env=env, **_sac_kwargs())
    path = tmp_path / "sac_state.pt"
    agent.save(path)

    provider = SACBasePolicy.from_checkpoint(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        checkpoint_path=path,
        device="cpu",
    )
    output = provider.select_action(torch.zeros(env.num_envs, 4))

    assert output.actions.shape == (env.num_envs, 2)
    assert torch.all(output.actions <= 1.0)
    assert torch.all(output.actions >= -1.0)


def test_sac_base_policy_loads_rgb_checkpoint(tmp_path) -> None:
    env = _rgb_env()
    agent = SAC(env=env, **_sac_kwargs(), image_keys=("rgb",))
    path = tmp_path / "sac_rgb.pt"
    agent.save(path)

    provider = SACBasePolicy.from_checkpoint(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        checkpoint_path=path,
        device="cpu",
    )
    obs = {
        "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
        "state": torch.zeros(env.num_envs, 4),
    }
    output = provider.select_action(obs)

    assert output.actions.shape == (env.num_envs, 2)


def test_sac_base_policy_rejects_mismatched_action_space(tmp_path) -> None:
    env = _state_env()
    agent = SAC(env=env, **_sac_kwargs())
    path = tmp_path / "sac_state.pt"
    agent.save(path)
    mismatched_action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(3,),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="action_space metadata"):
        SACBasePolicy.from_checkpoint(
            observation_space=env.single_observation_space,
            action_space=mismatched_action_space,
            checkpoint_path=path,
            device="cpu",
        )


def test_act_factory_uses_robotwin_ee_pose_bridge_by_control_semantics(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_from_checkpoint(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "rl_garden.policies.base_policies.factory."
        "RoboTwinACTEEPoseBasePolicy.from_checkpoint",
        fake_from_checkpoint,
    )
    env = _rgb_env()
    env.cfg = SimpleNamespace(control_mode="ee_pose")  # type: ignore[attr-defined]
    env.single_action_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(14,),
        dtype=np.float32,
    )
    env.qpos_targets_to_ee_pose = lambda actions: actions  # type: ignore[attr-defined]

    result = make_base_policy(
        base_policy="act",
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        env=env,
        base_ckpt_path="/checkpoints/policy_last.ckpt",
        base_act_stats_path="/checkpoints/dataset_stats.pkl",
    )

    assert result is sentinel
    assert captured["env"] is env
    assert captured["stats_path"] == "/checkpoints/dataset_stats.pkl"


def test_robotwin_act_ee_pose_base_policy_converts_local_qpos_output() -> None:
    class FakeACT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reset_calls = 0

        def select_action(self, obs):
            del obs
            return BasePolicyOutput(actions=torch.arange(28).reshape(2, 14).float())

        def reset(self, env_ids=None):
            del env_ids
            self.reset_calls += 1

        def bind_env(self, env):
            del env

    class FakeEEEnv:
        def __init__(self):
            self.calls = 0

        def qpos_targets_to_ee_pose(self, actions):
            self.calls += 1
            return actions + 100.0

    observation_space = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
    action_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
    act_policy = FakeACT()
    env = FakeEEEnv()
    policy = RoboTwinACTEEPoseBasePolicy(
        act_policy,  # type: ignore[arg-type]
        env=env,
        observation_space=observation_space,
        action_space=action_space,
    )

    output = policy.select_action(torch.zeros(2, 14))

    assert env.calls == 1
    assert output.actions.shape == (2, 14)
    torch.testing.assert_close(
        output.actions,
        torch.arange(28).reshape(2, 14).float() + 100.0,
    )
    assert output.info is not None
    assert output.info["qpos_actions"].shape == (2, 14)
    policy.reset()
    assert act_policy.reset_calls == 1
