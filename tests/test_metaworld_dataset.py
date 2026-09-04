"""Tests for the ``metaworld`` live scripted-policy dataset backend.

Fakes ``metaworld.policies.ENV_POLICY_MAP`` (inserted into ``sys.modules``)
and monkeypatches ``gymnasium.make`` -- no real MuJoCo/metaworld install
needed, same strategy as ``test_metaworld_env.py``.
"""
from __future__ import annotations

import sys
import types

import numpy as np
from gymnasium import spaces

from rl_garden.buffers import (
    MCTensorReplayBuffer,
    TensorReplayBuffer,
    infer_specs_from_metaworld,
    load_metaworld_dataset_to_replay_buffer,
)


class _FixedActionPolicy:
    def __init__(self, action=(0.5, -0.5)):
        self._action = np.asarray(action, dtype=np.float32)

    def get_action(self, obs):
        return self._action


class _FakeTaskEnv:
    """Scripted rollout: succeeds iff constructed with a seed that is even.
    3 steps per episode, terminates on the last step (mirrors
    terminate_on_success=True)."""

    def __init__(self, env_name=None, seed=None, terminate_on_success=False, **kwargs):
        self._seed = seed
        self._t = 0
        self._length = 3
        self.observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.full(3, float(self._t), dtype=np.float64), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self._length
        succeeds = (self._seed or 0) % 2 == 0
        info = {"success": 1.0 if (terminated and succeeds) else 0.0}
        obs = np.full(3, float(self._t), dtype=np.float64)
        return obs, 0.0, terminated, False, info

    def close(self):
        pass


def _install_fake_metaworld(monkeypatch, *, policy_action=(0.5, -0.5)):
    fake_metaworld = types.ModuleType("metaworld")
    fake_policies = types.ModuleType("metaworld.policies")
    fake_policies.ENV_POLICY_MAP = {"reach-v3": lambda: _FixedActionPolicy(policy_action)}
    fake_metaworld.policies = fake_policies

    monkeypatch.setitem(sys.modules, "metaworld", fake_metaworld)
    monkeypatch.setitem(sys.modules, "metaworld.policies", fake_policies)
    monkeypatch.setattr("gymnasium.make", lambda env_id, **kwargs: _FakeTaskEnv(**kwargs))


def test_infer_specs_from_metaworld_reads_env_spaces(monkeypatch):
    _install_fake_metaworld(monkeypatch)

    obs_space, action_space = infer_specs_from_metaworld("reach-v3")

    assert obs_space.shape == (3,)
    assert action_space.shape == (2,)


def test_load_metaworld_dataset_retries_until_success(monkeypatch):
    _install_fake_metaworld(monkeypatch)

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        num_envs=1,
        buffer_size=20,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_metaworld_dataset_to_replay_buffer(buffer, "reach-v3", num_traj=2)

    # 2 successful demos x 3 transitions each = 6 (odd-seed failed attempts
    # are skipped and don't contribute transitions).
    assert loaded == 6
    assert buffer.actions[0, 0].tolist() == [0.5, -0.5]


def test_load_metaworld_dataset_marks_reward_and_done_only_at_last_step(monkeypatch):
    _install_fake_metaworld(monkeypatch)

    buffer = MCTensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        num_envs=1,
        buffer_size=20,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_metaworld_dataset_to_replay_buffer(buffer, "reach-v3", num_traj=1)

    assert loaded == 3
    assert buffer.dones[:3, 0].tolist() == [0.0, 0.0, 1.0]
    assert buffer._episode_end[:3, 0].tolist() == [False, False, True]


def test_load_metaworld_dataset_applies_reward_scale_and_bias(monkeypatch):
    _install_fake_metaworld(monkeypatch)

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        num_envs=1,
        buffer_size=20,
        storage_device="cpu",
        sample_device="cpu",
    )

    load_metaworld_dataset_to_replay_buffer(
        buffer, "reach-v3", num_traj=1, reward_scale=2.0, reward_bias=1.0
    )

    assert buffer.rewards[:3, 0].tolist() == [1.0, 1.0, 3.0]  # 0*2+1, 0*2+1, 1*2+1
