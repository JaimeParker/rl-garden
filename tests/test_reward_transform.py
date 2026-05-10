"""Tests for RewardScaleBiasWrapper and the H5 loader's reward transform."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasWrapper


class _DummyEnv(gym.Env):
    """Tiny env returning a fixed reward sequence for testing."""

    metadata = {"render_modes": []}

    def __init__(self, rewards):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self._rewards = list(rewards)
        self._idx = 0

    def reset(self, *, seed=None, options=None):
        self._idx = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        r = self._rewards[self._idx]
        self._idx += 1
        done = self._idx >= len(self._rewards)
        return np.zeros(2, dtype=np.float32), r, done, False, {}


def test_reward_scale_bias_wrapper_applies_transform():
    base = _DummyEnv([0.0, 1.0, 0.5])
    env = RewardScaleBiasWrapper(base, scale=2.0, bias=-0.5)
    env.reset()
    _, r0, *_ = env.step(env.action_space.sample())
    _, r1, *_ = env.step(env.action_space.sample())
    _, r2, *_ = env.step(env.action_space.sample())
    assert r0 == pytest.approx(2.0 * 0.0 - 0.5)
    assert r1 == pytest.approx(2.0 * 1.0 - 0.5)
    assert r2 == pytest.approx(2.0 * 0.5 - 0.5)


def test_reward_scale_bias_wrapper_identity_default():
    base = _DummyEnv([0.7, -0.3])
    env = RewardScaleBiasWrapper(base)
    env.reset()
    _, r0, *_ = env.step(env.action_space.sample())
    _, r1, *_ = env.step(env.action_space.sample())
    assert r0 == pytest.approx(0.7)
    assert r1 == pytest.approx(-0.3)


def test_reward_scale_bias_wrapper_works_with_tensor_reward():
    """ManiSkill returns torch-tensor rewards; the wrapper must pass them through."""

    class _TensorRewardEnv(gym.Env):
        observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            return (
                np.zeros(2, dtype=np.float32),
                torch.tensor([1.0, 2.0]),
                False,
                False,
                {},
            )

    env = RewardScaleBiasWrapper(_TensorRewardEnv(), scale=0.5, bias=1.0)
    env.reset()
    _, r, *_ = env.step(env.action_space.sample())
    assert isinstance(r, torch.Tensor)
    torch.testing.assert_close(r, torch.tensor([1.5, 2.0]))


def test_h5_loader_applies_reward_scale_bias(tmp_path: Path):
    # Build a tiny H5 with one trajectory and verify the loaded buffer reflects
    # the scaled rewards.
    h5py = pytest.importorskip("h5py")
    from rl_garden.buffers.mc_buffer import MCTensorReplayBuffer
    from rl_garden.buffers.maniskill_h5 import load_maniskill_h5_to_replay_buffer

    path = tmp_path / "tiny.h5"
    obs = np.zeros((4, 3), dtype=np.float32)  # 3 transitions, 4 obs frames
    actions = np.zeros((3, 2), dtype=np.float32)
    rewards = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    terminated = np.array([False, False, True])
    truncated = np.array([False, False, False])
    with h5py.File(path, "w") as f:
        traj = f.create_group("traj_0")
        traj.create_dataset("obs", data=obs)
        traj.create_dataset("actions", data=actions)
        traj.create_dataset("rewards", data=rewards)
        traj.create_dataset("terminated", data=terminated)
        traj.create_dataset("truncated", data=truncated)

    buf = MCTensorReplayBuffer(
        observation_space=spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    n = load_maniskill_h5_to_replay_buffer(
        buf, path, reward_scale=2.0, reward_bias=-0.5
    )
    assert n == 3
    expected = torch.tensor([2.0 * 1.0 - 0.5, 2.0 * 0.0 - 0.5, 2.0 * 1.0 - 0.5])
    torch.testing.assert_close(buf.rewards[:3, 0], expected)
