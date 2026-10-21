"""_IsaacLabVecEnvAdapter against a duck-typed fake env (no Isaac Lab needed)."""
from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_garden.envs.isaaclab.env import _IsaacLabVecEnvAdapter

NUM_ENVS = 2


class _FakeIsaacLabEnv:
    def __init__(self, obs: dict) -> None:
        self._obs = obs
        self.unwrapped = SimpleNamespace(
            num_envs=NUM_ENVS,
            device="cpu",
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(NUM_ENVS, 1), dtype=np.float32),
        )

    def reset(self, seed=None, options=None):
        return self._obs, {}

    def step(self, actions):
        zeros = torch.zeros(NUM_ENVS, dtype=torch.bool)
        return self._obs, torch.zeros(NUM_ENVS), zeros, zeros, {}


def _state() -> torch.Tensor:
    return torch.randn(NUM_ENVS, 4)


def test_nonzero_rgb_and_state_build_dict_space() -> None:
    rgb = torch.zeros(NUM_ENVS, 64, 64, 3, dtype=torch.uint8)
    rgb[0, 10, 10] = 200
    adapter = _IsaacLabVecEnvAdapter(
        _FakeIsaacLabEnv({"rgb_front": rgb, "state": _state()}), seed=0, is_visual=True
    )

    space = adapter.single_observation_space
    assert set(space.spaces) == {"rgb_front", "state"}
    assert space["rgb_front"].shape == (64, 64, 3) and space["rgb_front"].dtype == np.uint8
    assert space["state"].shape == (4,) and space["state"].dtype == np.float32
    assert adapter.single_action_space.shape == (1,)


def test_all_zero_rgb_raises() -> None:
    rgb = torch.zeros(NUM_ENVS, 64, 64, 3, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="zero frames"):
        _IsaacLabVecEnvAdapter(
            _FakeIsaacLabEnv({"rgb_front": rgb, "state": _state()}), seed=0, is_visual=True
        )


def test_state_only_obs_unaffected() -> None:
    adapter = _IsaacLabVecEnvAdapter(_FakeIsaacLabEnv({"policy": torch.zeros(NUM_ENVS, 4)}), seed=0)

    assert set(adapter.single_observation_space.spaces) == {"state"}
