from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.envs.wrappers import ImageFrameStackWrapper


class _FakeBatchedEnv(gym.Env):
    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self._value = torch.zeros(num_envs, dtype=torch.uint8)
        self._init_raw_obs = self._obs()
        self.single_action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.update_obs_space(self._init_raw_obs)

    def _obs(self):
        rgb = self._value[:, None, None, None].expand(-1, 2, 2, 3).clone()
        state = self._value[:, None].float()
        return {"rgb_base_camera": rgb, "state": state}

    def update_obs_space(self, obs):
        self._init_raw_obs = obs
        self.single_observation_space = spaces.Dict(
            {
                key: spaces.Box(
                    low=0,
                    high=255 if value.dtype == torch.uint8 else np.inf,
                    shape=tuple(value.shape[1:]),
                    dtype=np.uint8 if value.dtype == torch.uint8 else np.float32,
                )
                for key, value in obs.items()
            }
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

    def reset(self, *, seed=None, options=None):
        del seed
        if options is None or "env_idx" not in options:
            self._value.zero_()
        else:
            self._value[options["env_idx"]] = 100
        return self._obs(), {}

    def step(self, action):
        del action
        self._value += 1
        zeros = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), self._value.float(), zeros, zeros, {}


def test_image_frame_stack_repeats_initial_frame_and_preserves_state_shape():
    env = ImageFrameStackWrapper(_FakeBatchedEnv(), frame_stack=3)
    obs, _ = env.reset()

    assert obs["rgb_base_camera"].shape == (2, 3, 2, 2, 3)
    assert obs["state"].shape == (2, 1)
    single_space = env.get_wrapper_attr("single_observation_space")
    assert single_space["rgb_base_camera"].shape == (3, 2, 2, 3)
    assert torch.equal(obs["rgb_base_camera"][:, 0], obs["rgb_base_camera"][:, 2])


def test_image_frame_stack_shifts_without_mutating_previous_observation():
    env = ImageFrameStackWrapper(_FakeBatchedEnv(), frame_stack=3)
    previous, _ = env.reset()
    previous_rgb = previous["rgb_base_camera"].clone()
    current, *_ = env.step(torch.zeros(2, 1))

    assert torch.equal(previous["rgb_base_camera"], previous_rgb)
    assert torch.all(current["rgb_base_camera"][:, :2] == 0)
    assert torch.all(current["rgb_base_camera"][:, 2] == 1)


class _FakeFrankaRealEnv(gym.Env):
    """Shaped like FrankaRealEnv: no _init_raw_obs, no update_obs_space,
    Dict obs space with a non-rgb/depth-prefixed camera key -- exercises the
    generic (image_keys=...) path."""

    num_envs = 1

    def __init__(self) -> None:
        self.single_observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "wrist_cam": spaces.Box(0, 255, (2, 2, 3), np.uint8),
            }
        )
        self.single_action_space = spaces.Box(-1, 1, (7,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.action_space = batch_space(self.single_action_space, 1)
        self.reset_calls = 0
        self._value = torch.zeros(1, dtype=torch.uint8)

    def _obs(self):
        rgb = self._value[:, None, None, None].expand(-1, 2, 2, 3).clone()
        state = self._value[:, None].float().expand(-1, 4).clone()
        return {"wrist_cam": rgb, "state": state}

    def reset(self, *, seed=None, options=None):
        self.reset_calls += 1
        self._value.zero_()
        return self._obs(), {}

    def step(self, action):
        del action
        self._value += 1
        zeros = torch.zeros(1, dtype=torch.bool)
        return self._obs(), self._value.float(), zeros, zeros, {}


def test_generic_image_keys_path_does_not_call_reset_at_construction():
    fake_env = _FakeFrankaRealEnv()
    ImageFrameStackWrapper(fake_env, frame_stack=3, image_keys=("wrist_cam",))
    assert fake_env.reset_calls == 0


def test_generic_image_keys_path_reports_stacked_observation_space():
    wrapped = ImageFrameStackWrapper(
        _FakeFrankaRealEnv(), frame_stack=3, image_keys=("wrist_cam",)
    )
    assert wrapped.single_observation_space["wrist_cam"].shape == (3, 2, 2, 3)
    assert wrapped.single_observation_space["state"].shape == (4,)
    assert wrapped.observation_space["wrist_cam"].shape == (1, 3, 2, 2, 3)


def test_generic_image_keys_path_forwards_other_attributes():
    fake_env = _FakeFrankaRealEnv()
    wrapped = ImageFrameStackWrapper(fake_env, frame_stack=3, image_keys=("wrist_cam",))
    assert wrapped.num_envs == 1
    assert wrapped.single_action_space.shape == (7,)


def test_generic_image_keys_path_reset_and_step_produce_correct_stacking():
    wrapped = ImageFrameStackWrapper(
        _FakeFrankaRealEnv(), frame_stack=3, image_keys=("wrist_cam",)
    )
    obs, _ = wrapped.reset()
    assert obs["wrist_cam"].shape == (1, 3, 2, 2, 3)
    assert torch.equal(obs["wrist_cam"][:, 0], obs["wrist_cam"][:, 2])

    current, *_ = wrapped.step(torch.zeros(1, 7))
    assert torch.all(current["wrist_cam"][:, :2] == 0)
    assert torch.all(current["wrist_cam"][:, 2] == 1)


def test_empty_image_keys_raises():
    with pytest.raises(ValueError, match="requires image observations"):
        ImageFrameStackWrapper(_FakeFrankaRealEnv(), frame_stack=3, image_keys=())


def test_image_frame_stack_partial_reset_only_replaces_selected_history():
    env = ImageFrameStackWrapper(_FakeBatchedEnv(), frame_stack=3)
    env.reset()
    terminal, *_ = env.step(torch.zeros(2, 1))
    terminal_rgb = terminal["rgb_base_camera"].clone()

    reset_obs, _ = env.reset(options={"env_idx": torch.tensor([0])})

    assert torch.equal(terminal["rgb_base_camera"], terminal_rgb)
    assert torch.all(reset_obs["rgb_base_camera"][0] == 100)
    assert torch.equal(reset_obs["rgb_base_camera"][1], terminal_rgb[1])
