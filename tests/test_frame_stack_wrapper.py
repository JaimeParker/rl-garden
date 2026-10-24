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


class _FakeMultiCameraEnv(gym.Env):
    """Generic-path fixture with two image keys (rgb + depth) alongside
    state -- exercises stacking multiple heterogeneous image keys at once,
    matching how mujoco_warp/robotwin/ogbench call the generic path with
    more than one ``rgb_<cam>``/``depth_<cam>`` key."""

    num_envs = 1

    def __init__(self, depth_space: spaces.Box | None = None) -> None:
        if depth_space is None:
            depth_space = spaces.Box(0, np.inf, (2, 2, 1), np.float32)
        self.single_observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "rgb_main": spaces.Box(0, 255, (2, 2, 3), np.uint8),
                "depth_main": depth_space,
            }
        )
        self.single_action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.action_space = batch_space(self.single_action_space, 1)
        self._value = torch.zeros(1, dtype=torch.uint8)

    def _obs(self):
        rgb = self._value[:, None, None, None].expand(-1, 2, 2, 3).clone()
        depth = self._value[:, None, None, None].expand(-1, 2, 2, 1).float()
        state = self._value[:, None].float().expand(-1, 4).clone()
        return {"rgb_main": rgb, "depth_main": depth, "state": state}

    def reset(self, *, seed=None, options=None):
        self._value.zero_()
        return self._obs(), {}

    def step(self, action):
        del action
        self._value += 1
        zeros = torch.zeros(1, dtype=torch.bool)
        return self._obs(), self._value.float(), zeros, zeros, {}


def test_generic_image_keys_path_stacks_multiple_image_keys_together():
    wrapped = ImageFrameStackWrapper(
        _FakeMultiCameraEnv(), frame_stack=3, image_keys=("rgb_main", "depth_main")
    )
    obs, _ = wrapped.reset()

    assert obs["rgb_main"].shape == (1, 3, 2, 2, 3)
    assert obs["depth_main"].shape == (1, 3, 2, 2, 1)
    assert obs["state"].shape == (1, 4)
    assert torch.equal(obs["rgb_main"][:, 0], obs["rgb_main"][:, 2])
    assert torch.equal(obs["depth_main"][:, 0], obs["depth_main"][:, 2])

    current, *_ = wrapped.step(torch.zeros(1, 2))
    assert torch.all(current["rgb_main"][:, :2] == 0)
    assert torch.all(current["rgb_main"][:, 2] == 1)
    assert torch.all(current["depth_main"][:, :2] == 0)
    assert torch.all(current["depth_main"][:, 2] == 1)
    # state is never stacked, even with multiple image keys present.
    assert current["state"].shape == (1, 4)


@pytest.mark.parametrize("depth_high", [1.0, np.inf])
def test_generic_image_keys_path_keeps_each_key_bounds(depth_high):
    depth_space = spaces.Box(0.0, depth_high, (2, 2, 1), np.float32)
    wrapped = ImageFrameStackWrapper(
        _FakeMultiCameraEnv(depth_space=depth_space),
        frame_stack=3,
        image_keys=("rgb_main", "depth_main"),
    )

    depth = wrapped.single_observation_space["depth_main"]
    assert depth.shape == (3, 2, 2, 1)
    assert depth.dtype == np.float32
    assert np.all(depth.low == 0.0)
    assert np.all(depth.high == depth_high)

    rgb = wrapped.single_observation_space["rgb_main"]
    assert rgb.shape == (3, 2, 2, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb.low == 0)
    assert np.all(rgb.high == 255)
    assert np.all(wrapped.observation_space["depth_main"].high == depth_high)


def test_image_frame_stack_partial_reset_only_replaces_selected_history():
    env = ImageFrameStackWrapper(_FakeBatchedEnv(), frame_stack=3)
    env.reset()
    terminal, *_ = env.step(torch.zeros(2, 1))
    terminal_rgb = terminal["rgb_base_camera"].clone()

    reset_obs, _ = env.reset(options={"env_idx": torch.tensor([0])})

    assert torch.equal(terminal["rgb_base_camera"], terminal_rgb)
    assert torch.all(reset_obs["rgb_base_camera"][0] == 100)
    assert torch.equal(reset_obs["rgb_base_camera"][1], terminal_rgb[1])


class _FakeSameStepAutoresetEnv(gym.Env):
    """Batched env that auto-resets done envs inside ``step()`` (gymnasium
    SAME_STEP): frame value = steps since episode start, env ``i`` ends every
    ``episode_lengths[i]`` steps, and the terminal frame goes to
    ``infos["final_observation"]`` (full width, like the torch adapter)."""

    def __init__(self, episode_lengths=(2, 4)) -> None:
        self.num_envs = len(episode_lengths)
        self._episode_lengths = torch.tensor(episode_lengths)
        self._value = torch.zeros(self.num_envs, dtype=torch.uint8)
        self.single_observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, (1,), np.float32),
                "rgb_main": spaces.Box(0, 255, (2, 2, 3), np.uint8),
            }
        )
        self.single_action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.last_info: dict = {}

    def _obs(self):
        rgb = self._value[:, None, None, None].expand(-1, 2, 2, 3).clone()
        return {"rgb_main": rgb, "state": self._value[:, None].float()}

    def reset(self, *, seed=None, options=None):
        self._value.zero_()
        return self._obs(), {}

    def step(self, action):
        del action
        self._value += 1
        terminated = self._value.long() >= self._episode_lengths
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        info: dict = {}
        if terminated.any():
            info["final_observation"] = self._obs()
            info["_final_observation"] = terminated.clone()
            self._value[terminated] = 0
        self.last_info = info
        return self._obs(), self._value.float(), terminated, truncated, info


def _autoreset_stack(autoreset_in_step: bool, episode_lengths=(2, 4)):
    return ImageFrameStackWrapper(
        _FakeSameStepAutoresetEnv(episode_lengths),
        frame_stack=3,
        image_keys=("rgb_main",),
        autoreset_in_step=autoreset_in_step,
    )


def _frame_values(stacked: torch.Tensor) -> torch.Tensor:
    """(N, T, H, W, C) constant frames -> (N, T) frame values."""
    return stacked[:, :, 0, 0, 0].long()


def test_autoreset_in_step_restarts_done_history_and_rolls_others():
    env = _autoreset_stack(autoreset_in_step=True)
    env.reset()
    env.step(torch.zeros(2, 1))
    obs, _, terminated, _, _ = env.step(torch.zeros(2, 1))

    assert terminated.tolist() == [True, False]
    # env 0 just auto-reset: T copies of its new first frame (value 0);
    # env 1 keeps rolling: [0, 1, 2].
    assert _frame_values(obs["rgb_main"]).tolist() == [[0, 0, 0], [0, 1, 2]]
    assert obs["state"].shape == (2, 1)

    obs, *_ = env.step(torch.zeros(2, 1))
    assert _frame_values(obs["rgb_main"]).tolist() == [[0, 0, 1], [1, 2, 3]]


def test_autoreset_in_step_stacks_final_observation_from_previous_history():
    env = _autoreset_stack(autoreset_in_step=True)
    env.reset()
    previous, *_ = env.step(torch.zeros(2, 1))
    previous_rgb = previous["rgb_main"].clone()
    _, _, _, _, info = env.step(torch.zeros(2, 1))

    final_rgb = info["final_observation"]["rgb_main"]
    assert final_rgb.shape == (2, 3, 2, 2, 3)
    terminal_frame = torch.full((2, 2, 3), 2, dtype=torch.uint8)
    expected_env0 = torch.cat((previous_rgb[0, 1:], terminal_frame.unsqueeze(0)))
    assert torch.equal(final_rgb[0], expected_env0)
    assert _frame_values(final_rgb).tolist() == [[0, 1, 2], [0, 1, 2]]
    # Non-image keys stay single-frame.
    assert info["final_observation"]["state"].shape == (2, 1)
    assert info["_final_observation"].tolist() == [True, False]


def test_autoreset_in_step_does_not_mutate_inner_info():
    env = _autoreset_stack(autoreset_in_step=True)
    env.reset()
    env.step(torch.zeros(2, 1))
    _, _, _, _, info = env.step(torch.zeros(2, 1))

    inner_info = env.env.last_info
    assert info is not inner_info
    assert info["final_observation"] is not inner_info["final_observation"]
    assert inner_info["final_observation"]["rgb_main"].shape == (2, 2, 2, 3)


def test_autoreset_in_step_restarts_on_truncation():
    class _TruncatingEnv(_FakeSameStepAutoresetEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            return obs, reward, truncated, terminated, info

    env = ImageFrameStackWrapper(
        _TruncatingEnv((2, 4)), frame_stack=3, image_keys=("rgb_main",), autoreset_in_step=True
    )
    env.reset()
    env.step(torch.zeros(2, 1))
    obs, _, terminated, truncated, _ = env.step(torch.zeros(2, 1))

    assert truncated.tolist() == [True, False] and not terminated.any()
    assert _frame_values(obs["rgb_main"]).tolist() == [[0, 0, 0], [0, 1, 2]]


def test_autoreset_in_step_false_keeps_rolling_history_and_raw_final_observation():
    env = _autoreset_stack(autoreset_in_step=False)
    env.reset()
    env.step(torch.zeros(2, 1))
    obs, _, _, _, info = env.step(torch.zeros(2, 1))

    # Previous behavior: the done env's history keeps the old episode's frames.
    assert _frame_values(obs["rgb_main"]).tolist() == [[0, 1, 0], [0, 1, 2]]
    assert info is env.env.last_info
    assert info["final_observation"]["rgb_main"].shape == (2, 2, 2, 3)
