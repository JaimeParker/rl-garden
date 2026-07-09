"""Tests for FrankaRealEnv against a fake bridge client (no HTTP/ROS/hardware)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from rl_garden.envs.franka_real.bridge_client import FrankaBridgeClient
from rl_garden.envs.franka_real.config import FrankaRealEnvConfig
from rl_garden.envs.franka_real.env import FrankaRealEnv


class _FakeBridgeClient(FrankaBridgeClient):
    """Overrides every network call with an in-memory fake state."""

    def __init__(self, initial_pose=None):
        self.reset_calls = 0
        self.sent_poses = []
        self.gripper_calls = []
        self._pose = (
            np.array(initial_pose, dtype=np.float64)
            if initial_pose is not None
            else np.array([0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
        )

    def reset_joints(self):
        self.reset_calls += 1

    def get_state(self):
        return {
            "pose": self._pose.tolist(),
            "vel": [0.0] * 6,
            "gripper_pos": 0.0,
            "force": [0.0, 0.0, 0.0],
            "torque": [0.0, 0.0, 0.0],
        }

    def send_pose(self, pose):
        self.sent_poses.append(np.asarray(pose, dtype=np.float64))
        self._pose = np.asarray(pose, dtype=np.float64)

    def send_gripper(self, open_gripper):
        self.gripper_calls.append(bool(open_gripper))


def _make_env(camera_capture=None, **cfg_kwargs) -> tuple[FrankaRealEnv, _FakeBridgeClient]:
    bridge = _FakeBridgeClient()
    cfg = FrankaRealEnvConfig(bridge_url="http://unused", **cfg_kwargs)
    env = FrankaRealEnv(cfg, bridge_client=bridge, camera_capture=camera_capture)
    return env, bridge


def test_reset_calls_bridge_and_returns_batched_state_obs():
    env, bridge = _make_env()
    obs, info = env.reset()
    assert bridge.reset_calls == 1
    assert obs["state"].shape == (1, 20)
    assert obs["state"].dtype == torch.float32
    assert info == {}


def test_step_before_reset_raises():
    env, _ = _make_env()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(torch.zeros(1, 7))


def test_step_scales_action_into_position_delta():
    env, bridge = _make_env(safety_box_low=(-10, -10, -10), safety_box_high=(10, 10, 10))
    env.reset()
    action = torch.zeros(1, 7)
    action[0, 0] = 1.0  # +x
    env.step(action)

    sent = bridge.sent_poses[0]
    # action_scale defaults to (0.02, 0.1); starting pose x = 0.5.
    assert sent[0] == pytest.approx(0.5 + 0.02)
    assert sent[1] == pytest.approx(0.0)
    assert sent[2] == pytest.approx(0.2)


def test_step_clips_target_position_to_safety_box():
    env, bridge = _make_env(safety_box_low=(0.0, -0.1, 0.0), safety_box_high=(0.51, 0.1, 0.3))
    env.reset()
    action = torch.zeros(1, 7)
    action[0, 0] = 1.0  # would push x to 0.52, past the 0.51 box limit
    env.step(action)

    assert bridge.sent_poses[0][0] == pytest.approx(0.51)


def test_step_composes_delta_rotation_with_current_orientation():
    env, bridge = _make_env(safety_box_low=(-10, -10, -10), safety_box_high=(10, 10, 10))
    env.reset()
    action = torch.zeros(1, 7)
    action[0, 5] = 1.0  # rotation about z, scaled by rot_scale=0.1
    env.step(action)

    sent_quat = bridge.sent_poses[0][3:7]
    expected = Rotation.from_rotvec([0.0, 0.0, 0.1]) * Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(sent_quat, expected.as_quat(), atol=1e-6)


def test_step_opens_and_closes_gripper_based_on_threshold():
    env, bridge = _make_env(gripper_threshold=0.5)
    env.reset()

    open_action = torch.zeros(1, 7)
    open_action[0, 6] = 0.9
    env.step(open_action)
    assert bridge.gripper_calls[-1] is True

    close_action = torch.zeros(1, 7)
    close_action[0, 6] = -0.9
    env.step(close_action)
    assert bridge.gripper_calls[-1] is False


def test_step_truncates_after_max_episode_steps():
    env, _ = _make_env(max_episode_steps=2)
    env.reset()
    action = torch.zeros(1, 7)
    _, _, _, truncated, _ = env.step(action)
    assert not bool(truncated[0])
    _, _, _, truncated, _ = env.step(action)
    assert bool(truncated[0])


def test_camera_keys_are_included_in_observation_via_capture_callable():
    capture_calls = []

    def _capture():
        capture_calls.append(1)
        return {"wrist": np.zeros((8, 8, 3), dtype=np.uint8)}

    env, _ = _make_env(
        camera_capture=_capture, camera_keys=("wrist",), camera_height=8, camera_width=8
    )
    obs, _ = env.reset()
    assert "wrist" in obs
    assert obs["wrist"].shape == (1, 8, 8, 3)
    assert obs["wrist"].dtype == torch.uint8
    assert len(capture_calls) == 1


def test_camera_keys_without_capture_callable_raises():
    with pytest.raises(ValueError, match="camera_capture"):
        _make_env(camera_keys=("wrist",))


def test_observation_and_action_spaces_are_batch_of_one():
    env, _ = _make_env()
    assert env.num_envs == 1
    assert env.action_space.shape == (1, 7)
    assert env.observation_space["state"].shape == (1, 20)
