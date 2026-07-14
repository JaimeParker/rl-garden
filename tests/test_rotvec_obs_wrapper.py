"""RotvecObsWrapper: converts FrankaRealEnv's quaternion state slice into a
rotation vector, against a fake bridge client (no HTTP/ROS/hardware)."""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from rl_garden.envs.franka_real.config import FrankaRealEnvConfig
from rl_garden.envs.franka_real.env import FrankaRealEnv
from rl_garden.envs.wrappers.rotvec_obs import RotvecObsWrapper

from test_franka_real_env import _FakeBridgeClient


def _make_wrapped_env(initial_pose=None) -> RotvecObsWrapper:
    bridge = _FakeBridgeClient(initial_pose=initial_pose)
    cfg = FrankaRealEnvConfig(bridge_url="http://unused")
    env = FrankaRealEnv(cfg, bridge_client=bridge, camera_capture=None)
    return RotvecObsWrapper(env)


def test_observation_space_shrinks_state_by_one():
    env = _make_wrapped_env()
    assert env.single_observation_space["state"].shape == (19,)
    assert env.observation_space["state"].shape == (1, 19)


def test_reset_converts_quat_to_rotvec():
    quat = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()
    pose = [0.5, 0.0, 0.2, *quat]
    env = _make_wrapped_env(initial_pose=pose)

    obs, _ = env.reset()
    assert obs["state"].shape == (1, 19)
    np.testing.assert_allclose(obs["state"][0, :3].numpy(), [0.5, 0.0, 0.2], atol=1e-6)
    expected_rotvec = Rotation.from_quat(quat).as_rotvec()
    np.testing.assert_allclose(obs["state"][0, 3:6].numpy(), expected_rotvec, atol=1e-6)
    # rest of the state (vel/gripper/force/torque) passes through unchanged.
    np.testing.assert_allclose(obs["state"][0, 6:].numpy(), np.zeros(13), atol=1e-6)


def test_step_converts_quat_to_rotvec():
    env = _make_wrapped_env()
    env.reset()
    obs, _, _, _, _ = env.step(torch.zeros(1, 7))
    assert obs["state"].shape == (1, 19)
    np.testing.assert_allclose(obs["state"][0, 3:6].numpy(), [0.0, 0.0, 0.0], atol=1e-6)
