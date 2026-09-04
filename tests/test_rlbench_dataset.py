"""Tests for building RLBench obs/actions and loading RLBench demo datasets.

Fakes the entire ``rlbench`` package tree in ``sys.modules`` (every dotted
level, not just the leaf) so these tests need no real ``pyrep``/CoppeliaSim
install -- same monkeypatch-the-module strategy ``test_ogbench_dataset.py``
uses, just deeper since RLBench's own package layout is more nested.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from gymnasium import spaces

from rl_garden.buffers import (
    MCTensorReplayBuffer,
    TensorReplayBuffer,
    infer_specs_from_rlbench,
    load_rlbench_dataset_to_replay_buffer,
)

_CAMERAS: tuple[str, ...] = ("left_shoulder", "right_shoulder", "overhead", "wrist", "front")


class _FakeObservation:
    def __init__(self, low_dim, joint_velocities, gripper_open, images: dict | None = None):
        self._low_dim = np.asarray(low_dim, dtype=np.float32)
        self.joint_velocities = np.asarray(joint_velocities, dtype=np.float32)
        self.gripper_open = gripper_open
        self._images = images or {}

    def get_low_dim_data(self):
        return self._low_dim

    def __getattr__(self, name):
        if name in self._images:
            return self._images[name]
        raise AttributeError(name)


class _FakeCameraConfig:
    def __init__(self):
        self.rgb = True
        self.depth = True
        self.point_cloud = True
        self.mask = True
        self.image_size = (128, 128)

    def set_all(self, value):
        self.rgb = value
        self.depth = value
        self.point_cloud = value
        self.mask = value


class _FakeObservationConfig:
    def __init__(self):
        for camera in _CAMERAS:
            setattr(self, f"{camera}_camera", _FakeCameraConfig())
        self.joint_velocities = True
        self.joint_positions = True
        self.joint_forces = True
        self.gripper_open = True
        self.gripper_pose = True
        self.gripper_joint_positions = True
        self.gripper_touch_forces = True
        self.task_low_dim_state = True

    def set_all_low_dim(self, value):
        self.joint_velocities = value
        self.joint_positions = value
        self.joint_forces = value
        self.gripper_open = value
        self.gripper_pose = value
        self.gripper_joint_positions = value
        self.gripper_touch_forces = value
        self.task_low_dim_state = value

    def set_all_high_dim(self, value):
        for camera in _CAMERAS:
            getattr(self, f"{camera}_camera").set_all(value)


def _two_demos(*, with_images: bool = False) -> list[list[_FakeObservation]]:
    def _obs(i, low_dim_dim=3, gripper_open=1.0, images=False):
        images_dict = None
        if images:
            images_dict = {}
            for camera in _CAMERAS:
                images_dict[f"{camera}_rgb"] = np.full((4, 4, 3), i, dtype=np.uint8)
                images_dict[f"{camera}_depth"] = np.full((4, 4), float(i), dtype=np.float32)
        return _FakeObservation(
            low_dim=[float(i)] * low_dim_dim,
            joint_velocities=[float(i)] * 7,
            gripper_open=gripper_open,
            images=images_dict,
        )

    demo1 = [_obs(i, images=with_images) for i in range(3)]  # length 3 -> 2 transitions
    demo2 = [_obs(i, images=with_images) for i in range(2)]  # length 2 -> 1 transition
    return [demo1, demo2]


def _install_fake_rlbench(monkeypatch, demos: list[list[_FakeObservation]]):
    fake_rlbench = types.ModuleType("rlbench")
    fake_utils = types.ModuleType("rlbench.utils")
    fake_observation_config = types.ModuleType("rlbench.observation_config")
    fake_environment = types.ModuleType("rlbench.environment")
    fake_action_modes = types.ModuleType("rlbench.action_modes")
    fake_action_mode = types.ModuleType("rlbench.action_modes.action_mode")
    fake_arm_action_modes = types.ModuleType("rlbench.action_modes.arm_action_modes")
    fake_gripper_action_modes = types.ModuleType("rlbench.action_modes.gripper_action_modes")

    captured = {}

    def _get_stored_demos(amount, image_paths, dataset_root, variation_number, task_name, obs_config, **kwargs):
        captured["dataset_root"] = dataset_root
        captured["task_name"] = task_name
        captured["amount"] = amount
        if amount == -1:
            return demos
        return demos[:amount]

    def _name_to_task_class(task_name):
        captured["live_task_name"] = task_name
        return object

    fake_utils.get_stored_demos = _get_stored_demos
    fake_utils.name_to_task_class = _name_to_task_class
    fake_observation_config.ObservationConfig = _FakeObservationConfig

    class _FakeMoveArmThenGripper:
        def __init__(self, arm_action_mode=None, gripper_action_mode=None):
            self.arm_action_mode = arm_action_mode
            self.gripper_action_mode = gripper_action_mode

        def action_bounds(self):
            return np.array([-1.0] * 8), np.array([1.0] * 8)

    class _FakeJointVelocity:
        pass

    class _FakeDiscrete:
        pass

    class _FakeEnvironment:
        def __init__(self, *, action_mode, obs_config, headless):
            self.action_mode = action_mode
            self.obs_config = obs_config
            self.headless = headless
            self.action_shape = (8,)
            self.launched = False
            self.shutdown_called = False

        def launch(self):
            self.launched = True

        def get_task(self, task_class):
            return _FakeTaskEnv(demos, captured)

        def shutdown(self):
            self.shutdown_called = True

    fake_action_mode.MoveArmThenGripper = _FakeMoveArmThenGripper
    fake_arm_action_modes.JointVelocity = _FakeJointVelocity
    fake_gripper_action_modes.Discrete = _FakeDiscrete
    fake_environment.Environment = _FakeEnvironment

    fake_rlbench.utils = fake_utils
    fake_rlbench.observation_config = fake_observation_config
    fake_rlbench.environment = fake_environment
    fake_rlbench.action_modes = fake_action_modes

    monkeypatch.setitem(sys.modules, "rlbench", fake_rlbench)
    monkeypatch.setitem(sys.modules, "rlbench.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "rlbench.observation_config", fake_observation_config)
    monkeypatch.setitem(sys.modules, "rlbench.environment", fake_environment)
    monkeypatch.setitem(sys.modules, "rlbench.action_modes", fake_action_modes)
    monkeypatch.setitem(sys.modules, "rlbench.action_modes.action_mode", fake_action_mode)
    monkeypatch.setitem(sys.modules, "rlbench.action_modes.arm_action_modes", fake_arm_action_modes)
    monkeypatch.setitem(sys.modules, "rlbench.action_modes.gripper_action_modes", fake_gripper_action_modes)
    return captured


class _FakeTaskEnv:
    def __init__(self, demos, captured):
        self._demos = demos
        self._captured = captured

    def get_demos(self, amount, live_demos=False):
        self._captured["live_demos_called"] = live_demos
        self._captured["live_amount"] = amount
        return self._demos[:amount]


def test_infer_specs_from_rlbench_state_only(monkeypatch):
    demos = _two_demos()
    _install_fake_rlbench(monkeypatch, demos)

    obs_space, action_space = infer_specs_from_rlbench("/data/rlbench_demos/reach_target")

    assert isinstance(obs_space, spaces.Box)
    assert obs_space.dtype == np.float32
    assert obs_space.shape == (3,)
    assert action_space.shape == (8,)  # 7 joint velocities + 1 discretized gripper


def test_infer_specs_from_rlbench_splits_dataset_root_and_task_name(monkeypatch):
    demos = _two_demos()
    captured = _install_fake_rlbench(monkeypatch, demos)

    infer_specs_from_rlbench("/data/rlbench_demos/reach_target")

    assert captured["dataset_root"] == "/data/rlbench_demos"
    assert captured["task_name"] == "reach_target"


def test_infer_specs_from_rlbench_rgb_mode_renames_image_keys(monkeypatch):
    demos = _two_demos(with_images=True)
    _install_fake_rlbench(monkeypatch, demos)

    obs_space, _ = infer_specs_from_rlbench(
        "/data/rlbench_demos/reach_target", obs_mode="rgb", cameras=("front",)
    )

    assert set(obs_space.spaces) == {"state", "rgb_front", "depth_front"}
    assert obs_space["rgb_front"].dtype == np.uint8
    assert obs_space["rgb_front"].shape == (4, 4, 3)
    assert obs_space["depth_front"].dtype == np.float32
    assert obs_space["depth_front"].shape == (4, 4, 1)


def test_load_rlbench_dataset_uses_single_observation_action_derivation(monkeypatch):
    demos = _two_demos()
    _install_fake_rlbench(monkeypatch, demos)

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_rlbench_dataset_to_replay_buffer(buffer, "/data/rlbench_demos/reach_target")

    # demo1 (len 3) -> 2 transitions, demo2 (len 2) -> 1 transition.
    assert loaded == 3
    # Action for transition i is derived from demo[i] itself (joint_velocities
    # + discretized gripper_open), not from demo[i+1].
    assert buffer.actions[0, 0].tolist() == [0.0] * 7 + [1.0]
    assert buffer.actions[1, 0].tolist() == [1.0] * 7 + [1.0]


def test_load_rlbench_dataset_marks_reward_and_done_only_at_last_step(monkeypatch):
    demos = _two_demos()
    _install_fake_rlbench(monkeypatch, demos)

    buffer = MCTensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_rlbench_dataset_to_replay_buffer(buffer, "/data/rlbench_demos/reach_target")

    assert loaded == 3
    # demo1's 2 transitions: done only at its own last step; demo2's 1
    # transition: done at its (only, hence last) step.
    assert buffer.dones[:3, 0].tolist() == [0.0, 1.0, 1.0]
    assert buffer._episode_end[:3, 0].tolist() == [False, True, True]


def test_load_rlbench_dataset_num_traj_truncates_by_demo_count(monkeypatch):
    demos = _two_demos()
    _install_fake_rlbench(monkeypatch, demos)

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_rlbench_dataset_to_replay_buffer(
        buffer, "/data/rlbench_demos/reach_target", num_traj=1
    )

    assert loaded == 2  # demo1 only (2 transitions), not demo2's extra 1.


def test_load_rlbench_dataset_live_demos_routes_to_get_demos_live(monkeypatch):
    demos = _two_demos()
    captured = _install_fake_rlbench(monkeypatch, demos)

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_rlbench_dataset_to_replay_buffer(
        buffer, "/data/rlbench_demos/reach_target", live_demos=True, num_traj=1
    )

    assert captured["live_demos_called"] is True
    assert captured["live_amount"] == 1
    assert captured["live_task_name"] == "reach_target"
    assert loaded == 2


def test_load_rlbench_dataset_raises_when_no_demos_found(monkeypatch):
    _install_fake_rlbench(monkeypatch, [])

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    with pytest.raises(ValueError, match="No usable RLBench demos"):
        load_rlbench_dataset_to_replay_buffer(buffer, "/data/rlbench_demos/reach_target")
