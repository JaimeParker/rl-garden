from __future__ import annotations

import json

import h5py
import numpy as np

from rl_garden.models.act.robotwin_dataset import (
    ROBOTWIN_ACT_CAMERA_NAMES,
    RoboTwinACTDataset,
    compute_delta_ee_actions,
    load_official_robotwin_episode,
    write_act_h5,
)


def _quat_identity(n: int) -> np.ndarray:
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    return quat


def _write_official_episode(path):
    with h5py.File(path, "w") as root:
        joint = root.create_group("joint_action")
        joint.create_dataset("vector", data=np.zeros((3, 14), dtype=np.float32))
        endpose = root.create_group("endpose")
        xyz = np.asarray([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.03, 0.0, 0.0]], dtype=np.float32)
        endpose.create_dataset("left_endpose", data=np.concatenate([xyz, _quat_identity(3)], axis=1))
        endpose.create_dataset("right_endpose", data=np.concatenate([-xyz, _quat_identity(3)], axis=1))
        endpose.create_dataset("left_gripper", data=np.asarray([0.2, 0.4, 0.4], dtype=np.float32))
        endpose.create_dataset("right_gripper", data=np.asarray([0.8, 0.6, 0.6], dtype=np.float32))
        obs = root.create_group("observation")
        for camera_idx, camera_name in enumerate(ROBOTWIN_ACT_CAMERA_NAMES):
            rgb = np.full((3, 8, 8, 3), camera_idx, dtype=np.uint8)
            obs.create_group(camera_name).create_dataset("rgb", data=rgb, dtype=np.uint8)


def test_compute_delta_ee_actions_scales_and_clips() -> None:
    left = np.concatenate(
        [
            np.asarray([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0]], dtype=np.float32),
            _quat_identity(2),
        ],
        axis=1,
    )
    right = left.copy()

    actions, stats = compute_delta_ee_actions(
        left,
        np.asarray([0.0, 0.5], dtype=np.float32),
        right,
        np.asarray([1.0, 0.5], dtype=np.float32),
        ee_delta_pos_scale=0.03,
        gripper_delta_scale=0.2,
    )

    assert actions.shape == (1, 14)
    assert actions[0, 0] == 1.0
    assert actions[0, 6] == 1.0
    assert actions[0, 13] == -1.0
    assert stats["action_abs_max_before_clip"] > 1.0
    assert stats["action_clip_fraction"] > 0.0


def test_official_robotwin_episode_converts_to_act_dataset(tmp_path) -> None:
    official_path = tmp_path / "episode0.hdf5"
    converted_path = tmp_path / "open_laptop_delta_ee.h5"
    _write_official_episode(official_path)

    episode = load_official_robotwin_episode(
        official_path,
        camera_names=ROBOTWIN_ACT_CAMERA_NAMES,
        image_size=(8, 8),
    )
    metadata = write_act_h5(
        converted_path,
        [episode],
        task_name="open_laptop",
        task_config="demo_clean",
        camera_names=ROBOTWIN_ACT_CAMERA_NAMES,
    )

    assert metadata["num_episodes"] == 1
    assert metadata["num_transitions"] == 2
    saved_metadata = json.loads(converted_path.with_suffix(".json").read_text())
    assert saved_metadata["env_info"]["env_kwargs"]["control_mode"] == "delta_ee"

    dataset = RoboTwinACTDataset(
        converted_path,
        num_queries=3,
        camera_names=ROBOTWIN_ACT_CAMERA_NAMES,
        image_size=(8, 8),
        control_mode="delta_ee",
    )
    first = dataset[0]
    assert first["observations"]["rgb"].shape == (3, 3, 8, 8)
    assert first["actions"].shape == (3, 14)
    second = dataset[1]
    np.testing.assert_allclose(second["actions"][1:].numpy(), np.zeros((2, 14)))
