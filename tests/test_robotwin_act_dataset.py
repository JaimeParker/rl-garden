from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from rl_garden.models.act.robotwin_dataset import (
    ROBOTWIN_ACT_CAMERA_NAMES,
    RoboTwinACTDataset,
    compute_delta_ee_actions,
    load_official_robotwin_episode,
    write_act_h5,
    RobotWinEpisode,
    subsample_episode,
    pose_to_matrix,
    matrix_to_rotvec,
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


def _aligned_episode(length):
    t = np.arange(length, dtype=np.float32)
    # Changing rotation axes distinguish relative transforms from vector addition.
    q = _quat_identity(length)
    for i in range(length):
        angle = i * .025
        q[i, 0] = np.cos(angle / 2)
        q[i, 1 + i % 3] = np.sin(angle / 2)
    poses = np.c_[t * .003, t * .002, t * .001, q].astype(np.float32)
    return RobotWinEpisode(states=np.repeat(t[:, None], 14, axis=1),
        left_endpose=poses, right_endpose=poses.copy(),
        left_gripper=t * .01, right_gripper=t * .02,
        images={name: np.broadcast_to(t[:, None, None, None] + j * 10,
                    (length, 8, 8, 3)).astype(np.uint8)
                for j, name in enumerate(ROBOTWIN_ACT_CAMERA_NAMES)})


@pytest.mark.parametrize('length,stride,expected', [
    (2, 2, [0, 1]), (3, 2, [0, 2]), (4, 2, [0, 2, 3]),
    (5, 2, [0, 2, 4]), (5, 1, [0, 1, 2, 3, 4]), (5, 10, [0, 4])])
def test_synchronized_sampling(length, stride, expected):
    original = _aligned_episode(length)
    sampled, indices = subsample_episode(original, stride)
    np.testing.assert_array_equal(indices, expected)
    for name in ('states', 'left_endpose', 'left_gripper', 'right_endpose', 'right_gripper'):
        np.testing.assert_array_equal(getattr(sampled, name), getattr(original, name)[expected])
    for name in original.images:
        np.testing.assert_array_equal(sampled.images[name], original.images[name][expected])


@pytest.mark.parametrize('stride', [0, -1, 1.5, True])
def test_invalid_stride_rejected_before_output(tmp_path, stride):
    path = tmp_path / 'invalid.h5'
    with pytest.raises(ValueError):
        write_act_h5(path, [_aligned_episode(4)], task_name='open_laptop',
                     task_config=None, frame_stride=stride)
    assert not path.exists()


def test_misaligned_episode_rejected():
    original = _aligned_episode(4)
    original.images['head_camera'] = original.images['head_camera'][:-1]
    with pytest.raises(ValueError, match='matching lengths'):
        subsample_episode(original, 2)


@pytest.mark.parametrize('stride', [1, 2])
def test_export_recomputes_endpoint_actions_and_loader_alignment(tmp_path, stride):
    original = _aligned_episode(6)
    path = tmp_path / 'sampled.h5'
    meta = write_act_h5(path, [original], task_name='open_laptop', task_config=None,
        frame_stride=stride, ee_delta_pos_scale=.06, ee_delta_rot_scale=.30)
    _, indices = subsample_episode(original, stride)
    with h5py.File(path) as root:
        traj = root['traj_0']
        np.testing.assert_array_equal(traj['source_frame_indices'], indices)
        actions = traj['actions'][()]
        for i, (a, b) in enumerate(zip(indices[:-1], indices[1:])):
            transform = np.linalg.inv(pose_to_matrix(original.left_endpose[a])) @ pose_to_matrix(original.left_endpose[b])
            np.testing.assert_allclose(actions[i, :3] * .06, transform[:3, 3], atol=1e-7)
            np.testing.assert_allclose(actions[i, 3:6] * .30, matrix_to_rotvec(transform[:3, :3]), atol=1e-7)
            np.testing.assert_allclose(actions[i, 6] * .2,
                original.left_gripper[b] - original.left_gripper[a], atol=1e-7)
        if stride == 1:
            expected, _ = compute_delta_ee_actions(original.left_endpose, original.left_gripper,
                original.right_endpose, original.right_gripper, ee_delta_pos_scale=.06, ee_delta_rot_scale=.30)
            np.testing.assert_array_equal(actions, expected)
    assert meta['sampling']['frame_stride'] == stride
    assert meta['episodes'][0]['terminal_short_interval'] == bool(5 % stride)
    dataset = RoboTwinACTDataset(path, num_queries=3, image_size=(8, 8))
    assert len(dataset) == len(indices) - 1
    np.testing.assert_array_equal(dataset[1]['observations']['state'].numpy(), original.states[indices[1]])


def test_converter_cli_stride_two(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    _write_official_episode(tmp_path / 'episode0.hdf5')
    output = tmp_path / 'converted.h5'
    subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'examples/convert_robotwin_act_delta_ee.py'),
        '--source-dir', str(tmp_path), '--output-path', str(output), '--conversion-mode', 'direct',
        '--frame-stride', '2', '--camera-width', '8', '--camera-height', '8',
        '--ee-delta-pos-scale', '.06', '--ee-delta-rot-scale', '.30'], check=True)
    meta = json.loads(output.with_suffix('.json').read_text())
    assert meta['num_transitions'] == 1
    assert meta['sampling']['frame_stride'] == 2
