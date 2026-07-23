from __future__ import annotations

import numpy as np

from rl_garden.envs.robotwin.kinematics import (
    RoboTwinJointTargetFK,
    quaternion_to_rotvec_wxyz,
    quaternion_to_matrix_wxyz,
    robotwin_gripper_pose,
    rotvec_to_quaternion_wxyz,
)


class _Pose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)


class _Link:
    pass


class _Joint:
    def __init__(self, child_link=None, pose_in_child=None):
        self.child_link = child_link
        self.pose_in_child = pose_in_child or _Pose([0.0, 0.0, 0.0])

    def get_pose_in_child(self):
        return self.pose_in_child


class _Pinocchio:
    def __init__(self, left_link_index: int, right_link_index: int):
        self.left_link_index = left_link_index
        self.right_link_index = right_link_index
        self.last_qpos = None
        self.compute_calls = 0

    def compute_forward_kinematics(self, qpos):
        self.last_qpos = np.asarray(qpos).copy()
        self.compute_calls += 1

    def get_link_pose(self, link_index):
        assert self.last_qpos is not None
        if link_index == self.left_link_index:
            return _Pose([self.last_qpos[0], self.last_qpos[1], 0.0])
        assert link_index == self.right_link_index
        return _Pose([self.last_qpos[6], self.last_qpos[7], 0.0])


class _Entity:
    def __init__(self):
        self.left_link = _Link()
        self.right_link = _Link()
        self.links = [self.left_link, self.right_link]
        self.active_joints = [_Joint() for _ in range(12)]
        self.qpos = np.arange(12, dtype=np.float64) + 100.0
        self.root_pose = _Pose([1.0, 2.0, 3.0])
        self.model = _Pinocchio(0, 1)
        self.set_qpos_calls = 0

    def create_pinocchio_model(self):
        return self.model

    def get_active_joints(self):
        return self.active_joints

    def get_links(self):
        return self.links

    def get_qpos(self):
        return self.qpos.copy()

    def get_root_pose(self):
        return self.root_pose

    def set_qpos(self, qpos):
        self.set_qpos_calls += 1
        self.qpos = np.asarray(qpos)


class _Robot:
    def __init__(self):
        entity = _Entity()
        self.entity = entity
        self.left_entity = entity
        self.right_entity = entity
        self.left_arm_joints = entity.active_joints[:6]
        self.right_arm_joints = entity.active_joints[6:]
        self.left_ee = _Joint(entity.left_link, _Pose([0.0, 0.0, 0.5]))
        self.right_ee = _Joint(entity.right_link, _Pose([0.0, 0.0, -0.5]))
        self.left_entity_origion_pose = entity.root_pose
        self.right_entity_origion_pose = entity.root_pose
        self.left_global_trans_matrix = np.eye(3)
        self.right_global_trans_matrix = np.eye(3)
        self.left_delta_matrix = np.eye(3)
        self.right_delta_matrix = np.eye(3)
        self.left_gripper_bias = 0.12
        self.right_gripper_bias = 0.12


def test_joint_target_fk_handles_shared_articulation_without_mutation() -> None:
    robot = _Robot()
    bridge = RoboTwinJointTargetFK(robot)
    action = np.array(
        [0, 1, 2, 3, 4, 5, 0.25, 6, 7, 8, 9, 10, 11, 0.75],
        dtype=np.float32,
    )

    result = bridge.transform(action)

    assert robot.entity.model.compute_calls == 1
    np.testing.assert_allclose(robot.entity.model.last_qpos, np.arange(12))
    assert robot.entity.set_qpos_calls == 0
    assert result.shape == (14,)
    np.testing.assert_allclose(result[:3], [1.0, 3.0, 3.5])
    np.testing.assert_allclose(result[3:6], [0.0, 0.0, 0.0])
    assert result[6] == 0.25
    np.testing.assert_allclose(result[7:10], [7.0, 9.0, 2.5])
    np.testing.assert_allclose(result[10:13], [0.0, 0.0, 0.0])
    assert result[13] == 0.75


def test_quaternion_rotvec_roundtrip_is_short_and_sign_invariant() -> None:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    nonzero = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
    half_turn = np.array([0.0, 0.0, 0.0, -1.0])

    for quaternion in (identity, nonzero, half_turn):
        rotvec = quaternion_to_rotvec_wxyz(quaternion)
        reconstructed = rotvec_to_quaternion_wxyz(rotvec)

        assert np.linalg.norm(rotvec) <= np.pi
        np.testing.assert_allclose(
            quaternion_to_rotvec_wxyz(-quaternion),
            rotvec,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            abs(np.dot(reconstructed, quaternion)),
            1.0,
            atol=1e-7,
        )


def test_robotwin_gripper_pose_reproduces_offset_and_noncommuting_rotation() -> None:
    quarter_turn_z = quaternion_to_matrix_wxyz(
        np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    )
    quarter_turn_x = quaternion_to_matrix_wxyz(
        np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    )

    pose = robotwin_gripper_pose(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        global_trans_matrix=quarter_turn_z,
        delta_matrix=quarter_turn_x,
        gripper_bias=0.22,
    )

    expected_rotation = quarter_turn_z @ quarter_turn_x
    np.testing.assert_allclose(pose[:3], [1.0, 2.1, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        quaternion_to_matrix_wxyz(pose[3:]),
        expected_rotation,
        atol=1e-6,
    )
