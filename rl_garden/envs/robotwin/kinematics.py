"""State-free ACT joint-target to RoboTwin absolute gripper-pose FK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(quaternion).all():
        raise ValueError("Quaternion values must be finite.")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("Quaternion norm must be non-zero.")
    return quaternion / norm


def _as_short_float32_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Cast a canonical rotvec without rounding its norm above pi."""

    result = np.asarray(rotvec, dtype=np.float32).reshape(3)
    while float(np.linalg.norm(result)) > np.pi:
        result = np.nextafter(result, np.zeros_like(result))
    return result


def quaternion_to_rotvec_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert a WXYZ quaternion to a canonical shortest rotation vector."""

    quaternion = normalize_quaternion_wxyz(quaternion)
    if abs(quaternion[0]) < 1e-12:
        vector = quaternion[1:]
        first_nonzero_is_negative = next(
            (
                component < 0
                for component in vector
                if abs(component) >= 1e-12
            ),
            False,
        )
        if first_nonzero_is_negative:
            vector = -vector
        return _as_short_float32_rotvec(
            vector * (np.pi / float(np.linalg.norm(vector)))
        )
    if quaternion[0] < 0:
        quaternion = -quaternion
    vector = quaternion[1:]
    sin_half_angle = float(np.linalg.norm(vector))
    if sin_half_angle < 1e-12:
        return _as_short_float32_rotvec(2.0 * vector)
    angle = 2.0 * np.arctan2(
        sin_half_angle,
        np.clip(quaternion[0], -1.0, 1.0),
    )
    return _as_short_float32_rotvec(vector * (angle / sin_half_angle))


def rotvec_to_quaternion_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector to a normalized WXYZ quaternion."""

    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    if not np.isfinite(rotvec).all():
        raise ValueError("Rotation-vector values must be finite.")
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        quaternion = np.concatenate(
            [np.ones(1, dtype=np.float64), 0.5 * rotvec]
        )
    else:
        half_angle = 0.5 * angle
        quaternion = np.concatenate(
            [
                np.array([np.cos(half_angle)], dtype=np.float64),
                rotvec * (np.sin(half_angle) / angle),
            ]
        )
    return normalize_quaternion_wxyz(quaternion).astype(np.float32)


def quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError("Rotation matrix values must be finite.")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        quaternion = np.array(
            [0.25 * scale,
             (matrix[2, 1] - matrix[1, 2]) / scale,
             (matrix[0, 2] - matrix[2, 0]) / scale,
             (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quaternion = np.array(
                [(matrix[2, 1] - matrix[1, 2]) / scale,
                 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quaternion = np.array(
                [(matrix[0, 2] - matrix[2, 0]) / scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale,
                 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quaternion = np.array(
                [(matrix[1, 0] - matrix[0, 1]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale,
                 0.25 * scale]
            )
    quaternion = normalize_quaternion_wxyz(quaternion)
    return quaternion if quaternion[0] >= 0 else -quaternion


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = normalize_quaternion_wxyz(left)
    rw, rx, ry, rz = normalize_quaternion_wxyz(right)
    return normalize_quaternion_wxyz(
        np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ]
        )
    )


def compose_pose(root_pose: Any, local_pose: Any) -> tuple[np.ndarray, np.ndarray]:
    return compose_pose_components(root_pose.p, root_pose.q, local_pose.p, local_pose.q)


def compose_pose_components(
    parent_position: np.ndarray,
    parent_quaternion: np.ndarray,
    child_position: np.ndarray,
    child_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent_position = np.asarray(parent_position, dtype=np.float64).reshape(3)
    child_position = np.asarray(child_position, dtype=np.float64).reshape(3)
    parent_quaternion = normalize_quaternion_wxyz(parent_quaternion)
    child_quaternion = normalize_quaternion_wxyz(child_quaternion)
    position = (
        parent_position
        + quaternion_to_matrix_wxyz(parent_quaternion) @ child_position
    )
    quaternion = quaternion_multiply_wxyz(parent_quaternion, child_quaternion)
    return position, quaternion


def robotwin_gripper_pose(
    link_position: np.ndarray,
    link_quaternion: np.ndarray,
    *,
    global_trans_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    gripper_bias: float,
) -> np.ndarray:
    """Reproduce RoboTwin ``_trans_endpose(..., is_endpose=False)``."""

    rotation = (
        quaternion_to_matrix_wxyz(link_quaternion)
        @ np.asarray(global_trans_matrix, dtype=np.float64).reshape(3, 3)
        @ np.asarray(delta_matrix, dtype=np.float64).reshape(3, 3)
    )
    position = np.asarray(link_position, dtype=np.float64).reshape(3)
    position = position + rotation @ np.array([float(gripper_bias) - 0.12, 0.0, 0.0])
    return np.concatenate([position, matrix_to_quaternion_wxyz(rotation)]).astype(np.float32)


def _identity_index(items: list[Any], target: Any, *, kind: str) -> int:
    for index, item in enumerate(items):
        if item is target:
            return index
    raise ValueError(f"RoboTwin {kind} object is not part of its articulation.")


def _joint_child_link(joint: Any) -> Any:
    getter = getattr(joint, "get_child_link", None)
    if getter is not None:
        return getter()
    child_link = getattr(joint, "child_link", None)
    if child_link is not None:
        return child_link
    raise AttributeError("RoboTwin EE joint does not expose its child link.")


def _entity_root_pose(entity: Any, fallback: Any) -> Any:
    for method_name in ("get_root_pose", "get_pose"):
        method = getattr(entity, method_name, None)
        if method is not None:
            return method()
    return fallback


@dataclass(frozen=True)
class _ArmSpec:
    entity: Any
    arm_joint_indices: tuple[int, ...]
    ee_link_index: int
    ee_pose_in_child: Any
    root_pose_fallback: Any
    global_trans_matrix: np.ndarray
    delta_matrix: np.ndarray
    gripper_bias: float


@dataclass
class _EntityCache:
    entity: Any
    model: Any


class RoboTwinJointTargetFK:
    """Convert a 14D ACT target into rl-garden's 14D absolute EE action."""

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self._entities: dict[int, _EntityCache] = {}
        self._arms = {
            arm: self._build_arm_spec(arm)
            for arm in ("left", "right")
        }

    def _build_arm_spec(self, arm: str) -> _ArmSpec:
        entity = getattr(self.robot, f"{arm}_entity")
        entity_key = id(entity)
        if entity_key not in self._entities:
            self._entities[entity_key] = _EntityCache(
                entity=entity,
                model=entity.create_pinocchio_model(),
            )
        active_joints = list(entity.get_active_joints())
        arm_joints = list(getattr(self.robot, f"{arm}_arm_joints"))
        if len(arm_joints) != 6:
            raise ValueError(
                f"ACT RoboTwin bridge expects six {arm} arm joints, got {len(arm_joints)}."
            )
        arm_indices = tuple(
            _identity_index(active_joints, joint, kind=f"{arm} arm joint")
            for joint in arm_joints
        )
        links = list(entity.get_links())
        ee_joint = getattr(self.robot, f"{arm}_ee")
        ee_link = _joint_child_link(ee_joint)
        get_link_index = getattr(ee_link, "get_index", None)
        ee_link_index = (
            int(get_link_index())
            if get_link_index is not None
            else _identity_index(links, ee_link, kind=f"{arm} EE link")
        )
        get_pose_in_child = getattr(ee_joint, "get_pose_in_child", None)
        if get_pose_in_child is None:
            raise AttributeError("RoboTwin EE joint does not expose pose_in_child.")
        return _ArmSpec(
            entity=entity,
            arm_joint_indices=arm_indices,
            ee_link_index=ee_link_index,
            ee_pose_in_child=get_pose_in_child(),
            root_pose_fallback=getattr(self.robot, f"{arm}_entity_origion_pose"),
            global_trans_matrix=np.asarray(
                getattr(self.robot, f"{arm}_global_trans_matrix"), dtype=np.float64
            ),
            delta_matrix=np.asarray(
                getattr(self.robot, f"{arm}_delta_matrix"), dtype=np.float64
            ),
            gripper_bias=float(getattr(self.robot, f"{arm}_gripper_bias")),
        )

    def transform(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (14,):
            raise ValueError(f"ACT qpos target must have shape (14,), got {action.shape}.")
        if not np.isfinite(action).all():
            raise ValueError("ACT qpos target values must be finite.")

        arm_targets = {
            "left": action[:6],
            "right": action[7:13],
        }
        target_qpos_by_entity: dict[int, np.ndarray] = {}
        for arm, spec in self._arms.items():
            entity_key = id(spec.entity)
            target_qpos = target_qpos_by_entity.setdefault(
                entity_key,
                np.asarray(spec.entity.get_qpos(), dtype=np.float64).copy(),
            )
            for index, value in zip(spec.arm_joint_indices, arm_targets[arm]):
                target_qpos[index] = value

        for entity_key, target_qpos in target_qpos_by_entity.items():
            self._entities[entity_key].model.compute_forward_kinematics(target_qpos)

        output = np.empty(14, dtype=np.float32)
        output[6] = action[6]
        output[13] = action[13]
        for arm, position_slice, rotvec_slice in (
            ("left", slice(0, 3), slice(3, 6)),
            ("right", slice(7, 10), slice(10, 13)),
        ):
            spec = self._arms[arm]
            model = self._entities[id(spec.entity)].model
            local_link_pose = model.get_link_pose(spec.ee_link_index)
            joint_local_position, joint_local_quaternion = compose_pose(
                local_link_pose,
                spec.ee_pose_in_child,
            )
            root_pose = _entity_root_pose(spec.entity, spec.root_pose_fallback)
            link_position, link_quaternion = compose_pose_components(
                root_pose.p,
                root_pose.q,
                joint_local_position,
                joint_local_quaternion,
            )
            gripper_pose = robotwin_gripper_pose(
                link_position,
                link_quaternion,
                global_trans_matrix=spec.global_trans_matrix,
                delta_matrix=spec.delta_matrix,
                gripper_bias=spec.gripper_bias,
            )
            output[position_slice] = gripper_pose[:3]
            output[rotvec_slice] = quaternion_to_rotvec_wxyz(gripper_pose[3:])
        return output
