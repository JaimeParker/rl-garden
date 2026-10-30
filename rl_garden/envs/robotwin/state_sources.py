"""Opt-in, read-only task state. No simulator imports, steps, or random draws.

Versioned schemas describe observations, not a complete physics-engine snapshot.
Units are SI (arm angles radians); poses are world xyz plus quaternion wxyz,
using RoboTwin's tool-axis convention for TCP/controller poses.
"""
from __future__ import annotations

import numpy as np


def _pose_fields(prefix):
    return tuple(f"{prefix}.{s}" for s in ("x", "y", "z", "qw", "qx", "qy", "qz"))


SOURCE_FIELDS = {
    "robot_actual_v1": tuple(
        f"{side}.{quantity}.{joint}"
        for side in ("left", "right")
        for quantity in ("position", "velocity")
        for joint in (*[f"arm_{i}" for i in range(6)], "finger_0", "finger_1")
    ),
    "open_laptop_v1": (
        *_pose_fields("base"), "hinge.position", "hinge.velocity",
        "hinge.lower", "hinge.upper",
        *_pose_fields("contact_0"), *_pose_fields("contact_1"),
    ),
    "delta_ee_v1": tuple(
        field for side in ("left", "right")
        for field in (*_pose_fields(f"{side}.tcp"),
                      *_pose_fields(f"{side}.command"), f"{side}.command_valid")
    ),
}
V1_SOURCES = tuple(SOURCE_FIELDS)
SOURCE_FIELDS["open_laptop_robot_v2"] = (
    *SOURCE_FIELDS["robot_actual_v1"],
    *SOURCE_FIELDS["delta_ee_v1"],
    *_pose_fields("hinge_parent"), *_pose_fields("hinge_child"),
    "hinge.position", "hinge.velocity", "hinge.lower", "hinge.upper",
    "hinge.origin.x", "hinge.origin.y", "hinge.origin.z",
    "hinge.axis.x", "hinge.axis.y", "hinge.axis.z",
    *_pose_fields("contact_0"), *_pose_fields("contact_1"),
)
SOURCE_FIELDS["open_laptop_robot_v3"] = (
    *SOURCE_FIELDS["open_laptop_robot_v2"],
    *tuple(f"{part}.extent_{axis}" for part in ("hinge_parent", "hinge_child") for axis in "xyz"),
)
SOURCE_FIELDS["open_laptop_robot_v4"] = (
    *SOURCE_FIELDS["open_laptop_robot_v3"],
    *tuple(f"{part}.geometry_center_{axis}" for part in ("hinge_parent", "hinge_child") for axis in "xyz"),
    *tuple(f"{side}.to_contact_{i}.{axis}_parent_frame" for side in ("left", "right") for i in (0, 1) for axis in "xyz"),
    *tuple(f"{side}.contact_{part}_any_tick" for side in ("left", "right") for part in ("base", "lid")),
)
SOURCE_FIELDS["open_laptop_robot_v5"] = (
    *SOURCE_FIELDS["open_laptop_robot_v4"],
    *tuple(f"{side}.contact_{part}_last_tick" for side in ("left", "right") for part in ("base", "lid")),
)
SOURCE_DIMS = {name: len(fields) for name, fields in SOURCE_FIELDS.items()}


def validate_sources(names, task_name, control_mode):
    if isinstance(names, str) or not isinstance(names, (tuple, list)):
        raise ValueError("extra_state must be a sequence of source names")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("extra_state names must be strings")
    if len(set(names)) != len(names):
        raise ValueError("extra_state contains duplicate sources")
    unknown = set(names) - SOURCE_DIMS.keys()
    if unknown:
        raise ValueError(f"Unsupported extra_state sources: {sorted(unknown)}")
    if any(n.startswith("open_laptop_") for n in names) and task_name != "open_laptop":
        raise ValueError("open_laptop_v1 requires the open_laptop task")
    if any(n in names for n in ("delta_ee_v1", "open_laptop_robot_v2", "open_laptop_robot_v3", "open_laptop_robot_v4", "open_laptop_robot_v5")) and control_mode != "delta_ee":
        raise ValueError("delta_ee_v1 requires delta_ee control")
    if any(n in names for n in ("open_laptop_robot_v2", "open_laptop_robot_v3", "open_laptop_robot_v4", "open_laptop_robot_v5")) and len(names) != 1:
        raise ValueError("Select one task-state version alone; mixing versions duplicates fields")


def _vector(value, dim, label):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (dim,) or not np.isfinite(out).all():
        raise ValueError(f"{label}: expected finite shape ({dim},), got {out.shape}")
    return out.copy()


def _pose(value, label):
    if hasattr(value, "p") and hasattr(value, "q"):
        value = np.concatenate((value.p, value.q))
    out = _vector(value, 7, label)
    norm = np.linalg.norm(out[3:])
    if norm < 1e-8:
        raise ValueError(f"{label}: zero quaternion")
    out[3:] /= norm
    # Canonicalize q/-q without changing the simulator's own array.
    pivot = next(i for i in range(3, 7) if abs(out[i]) > 1e-8)
    if out[pivot] < 0:
        out[3:] *= -1
    return out


def _robot_actual(task):
    parts = []
    for side in ("left", "right"):
        robot = task.robot
        entity = getattr(robot, f"{side}_entity")
        active = list(entity.get_active_joints())
        arms = list(getattr(robot, f"{side}_arm_joints"))
        fingers = [g[0] for g in getattr(robot, f"{side}_gripper") if g is not None]
        if len(arms) != 6 or len(fingers) != 2:
            raise ValueError("robot_actual_v1 requires six arm and two finger joints per side")
        # Stable semantic order; never assume indices 0:6 or duplicate an articulation.
        indices = [active.index(j) for j in (*arms, *fingers)]
        if len(set(indices)) != 8:
            raise ValueError("Duplicate robot joints in state schema")
        for quantity in ("qpos", "qvel"):
            values = _vector(getattr(entity, f"get_{quantity}")(), len(active), quantity)
            parts.extend(values[indices])
    return parts


def _open_laptop(task):
    laptop = task.laptop
    limits = np.asarray(laptop.get_qlimits(), dtype=np.float64)
    if limits.shape != (1, 2) or not np.isfinite(limits).all() or limits[0, 1] <= limits[0, 0]:
        raise ValueError("open_laptop_v1 requires one bounded hinge")
    return np.concatenate((
        _pose(laptop.get_pose(), "laptop base"),
        _vector(laptop.get_qpos(), 1, "hinge qpos"),
        _vector(laptop.get_qvel(), 1, "hinge qvel"), limits[0],
        _pose(laptop.get_contact_point(0), "contact 0"),
        _pose(laptop.get_contact_point(1), "contact 1"),
    ))


def _delta_ee(task):
    parts = []
    for side in ("left", "right"):
        parts.extend(_pose(getattr(task.robot, f"get_{side}_tcp_pose")(), f"{side} TCP"))
        # Require the runtime's cache attribute even when its valid value is None.
        command = getattr(task, f"_delta_ee_{side}_command_pose")
        valid = bool(task.delta_ee_command_reference and command is not None)
        # At reset the next action initializes the cache from the measured EE pose.
        reference = command if valid else getattr(task.robot, f"get_{side}_ee_pose")()
        parts.extend(_pose(reference, f"{side} command reference"))
        parts.append(float(valid))
    return parts


def _open_laptop_robot_v2(task):
    from .task_state_v2 import read_task_state_v2

    return read_task_state_v2(task)


def _open_laptop_robot_v3(task):
    from .laptop_dimensions import read_task_state_v3

    return read_task_state_v3(task)


def prepare_sources(task, names):
    if "open_laptop_robot_v5" in names:
        from .task_state_v5 import prepare_task_state_v5

        prepare_task_state_v5(task)
        return
    if "open_laptop_robot_v4" in names:
        from .task_state_v4 import prepare_task_state_v4

        prepare_task_state_v4(task)
        return
    if "open_laptop_robot_v3" in names:
        from .laptop_dimensions import prepare_laptop_dimensions

        prepare_laptop_dimensions(task)


def _open_laptop_robot_v4(task):
    from .task_state_v4 import read_task_state_v4

    return read_task_state_v4(task)


def _open_laptop_robot_v5(task):
    from .task_state_v5 import read_task_state_v5

    return read_task_state_v5(task)


_READERS = {"robot_actual_v1": _robot_actual,
            "open_laptop_v1": _open_laptop,
            "delta_ee_v1": _delta_ee,
            "open_laptop_robot_v2": _open_laptop_robot_v2,
            "open_laptop_robot_v3": _open_laptop_robot_v3,
            "open_laptop_robot_v4": _open_laptop_robot_v4,
            "open_laptop_robot_v5": _open_laptop_robot_v5}


def read_sources(task, names):
    result = {}
    for name in names:
        if name not in _READERS:
            raise ValueError(f"Unsupported extra_state source: {name}")
        values = _vector(_READERS[name](task), SOURCE_DIMS[name], name)
        values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: float32 overflow")
        result[f"state_{name}"] = values
    return result
