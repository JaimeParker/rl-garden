"""Task-oriented state: verified arm chains and topology-selected laptop parts.

Read-only, no stepping/RNG, no cached simulator handles across reset. The single
revolute hinge identifies its parent and moving child; link names are not semantics.
This exposes kinematic geometry, not collision mesh dimensions or full physics state.
"""
from __future__ import annotations

import numpy as np

from .state_sources import _delta_ee, _pose, _robot_actual, _vector


def _chain(link):
    """Return non-root links and incoming joints up to the articulation root."""
    result, seen = [], set()
    while link is not None:
        name = link.get_name()
        if name in seen:
            raise ValueError("Cycle or duplicate link names in articulation")
        seen.add(name)
        parent = link.get_parent()
        if parent is None:
            break
        joint = link.get_joint()
        if joint is None:
            raise ValueError("Non-root link has no incoming joint")
        result.append((link, joint))
        link = parent
    return result


def _check_robot(task):
    description = {}
    for side in ("left", "right"):
        robot = task.robot
        arms = list(getattr(robot, side + "_arm_joints"))
        expected = [j.get_name() for j in arms]
        if len(expected) != 6 or len(set(expected)) != 6:
            raise ValueError("v2 requires six unique arm joints per side")
        ee = getattr(robot, side + "_ee")
        chain = _chain(ee.get_child_link())
        actual = [j.get_name() for _, j in chain if j.get_dof() > 0]
        if set(actual) != set(expected) or len(actual) != len(expected):
            raise ValueError(f"{side}: selected joints do not cover the complete EE chain: {actual}")
        if any(j.get_dof() != 1 or j.get_type() not in ("revolute", "revolute_unwrapped") for j in arms):
            raise ValueError("v2 arm state requires single-DOF revolute joints")
        description[side] = {"arm_joints": expected,
                             "ee_link": ee.get_child_link().get_name(),
                             "root_pose_world": _pose(getattr(robot, side + "_entity").get_pose(), "robot root").tolist()}
    return description


def _resolve_laptop(task):
    laptop = task.laptop
    joints = list(laptop.actor.get_active_joints())
    if len(joints) != 1:
        raise ValueError("v2 requires exactly one active laptop hinge")
    hinge = joints[0]
    if hinge.get_dof() != 1 or hinge.get_type() not in ("revolute", "revolute_unwrapped"):
        raise ValueError("Laptop active joint must be a single-DOF revolute hinge")
    parent, child = hinge.get_parent_link(), hinge.get_child_link()
    if parent is None or child is None:
        raise ValueError("Laptop hinge is missing a parent or child link")
    # A second movable ancestor would invalidate the single-hinge schema.
    if any(j.get_dof() for _, j in _chain(parent)):
        raise ValueError("Laptop hinge parent is not rigidly attached to the root")
    links = laptop.link_dict
    if links.get(parent.get_name()) is not parent or links.get(child.get_name()) is not child:
        # pybind wrappers can differ in identity; compare their names instead.
        if parent.get_name() not in links or child.get_name() not in links:
            raise ValueError("Hinge links are absent from laptop link mapping")
    contact_links = []
    for index in (0, 1):
        name = laptop.config["contact_points"][index]["base"]
        link = links[name]
        active_path = [j.get_name() for _, j in _chain(link) if j.get_dof() > 0]
        if any(n != hinge.get_name() for n in active_path):
            raise ValueError("Contact point is on an unsupported moving branch")
        contact_links.append(name)
    # open_laptop's opening operation point must move with the hinge.
    opening_link = links[contact_links[1]]
    if hinge.get_name() not in [j.get_name() for _, j in _chain(opening_link)]:
        raise ValueError("Opening point 1 is not attached to the hinge's moving subtree")
    return hinge, parent, child, contact_links


def describe_task_state_v2(task):
    """Audit metadata only; not an observation or model-ID input."""
    robot = _check_robot(task)
    hinge, parent, child, contacts = _resolve_laptop(task)
    return {"version": 2, "robot": robot, "hinge_joint": hinge.get_name(),
            "hinge_parent_link": parent.get_name(), "hinge_child_link": child.get_name(),
            "contact_links": contacts, "pose_frame": "world_xyz_wxyz",
            "hinge_axis_convention": "joint_frame_positive_x",
            "full_physics_state": False}


def read_task_state_v2(task):
    _check_robot(task)
    hinge, parent, child, _ = _resolve_laptop(task)
    laptop = task.laptop
    limits = np.asarray(laptop.get_qlimits(), dtype=np.float64)
    if limits.shape != (1, 2) or not np.isfinite(limits).all() or limits[0, 1] <= limits[0, 0]:
        raise ValueError("Expected finite, ordered limits for a single hinge")
    joint_pose = _pose(hinge.get_global_pose(), "hinge frame")
    w, x, y, z = joint_pose[3:]
    # SAPIEN revolute axes are joint-frame X, not URDF/world X without rotation.
    axis = np.array([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)])
    return np.concatenate((
        _robot_actual(task), _delta_ee(task),
        _pose(parent.get_entity_pose(), "hinge parent"),
        _pose(child.get_entity_pose(), "hinge child"),
        _vector(laptop.get_qpos(), 1, "hinge angle"),
        _vector(laptop.get_qvel(), 1, "hinge velocity"), limits[0],
        joint_pose[:3], axis,
        _pose(laptop.get_contact_point(0), "contact 0"),
        _pose(laptop.get_contact_point(1), "contact 1"),
    ))
