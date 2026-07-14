"""FrankaGripperServer._on_joint_state normalization -- constructing
FrankaGripperServer normally requires ROS (module-level rospy import inside
__init__), so this bypasses __init__ via object.__new__, matching the
pattern used elsewhere in this repo for hardware-backed classes (see
test_real_world_hil_serl_loops.py)."""
from __future__ import annotations

from robot_infra.controller.real.gripper_server import FrankaGripperServer


class _FakeJointState:
    def __init__(self, position):
        self.position = position


def test_on_joint_state_normalizes_by_fully_open_width():
    server = object.__new__(FrankaGripperServer)
    server._on_joint_state(_FakeJointState(position=[0.04, 0.04]))
    assert server.gripper_pos == (0.04 + 0.04) / 0.08
