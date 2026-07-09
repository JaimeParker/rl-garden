"""franka_bridge.py must be importable without ROS/Flask installed (this
sandbox has neither) -- all ROS/Flask/Franka message imports must stay
inside functions/methods, never at module level."""
from __future__ import annotations

import sys

import pytest


def test_module_imports_without_ros_or_flask_installed():
    for name in list(sys.modules):
        if name == "rospy" or name.startswith("flask") or name.startswith("robot_infra.controller.real"):
            del sys.modules[name]

    import robot_infra.controller.real.franka_bridge as m
    import robot_infra.controller.real.gripper_server as g

    assert hasattr(m, "create_app")
    assert hasattr(m, "FrankaBridgeController")
    assert hasattr(g, "FrankaGripperServer")


def test_missing_serl_franka_controllers_raises_actionable_error():
    from robot_infra.controller.real.franka_bridge import _check_serl_franka_controllers_installed

    with pytest.raises(SystemExit, match="catkin_make"):
        _check_serl_franka_controllers_installed()
