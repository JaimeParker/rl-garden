"""Minimal Franka-hand gripper server, ported from SERL's
``serl_robot_infra/robot_servers/franka_gripper_server.py`` (not imported --
``3rd_party/serl`` is reference code, unlike the vendored ROS controller
``3rd_party/serl_franka_controllers``, which this repo treats as a real
dependency; see ``docs/superpowers/specs/2026-07-09-real-robot-rl-design.md``).

Only the Franka hand (``franka_gripper`` ROS action interface) is supported
in v1. A Robotiq gripper would need its own ``GripperServer`` subclass
following the same shape, added when there's an actual need for one.
"""
from __future__ import annotations


class GripperServer:
    def __init__(self) -> None:
        self.gripper_pos: float = 0.0

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class FrankaGripperServer(GripperServer):
    """Talks to the standard Franka hand via the ``franka_gripper`` ROS action API."""

    def __init__(self) -> None:
        super().__init__()
        import rospy
        from franka_gripper.msg import GraspActionGoal, MoveActionGoal
        from sensor_msgs.msg import JointState

        self._MoveActionGoal = MoveActionGoal
        self._GraspActionGoal = GraspActionGoal

        self._move_pub = rospy.Publisher(
            "/franka_gripper/move/goal", MoveActionGoal, queue_size=1
        )
        self._grasp_pub = rospy.Publisher(
            "/franka_gripper/grasp/goal", GraspActionGoal, queue_size=1
        )
        self._state_sub = rospy.Subscriber(
            "/franka_gripper/joint_states", JointState, self._on_joint_state
        )

    def _on_joint_state(self, msg) -> None:
        # Two-finger width in meters, summed across both fingers -- matches
        # SERL's own convention for `gripper_pos`.
        self.gripper_pos = float(sum(msg.position))

    def open(self) -> None:
        msg = self._MoveActionGoal()
        msg.goal.width = 0.09
        msg.goal.speed = 0.3
        self._move_pub.publish(msg)

    def close(self) -> None:
        msg = self._GraspActionGoal()
        msg.goal.width = 0.01
        msg.goal.speed = 0.3
        msg.goal.force = 10.0
        msg.goal.epsilon.inner = 0.08
        msg.goal.epsilon.outer = 0.08
        self._grasp_pub.publish(msg)
