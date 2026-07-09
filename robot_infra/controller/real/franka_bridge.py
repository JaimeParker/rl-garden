"""Franka bridge: thin Flask/HTTP forwarder to the ROS-side
``serl_franka_controllers`` Cartesian impedance controller.

This is a close port of SERL's own ``franka_server.py``
(``3rd_party/serl/serl_robot_infra/robot_servers/franka_server.py``, reference
code -- not imported, since only ``serl_franka_controllers`` itself is
treated as a real vendored dependency in this repo), scoped down to the
endpoint set ``rl_garden.envs.franka_real.bridge_client.FrankaBridgeClient``
actually calls: ``/pose``, ``/getstate``, ``/jointreset``, ``/open_gripper``,
``/close_gripper``, ``/startimp``, ``/stopimp``.

Explicitly a **transitional** solution -- see
``docs/superpowers/specs/2026-07-09-real-robot-rl-design.md`` and
``docs/robot_infra_roadmap.md``: this file does no control-law computation
itself, it only forwards to the ROS controller so the ROS dependency stays
confined to this one module.

All ROS/Flask imports are lazy (inside functions/methods) so this module can
be imported -- e.g. for discovery/inspection -- without ROS, Flask, or any
Franka message packages installed. Running the server obviously still
requires them plus a real robot; that is explicitly out of scope for this
repo's automated tests (see the design doc's Testing & Verification section).
"""
from __future__ import annotations

import subprocess
import time
from typing import Any, Optional


class FrankaBridgeController:
    """Owns the ROS pub/sub wiring and the impedance controller's lifecycle."""

    def __init__(
        self,
        robot_ip: str,
        reset_joint_target: list[float],
        ros_pkg_name: str = "serl_franka_controllers",
    ) -> None:
        import geometry_msgs.msg as geom_msg
        import rospy
        from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
        from serl_franka_controllers.msg import ZeroJacobian

        self._geom_msg = geom_msg
        self._rospy = rospy
        self.robot_ip = robot_ip
        self.ros_pkg_name = ros_pkg_name
        self.reset_joint_target = reset_joint_target

        self.pose = None  # xyz(3) + quat_xyzw(4)
        self.vel = [0.0] * 6
        self.force = [0.0] * 3
        self.torque = [0.0] * 3
        self.q = [0.0] * 7
        self.dq = [0.0] * 7
        self._jacobian = None
        self._impedance_process: Optional[subprocess.Popen] = None
        self._joint_reset_process: Optional[subprocess.Popen] = None

        self._eq_pose_pub = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geom_msg.PoseStamped,
            queue_size=10,
        )
        self._error_recovery_pub = rospy.Publisher(
            "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1
        )
        rospy.Subscriber(
            "/cartesian_impedance_controller/franka_jacobian",
            ZeroJacobian,
            self._on_jacobian,
        )
        rospy.Subscriber("franka_state_controller/franka_states", FrankaState, self._on_state)

    def _on_jacobian(self, msg) -> None:
        import numpy as np

        self._jacobian = np.array(list(msg.zero_jacobian)).reshape((6, 7), order="F")

    def _on_state(self, msg) -> None:
        import numpy as np
        from scipy.spatial.transform import Rotation

        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        rot = Rotation.from_matrix(tmatrix[:3, :3])
        self.pose = np.concatenate([tmatrix[:3, -1], rot.as_quat()])
        self.dq = np.array(list(msg.dq)).reshape((7,))
        self.q = np.array(list(msg.q)).reshape((7,))
        self.force = np.array(list(msg.K_F_ext_hat_K)[:3])
        self.torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        if self._jacobian is not None:
            self.vel = self._jacobian @ self.dq

    def clear_errors(self) -> None:
        from franka_msgs.msg import ErrorRecoveryActionGoal

        self._error_recovery_pub.publish(ErrorRecoveryActionGoal())

    def move(self, pose) -> None:
        """``pose``: 7D absolute target [x, y, z, qx, qy, qz, qw]."""
        assert len(pose) == 7
        msg = self._geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = self._rospy.Time.now()
        msg.pose.position = self._geom_msg.Point(pose[0], pose[1], pose[2])
        msg.pose.orientation = self._geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
        self._eq_pose_pub.publish(msg)

    def start_impedance(self) -> None:
        self._impedance_process = subprocess.Popen(
            [
                "roslaunch",
                self.ros_pkg_name,
                "impedance.launch",
                f"robot_ip:={self.robot_ip}",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(5)

    def stop_impedance(self) -> None:
        if self._impedance_process is not None:
            self._impedance_process.terminate()
            self._impedance_process = None
        time.sleep(1)

    def reset_joint(self) -> None:
        import numpy as np

        try:
            self.stop_impedance()
            self.clear_errors()
        except Exception:
            pass
        time.sleep(3)
        self.clear_errors()

        self._rospy.set_param("/target_joint_positions", self.reset_joint_target)
        self._joint_reset_process = subprocess.Popen(
            ["roslaunch", self.ros_pkg_name, "joint.launch", f"robot_ip:={self.robot_ip}"],
            stdout=subprocess.PIPE,
        )
        time.sleep(1)
        self.clear_errors()

        count = 0
        time.sleep(1)
        while not np.allclose(
            np.array(self.reset_joint_target) - np.array(self.q), 0, atol=1e-2, rtol=1e-2
        ):
            time.sleep(1)
            count += 1
            if count > 30:
                break

        self._joint_reset_process.terminate()
        self._joint_reset_process = None
        time.sleep(1)
        self.clear_errors()

        self.start_impedance()

    def state_dict(self) -> dict[str, Any]:
        return {
            "pose": list(self.pose),
            "vel": list(self.vel),
            "force": list(self.force),
            "torque": list(self.torque),
        }


def create_app(
    robot_ip: str,
    reset_joint_target: list[float],
    gripper_server: Any,
    ros_pkg_name: str = "serl_franka_controllers",
):
    """Builds and returns the Flask app. Call ``app.run(host=..., port=...)``
    on the returned object; not done here so callers control the bind
    address/port explicitly (matches ``FrankaRealEnvConfig.bridge_url``)."""
    from flask import Flask, jsonify, request

    controller = FrankaBridgeController(robot_ip, reset_joint_target, ros_pkg_name)
    controller.start_impedance()

    app = Flask(__name__)

    @app.route("/startimp", methods=["POST"])
    def start_impedance():
        controller.clear_errors()
        controller.start_impedance()
        return "Started impedance"

    @app.route("/stopimp", methods=["POST"])
    def stop_impedance():
        controller.stop_impedance()
        return "Stopped impedance"

    @app.route("/jointreset", methods=["POST"])
    def joint_reset():
        controller.clear_errors()
        controller.reset_joint()
        return "Reset Joint"

    @app.route("/pose", methods=["POST"])
    def pose():
        controller.move(request.json["arr"])
        return "Moved"

    @app.route("/getstate", methods=["POST"])
    def get_state():
        state = controller.state_dict()
        state["gripper_pos"] = gripper_server.gripper_pos
        return jsonify(state)

    @app.route("/open_gripper", methods=["POST"])
    def open_gripper():
        gripper_server.open()
        return "Opened"

    @app.route("/close_gripper", methods=["POST"])
    def close_gripper():
        gripper_server.close()
        return "Closed"

    return app


def _check_serl_franka_controllers_installed() -> None:
    """Fail fast with a clear message if the catkin package this bridge
    forwards to hasn't been built into the sourced ROS workspace, instead of
    letting an unrelated-looking ImportError surface from inside
    ``FrankaBridgeController.__init__``."""
    try:
        import serl_franka_controllers.msg  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "franka_bridge: could not import 'serl_franka_controllers' -- it must be "
            "built as a catkin package in your sourced ROS workspace before running "
            "this bridge.\n"
            "  git clone https://github.com/rail-berkeley/serl_franka_controllers.git "
            "<catkin_ws>/src/serl_franka_controllers\n"
            "  cd <catkin_ws> && catkin_make && source devel/setup.bash\n"
            f"(original error: {exc})"
        ) from exc


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_ip", default="172.16.0.2")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--reset_joint_target", type=float, nargs=7, default=[0, 0, 0, -1.9, 0, 2, 0]
    )
    args = parser.parse_args()

    _check_serl_franka_controllers_installed()

    import rospy

    subprocess.Popen("roscore")
    time.sleep(1)
    rospy.init_node("franka_bridge")

    from robot_infra.controller.real.gripper_server import FrankaGripperServer

    gripper_server = FrankaGripperServer()
    app = create_app(args.robot_ip, args.reset_joint_target, gripper_server)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
