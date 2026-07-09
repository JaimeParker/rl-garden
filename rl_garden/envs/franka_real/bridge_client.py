"""HTTP client for the ``robot_infra`` Franka bridge (the transitional
ROS/``serl_franka_controllers`` bridge -- see
``docs/superpowers/specs/2026-07-09-real-robot-rl-design.md``).

Endpoint paths, HTTP verbs, and JSON field names match SERL's own
``franka_server.py`` (``3rd_party/serl/serl_robot_infra/robot_servers/``)
exactly, since ``robot_infra/controller/real/franka_bridge.py`` is a close
port of that file -- this client is not a new protocol, it's the same one.

Kept as a small, injectable object (rather than inlining ``requests`` calls
directly in ``FrankaRealEnv``) specifically so tests can substitute a fake
client and exercise ``FrankaRealEnv``'s action/observation/safety logic
without any real hardware, ROS, or HTTP server.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


class FrankaBridgeClient:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _post(self, path: str, json_body: Optional[dict] = None) -> dict[str, Any]:
        import requests  # lazy: only real-robot runs need this installed

        resp = requests.post(f"{self._base_url}{path}", json=json_body, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def get_state(self) -> dict[str, Any]:
        """Returns SERL's ``/getstate`` shape: ``pose`` (7), ``vel`` (6),
        ``force`` (3), ``torque`` (3), ``gripper_pos`` (scalar), plus
        ``q``/``dq``/``jacobian`` that ``FrankaRealEnv`` doesn't consume."""
        return self._post("/getstate")

    def send_pose(self, pose: np.ndarray) -> None:
        """``pose``: 7D absolute target end-effector pose (xyz + quat xyzw)."""
        self._post("/pose", {"arr": np.asarray(pose, dtype=float).tolist()})

    def send_gripper(self, open_gripper: bool) -> None:
        self._post("/open_gripper" if open_gripper else "/close_gripper")

    def reset_joints(self) -> None:
        self._post("/jointreset")

    def start_impedance(self) -> None:
        self._post("/startimp")

    def stop_impedance(self) -> None:
        self._post("/stopimp")
