"""Ported from HIL-SERL's ``franka_env/spacemouse/spacemouse_expert.py``
(``SpaceMouseExpert``), close to verbatim.

``pyspacemouse`` (the HID device reader) is a real PyPI package -- HIL-SERL
vendored a copy of it locally, but this repo treats real pip dependencies as
lazy imports rather than vendored source (same reasoning as ``agentlace``:
no reason to carry a stale copy of someone else's package). Install it with
``pip install pyspacemouse`` on a machine with a SpaceMouse attached.
"""
from __future__ import annotations

import multiprocessing
from typing import Tuple

import numpy as np


class SpaceMouseExpert:
    """Continuously reads the SpaceMouse state in a background process and
    exposes the latest 6-DoF action + button state via :meth:`get_action`.
    """

    def __init__(self) -> None:
        import pyspacemouse

        pyspacemouse.open()

        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["buttons"] = [0, 0, 0, 0]

        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self) -> None:
        import pyspacemouse

        while True:
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw,
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                ]
                buttons = state[0].buttons

            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons

    def close(self) -> None:
        self.process.terminate()
