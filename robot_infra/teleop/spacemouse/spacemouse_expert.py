"""Ported from HIL-SERL's ``franka_env/spacemouse/spacemouse_expert.py``
(``SpaceMouseExpert``), close to verbatim.

``pyspacemouse`` (the HID device reader) is a real PyPI package -- HIL-SERL
vendored a copy of it locally, but this repo treats real pip dependencies as
lazy imports rather than vendored source (same reasoning as ``agentlace``:
no reason to carry a stale copy of someone else's package). Install it with
``pip install pyspacemouse`` on a machine with a SpaceMouse attached.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Tuple

import numpy as np


class SpaceMouseExpert:
    """Continuously reads the SpaceMouse state in a background process and
    exposes the latest 6-DoF action + button state via :meth:`get_action`.
    """

    def __init__(self) -> None:
        import pyspacemouse

        try:
            device = pyspacemouse.open()
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to open SpaceMouse teleop device. If pyspacemouse can "
                "list the device but cannot open it on Linux, check hidraw "
                "permissions or udev rules for the 3Dconnexion device."
            ) from exc
        if device is None:
            raise RuntimeError(
                "Failed to open SpaceMouse teleop device. "
                "Check that a supported SpaceMouse is connected and accessible."
            )
        print("[teleop] spacemouse input connected", flush=True)

        self._pyspacemouse = pyspacemouse
        self._device = device
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.latest_data = {}
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["buttons"] = [0, 0, 0, 0]

        self.thread = threading.Thread(target=self._read_spacemouse, daemon=True)
        self.thread.start()

    @staticmethod
    def _state_to_action_buttons(state: Any) -> Tuple[list[float], list]:
        return [
            state.y, state.x, -state.z,
            state.roll, -state.pitch, state.yaw,
        ], list(state.buttons)

    def _read_spacemouse(self) -> None:
        while not self._stop_event.is_set():
            if hasattr(self._device, "read"):
                single_state = self._device.read()
                state = [] if single_state is None else [single_state]
            elif hasattr(self._pyspacemouse, "read_all"):
                state = self._pyspacemouse.read_all()
            else:
                raise RuntimeError(
                    "Installed pyspacemouse exposes neither device.read() nor "
                    "pyspacemouse.read_all()."
                )
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                action_0, buttons_0 = self._state_to_action_buttons(state[0])
                action_1, buttons_1 = self._state_to_action_buttons(state[1])
                action = action_0 + action_1
                buttons = buttons_0 + buttons_1
            elif len(state) == 1:
                action, buttons = self._state_to_action_buttons(state[0])

            with self._lock:
                self.latest_data["action"] = action
                self.latest_data["buttons"] = buttons
            time.sleep(0.001)

    def get_action(self) -> Tuple[np.ndarray, list]:
        with self._lock:
            action = self.latest_data["action"]
            buttons = self.latest_data["buttons"]
        return np.array(action), buttons

    def close(self) -> None:
        self._stop_event.set()
        self.thread.join(timeout=1.0)
        close = getattr(self._device, "close", None)
        if callable(close):
            close()
            return
        close = getattr(self._pyspacemouse, "close", None)
        if callable(close):
            close()
