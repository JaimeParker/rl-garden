"""robot_infra/teleop/spacemouse must be importable without pyspacemouse
installed (this sandbox has neither) -- the HID device import must stay
inside functions/methods, never at module level."""
from __future__ import annotations

import sys
import time
import types

import numpy as np
import pytest


def test_module_imports_without_pyspacemouse_installed():
    for name in list(sys.modules):
        if name == "pyspacemouse" or name.startswith("robot_infra.teleop.spacemouse"):
            del sys.modules[name]

    import robot_infra.teleop.spacemouse as m

    assert hasattr(m, "SpaceMouseExpert")
    assert hasattr(m, "SpaceMouseTeleOpWrapper")


def test_spacemouse_expert_raises_actionable_error_without_pyspacemouse():
    from robot_infra.teleop.spacemouse import SpaceMouseExpert

    with pytest.raises(ModuleNotFoundError, match="pyspacemouse"):
        SpaceMouseExpert()


def test_spacemouse_expert_raises_when_open_returns_none(monkeypatch):
    from robot_infra.teleop.spacemouse import SpaceMouseExpert

    monkeypatch.setitem(
        sys.modules,
        "pyspacemouse",
        types.SimpleNamespace(open=lambda: None),
    )

    with pytest.raises(RuntimeError, match="Failed to open SpaceMouse"):
        SpaceMouseExpert()


def _wait_for_action(expert, expected_buttons):
    for _ in range(100):
        action, buttons = expert.get_action()
        if buttons == expected_buttons:
            return action, buttons
        time.sleep(0.01)
    return expert.get_action()


def test_spacemouse_expert_reads_pyspacemouse_v2_device_api(monkeypatch):
    from robot_infra.teleop.spacemouse import SpaceMouseExpert

    class FakeDevice:
        def __init__(self):
            self.closed = False

        def read(self):
            return types.SimpleNamespace(
                x=2.0,
                y=-1.0,
                z=3.0,
                roll=-4.0,
                pitch=5.0,
                yaw=-6.0,
                buttons=[1, 0],
            )

        def close(self):
            self.closed = True

    device = FakeDevice()
    monkeypatch.setitem(
        sys.modules,
        "pyspacemouse",
        types.SimpleNamespace(open=lambda: device),
    )

    expert = SpaceMouseExpert()
    try:
        action, buttons = _wait_for_action(expert, [1, 0])
    finally:
        expert.close()

    assert np.allclose(action, [-1.0, 2.0, -3.0, -4.0, -5.0, -6.0])
    assert buttons == [1, 0]
    assert device.closed is True


def test_spacemouse_expert_falls_back_to_pyspacemouse_v1_read_all(monkeypatch):
    from robot_infra.teleop.spacemouse import SpaceMouseExpert

    state = types.SimpleNamespace(
        x=2.0,
        y=-1.0,
        z=3.0,
        roll=-4.0,
        pitch=5.0,
        yaw=-6.0,
        buttons=[0, 1],
    )
    closed = {"value": False}
    monkeypatch.setitem(
        sys.modules,
        "pyspacemouse",
        types.SimpleNamespace(
            open=lambda: object(),
            read_all=lambda: [state],
            close=lambda: closed.__setitem__("value", True),
        ),
    )

    expert = SpaceMouseExpert()
    try:
        action, buttons = _wait_for_action(expert, [0, 1])
    finally:
        expert.close()

    assert np.allclose(action, [-1.0, 2.0, -3.0, -4.0, -5.0, -6.0])
    assert buttons == [0, 1]
    assert closed["value"] is True


class _FakeExpert:
    def __init__(self, action, buttons):
        self.action = action
        self.buttons = buttons
        self.closed = False

    def get_action(self):
        return self.action, self.buttons

    def close(self):
        self.closed = True


def _wrapper_with_fake_expert(monkeypatch, action, buttons, **kwargs):
    from robot_infra.teleop.spacemouse import spacemouse_teleop_wrapper as module

    fake = _FakeExpert(action, buttons)
    monkeypatch.setattr(module, "SpaceMouseExpert", lambda: fake)
    return module.SpaceMouseTeleOpWrapper(**kwargs), fake


def test_spacemouse_wrapper_reports_no_intervention_for_zero_input(monkeypatch):
    wrapper, _ = _wrapper_with_fake_expert(
        monkeypatch, action=[0.0] * 6, buttons=[0, 0]
    )

    sample = wrapper.poll()

    assert sample.intervened is False
    assert sample.gripper == 1.0
    assert sample.action.shape == (7,)


def test_spacemouse_wrapper_reports_motion_intervention(monkeypatch):
    wrapper, _ = _wrapper_with_fake_expert(
        monkeypatch, action=[0.0, 0.002, 0.0, 0.0, 0.0, 0.0], buttons=[0, 0]
    )

    sample = wrapper.poll()

    assert sample.intervened is True


def test_spacemouse_wrapper_scales_twist(monkeypatch):
    wrapper, _ = _wrapper_with_fake_expert(
        monkeypatch,
        action=[1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
        buttons=[0, 0],
        spacemouse_scale=0.5,
    )

    sample = wrapper.poll()

    assert np.allclose(sample.twist, [0.5, -1.0, 1.5, -2.0, 2.5, -3.0])
    assert np.allclose(sample.action[:6], sample.twist)
    assert sample.gripper == 1.0


def test_spacemouse_wrapper_maps_buttons_to_gripper(monkeypatch):
    close_wrapper, _ = _wrapper_with_fake_expert(
        monkeypatch, action=[0.0] * 6, buttons=[1, 0]
    )
    open_wrapper, _ = _wrapper_with_fake_expert(
        monkeypatch, action=[0.0] * 6, buttons=[0, 1]
    )

    assert close_wrapper.poll().gripper == -1.0
    assert open_wrapper.poll().gripper == 1.0


def test_spacemouse_wrapper_selects_index_from_multiple_devices(monkeypatch):
    wrapper, _ = _wrapper_with_fake_expert(
        monkeypatch,
        action=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        buttons=[1, 0, 0, 1],
        spacemouse_index=1,
    )

    sample = wrapper.poll()

    assert sample.twist[0] == 2.0
    assert sample.gripper == 1.0


def test_spacemouse_wrapper_rejects_missing_index(monkeypatch):
    wrapper, _ = _wrapper_with_fake_expert(
        monkeypatch,
        action=[0.0] * 6,
        buttons=[0, 0],
        spacemouse_index=1,
    )

    with pytest.raises(ValueError, match="spacemouse_index=1"):
        wrapper.poll()
