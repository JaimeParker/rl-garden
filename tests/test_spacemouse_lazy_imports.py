"""robot_infra/teleop/spacemouse must be importable without pyspacemouse
installed (this sandbox has neither) -- the HID device import must stay
inside functions/methods, never at module level."""
from __future__ import annotations

import sys

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
