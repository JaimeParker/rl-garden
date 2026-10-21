"""Isaac Lab generation detection, import-free (never boots Kit).

``isaaclab_physx`` ships only with Isaac Lab 3.0 and is the package the
``v3`` tasks hard-import, so its presence decides the generation. Version
strings are unreliable for this (2.x editable installs report the extension
version 0.54.x, 3.0 source installs 28.0.0, a 3.0 install without dist
metadata "0.0.0"); they are only used to reject a 3.x install that lacks
``isaaclab_physx`` and in error messages.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import Literal

Generation = Literal["v2", "v3"]


def generation() -> Generation:
    """Return ``"v3"`` for Isaac Lab 3.0, ``"v2"`` for 2.x; raise if neither is installed."""
    if importlib.util.find_spec("isaaclab_physx") is not None:
        return "v3"
    if importlib.util.find_spec("isaaclab") is not None:
        version = isaaclab_version()
        if _major_version(version) >= 3:
            raise ImportError(
                f"Installed isaaclab {version} looks like Isaac Lab 3.x, but its 'isaaclab_physx' "
                "package is missing (partial or broken install); reinstall isaaclab==3.0.0."
            )
        return "v2"
    raise ImportError(
        "IsaacLab backend needs Isaac Lab installed: 3.0 (recommended; Isaac Sim 6.2, Python 3.12) "
        "or 2.3.x (Isaac Sim 5.x, Python 3.11). No 'isaaclab' package was found; see "
        "docs/guides/isaaclab-custom-tasks.md#supported-isaac-lab-versions."
    )


def isaaclab_version() -> str:
    """Installed ``isaaclab`` distribution version, or ``"?"`` without dist metadata."""
    try:
        return importlib.metadata.version("isaaclab")
    except importlib.metadata.PackageNotFoundError:
        return "?"


def _major_version(version: str) -> int:
    """Leading numeric component of ``version``; -1 when it is not numeric."""
    major = version.split(".", 1)[0]
    return int(major) if major.isdigit() else -1
