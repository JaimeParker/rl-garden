"""Process-wide Isaac Sim application singleton.

IsaacLab requires constructing an ``AppLauncher`` (which boots the Isaac
Sim/Kit application) before any ``isaaclab.*`` / ``isaaclab_tasks.*`` module
is imported, and a process can only host one such application. ``rl_garden``
has no separate eval-env process for the IsaacLab backend in v1 (see
``rl_garden.envs.backends.isaaclab``), so this singleton only ever needs to
be launched once per training run.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

_APP: Optional[Any] = None


def _scrub_bundled_torch_from_syspath() -> None:
    """Remove Kit extension ``pip_prebundle`` dirs that ship their own torch.

    Several Isaac Sim extensions (e.g. ``omni.isaac.ml_archive``) bundle a
    separate copy of ``torch`` -- including a ``torch._dynamo`` build from a
    different torch version -- and add their ``pip_prebundle`` directory to
    ``sys.path`` as part of AppLauncher's own Kit bootstrap. Because
    ``torch`` itself is already imported (resolving correctly to the venv's
    copy) by the time this runs, only *lazily*-imported submodules of
    ``torch`` (e.g. ``torch._dynamo.repro.after_dynamo``, touched only once
    something calls ``torch.compile``) are at risk of resolving to the
    bundled copy instead, producing cross-version ImportErrors. Drop any
    such path so every subsequent ``torch.*`` submodule import can only find
    the venv's own copy.
    """
    sys.path[:] = [
        p
        for p in sys.path
        if not (os.path.basename(p.rstrip("/")) == "pip_prebundle" and os.path.isdir(os.path.join(p, "torch")))
    ]


def get_or_launch_app(headless: bool, sim_device: str, enable_cameras: bool = False) -> Any:
    """Return the process-wide Isaac Sim app, launching it on first call.

    ``enable_cameras`` must be True if the env being built spawns any camera
    sensor (e.g. ``TiledCamera``) -- IsaacLab raises at sensor-init time
    otherwise ("A camera was spawned without the --enable_cameras flag").
    Since this is a process-wide singleton, once launched without cameras a
    later call requesting them can't retroactively enable rendering.
    """
    global _APP
    if _APP is not None:
        return _APP

    # Exercise torch.compile end-to-end from the venv's own torch *before*
    # AppLauncher boots Kit. Some Kit extensions (e.g. omni.isaac.ml_archive)
    # bundle their own torch._dynamo build and import pieces of it as part
    # of their own initialization; whichever copy of a given torch._dynamo
    # submodule gets imported first wins the sys.modules caching race for
    # the rest of the process. Actually calling a compiled function (not
    # just importing torch._dynamo) walks the same lazy-import chain
    # torch.compile itself needs (eval_frame -> get_compiler_fn ->
    # repro.after_dynamo -> debug_utils -> testing -> types, at least as of
    # torch 2.6), so every submodule it touches gets cached from the venv
    # copy first, regardless of how deep that chain is or changes across
    # torch versions.
    import torch

    @torch.compile
    def _warm_up_dynamo(x: "torch.Tensor") -> "torch.Tensor":
        return x + 1

    _warm_up_dynamo(torch.zeros(1))

    from isaaclab.app import AppLauncher

    launcher_kwargs = dict(headless=headless, device=sim_device, enable_cameras=enable_cameras)
    if enable_cameras:
        # "performance" trims DLSS/post-processing overhead relative to the
        # "balanced" default -- camera-observation training has no need for
        # display-quality rendering.
        launcher_kwargs["rendering_mode"] = "performance"
    launcher = AppLauncher(**launcher_kwargs)
    _APP = launcher.app
    _scrub_bundled_torch_from_syspath()
    return _APP
