"""Process-wide simulation-runtime singleton.

Isaac Lab 3.0 decides which runtime a task needs from its *resolved* env cfg:
``isaaclab.app.launch_simulation(env_cfg, launcher_args)`` walks the cfg and
boots Isaac Sim/Kit only when something in it requires Kit (Isaac Sim PhysX
physics, a Kit/RTX camera renderer); Newton physics with the Newton renderer
runs Kit-less. ``isaaclab.*`` / ``isaaclab_tasks.*`` modules can be imported
before the launch, so the env cfg is parsed first and passed in here.

A process hosts at most one Kit app, and ``rl_garden`` has no separate
eval-env process for the IsaacLab backend in v1 (see
``rl_garden.envs.backends.isaaclab``), so the runtime is launched once per
training run and kept open until process exit.
"""
from __future__ import annotations

import contextlib
from typing import Any

_STACK = contextlib.ExitStack()
_LAUNCHED = False


def get_or_launch_app(env_cfg: Any, launcher_args: dict) -> None:
    """Launch the runtime ``env_cfg`` requires, once per process.

    ``launcher_args`` is forwarded to ``launch_simulation`` (e.g. ``headless``,
    ``device``); it is mutated in place by Isaac Lab. ``launch_simulation`` also
    rewrites ``env_cfg.sim.device`` with the device the launcher resolved, so
    always pass ``device`` explicitly. Later calls are no-ops: the first
    launched runtime serves every env in the process.
    """
    global _LAUNCHED
    if _LAUNCHED:
        return
    from isaaclab.app import launch_simulation

    _STACK.enter_context(launch_simulation(env_cfg, launcher_args))
    _LAUNCHED = True


def close_app() -> None:
    """Close the launched runtime.

    When Kit was launched this shuts the Kit app down, which ends the Python
    process, so only call it as the very last step of a run. Without an
    explicit call, Kit's own ``atexit`` hook closes it at process exit.
    """
    global _LAUNCHED
    _STACK.close()
    _LAUNCHED = False
