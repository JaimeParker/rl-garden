"""Raw-env construction for Isaac Lab 2.x.

Order: launch Kit (``AppLauncher``) -> import tasks -> parse the env cfg ->
instantiate the env. Importing ``isaaclab_tasks`` or the ``v2`` task modules
before Kit runs fails on 2.x (they import ``omni``/``pxr``), so this module
keeps every Isaac Lab import inside ``build``.
"""
from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym

from rl_garden.envs.isaaclab.config import IsaacLabEnvConfig


def _reject_presets(overrides: Sequence[str], version: str) -> None:
    if overrides:
        raise ValueError(
            f"--isaaclab.physics/--isaaclab.renderer presets need Isaac Lab 3.0, detected 2.x {version} "
            f"(got overrides {list(overrides)!r}). Isaac Lab 2.x always runs Kit + Isaac Sim PhysX + "
            "TiledCamera (RTX); drop these flags or install Isaac Lab 3.0."
        )


def _reject_camera_off_primary_gpu(is_visual: bool, sim_device: str) -> None:
    if is_visual and sim_device not in ("cuda:0", "cuda"):
        raise ValueError(
            f"Isaac Sim 5.1 / Isaac Lab 2.x renders cameras only on cuda:0 (got sim_device={sim_device!r}; "
            "usdrt: 'GPUs other than cuda:0 are not currently supported'), and the process otherwise "
            "hangs at simulation start. Pass --isaaclab.sim_device cuda:0."
        )


class _CloseAppOnClose(gym.Wrapper):
    """Shuts the Kit app down after the env closes; see ``v2.app.close_app``."""

    def close(self) -> None:
        from rl_garden.envs.isaaclab.v2.app import close_app

        super().close()
        close_app()


def build(cfg: IsaacLabEnvConfig) -> Any:
    from rl_garden.envs.isaaclab.detect import isaaclab_version
    from rl_garden.envs.isaaclab.v2.app import get_or_launch_app

    _reject_presets(cfg.overrides, isaaclab_version())
    _reject_camera_off_primary_gpu(cfg.is_visual, cfg.sim_device)
    get_or_launch_app(cfg.headless, cfg.sim_device, enable_cameras=cfg.is_visual)

    import isaaclab_tasks  # noqa: F401  (registers native gym ids)

    import rl_garden.envs.isaaclab.v2.tasks  # noqa: F401  (registers RlGarden-* ids)
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(cfg.env_id, device=cfg.sim_device, num_envs=cfg.num_envs)
    for key, value in cfg.env_kwargs.items():
        setattr(env_cfg, key, value)

    return _CloseAppOnClose(gym.make(cfg.env_id, cfg=env_cfg))
