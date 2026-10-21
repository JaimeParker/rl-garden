"""Raw-env construction for Isaac Lab 3.0.

Order: import tasks -> parse the env cfg (presets resolved) -> launch the
runtime that cfg requires (Kit only if needed) -> instantiate the env.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from rl_garden.envs.isaaclab.config import IsaacLabEnvConfig


def build(cfg: IsaacLabEnvConfig) -> Any:
    import isaaclab_tasks  # noqa: F401  (registers native gym ids)

    import rl_garden.envs.isaaclab.v3.tasks  # noqa: F401  (registers RlGarden-* ids)
    from isaaclab_tasks.utils import parse_env_cfg
    from rl_garden.envs.isaaclab.v3.app import get_or_launch_app

    env_cfg = parse_env_cfg(
        cfg.env_id, device=cfg.sim_device, num_envs=cfg.num_envs, overrides=list(cfg.overrides)
    )
    for key, value in cfg.env_kwargs.items():
        setattr(env_cfg, key, value)

    get_or_launch_app(env_cfg, {"headless": cfg.headless, "device": cfg.sim_device})
    return gym.make(cfg.env_id, cfg=env_cfg)
