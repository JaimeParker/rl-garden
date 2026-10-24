"""Raw-env construction for Isaac Lab 3.0.

Order: import tasks -> parse the env cfg (presets resolved) -> launch the
runtime that cfg requires (Kit only if needed) -> instantiate the env.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from rl_garden.envs.isaaclab.config import IsaacLabEnvConfig


def _apply_env_kwargs_and_final_obs(env_cfg: Any, cfg: IsaacLabEnvConfig) -> None:
    """Apply ``env_kwargs`` overrides, then ``compute_final_obs`` (on unless
    ``final_obs`` is False). ``--isaaclab.final_obs`` is the only way to set
    it: ``compute_final_obs`` inside ``env_kwargs`` is rejected rather than
    silently overridden."""
    if "compute_final_obs" in cfg.env_kwargs:
        raise ValueError(
            "Set terminal observations with --isaaclab.final_obs, not "
            "--isaaclab.env_kwargs_json's 'compute_final_obs'."
        )
    for key, value in cfg.env_kwargs.items():
        setattr(env_cfg, key, value)
    compute_final_obs = cfg.final_obs is not False
    if not hasattr(env_cfg, "compute_final_obs"):
        if not compute_final_obs:
            return
        raise ValueError(
            f"Env cfg {type(env_cfg).__name__} for {cfg.env_id!r} has no 'compute_final_obs' field "
            "(Isaac Lab 3.0 DirectRLEnvCfg/ManagerBasedRLEnvCfg declare it); pass "
            "--isaaclab.final_obs False to run without terminal observations."
        )
    env_cfg.compute_final_obs = compute_final_obs


def build(cfg: IsaacLabEnvConfig) -> Any:
    import isaaclab_tasks  # noqa: F401  (registers native gym ids)

    import rl_garden.envs.isaaclab.v3.tasks  # noqa: F401  (registers RlGarden-* ids)
    from isaaclab_tasks.utils import parse_env_cfg
    from rl_garden.envs.isaaclab.v3.app import get_or_launch_app

    env_cfg = parse_env_cfg(
        cfg.env_id, device=cfg.sim_device, num_envs=cfg.num_envs, overrides=list(cfg.overrides)
    )
    _apply_env_kwargs_and_final_obs(env_cfg, cfg)

    get_or_launch_app(env_cfg, {"headless": cfg.headless, "device": cfg.sim_device})
    return gym.make(cfg.env_id, cfg=env_cfg)
