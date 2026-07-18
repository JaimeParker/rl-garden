"""mujoco_warp GPU environment factory and task registry.

``mujoco_warp`` has no ``gym.make()``/registration ecosystem of its own
(unlike Gymnasium's ``envs.mujoco`` or IsaacLab's ``isaaclab_tasks``), so this
module keeps a small rl-garden-owned name -> class registry instead of
faking a ``gym.register()`` wrapper around something that isn't really a gym
environment factory underneath.

``make_mujoco_warp_env(cfg)`` just looks up ``cfg.env_id`` and constructs the
task directly -- the task class (a ``CustomMujocoWarpEnv`` subclass) already
*is* the final rl-garden env shape (see ``custom_mujoco_warp_env.py`` module
docstring for why no adapter layer is needed here, unlike the CPU MuJoCo
backend).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from rl_garden.envs.mujoco_warp.config import MujocoWarpEnvConfig
from rl_garden.envs.mujoco_warp.custom_mujoco_warp_env import CustomMujocoWarpEnv

_TASKS: dict[str, type[CustomMujocoWarpEnv]] = {}


def register_mujoco_warp_task(name: str, cls: type[CustomMujocoWarpEnv]) -> None:
    if name in _TASKS:
        raise ValueError(f"mujoco_warp task {name!r} already registered")
    _TASKS[name] = cls


def make_mujoco_warp_env(cfg: MujocoWarpEnvConfig) -> gym.vector.VectorEnv:
    import rl_garden.envs.mujoco_warp.tasks  # noqa: F401  (triggers task registration)

    if cfg.env_id not in _TASKS:
        raise KeyError(
            f"Unknown mujoco_warp task {cfg.env_id!r}. Available: {sorted(_TASKS)}."
        )
    task_cls = _TASKS[cfg.env_id]
    env: gym.vector.VectorEnv = task_cls(
        nworld=cfg.num_envs,
        device=cfg.device,
        camera_width=cfg.camera_width,
        camera_height=cfg.camera_height,
        render_rgb=cfg.render_rgb,
        render_depth=cfg.render_depth,
        **cfg.env_kwargs,
    )

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper

        env = RewardScaleBiasVectorWrapper(env, scale=cfg.reward_scale, bias=cfg.reward_bias)

    return env
