"""IsaacLab env backend — registered as ``"isaaclab"``.

Supports both ``ManagerBasedRLEnv`` and ``DirectRLEnv`` tasks (native
``isaaclab_tasks`` or rl-garden's own ``RLGardenDirectRLEnv``-scaffold
tasks), state or image observations. No live eval env (see
``rl_garden.envs.isaaclab.env`` module docstring and
``.agents/rules/adding-env-backend.md``).
"""
from __future__ import annotations

import json

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class IsaacLabBackend(EnvBackend):
    config_field = "isaaclab"

    @classmethod
    def _make_cfg(cls, req: EnvRequest):
        from rl_garden.envs.isaaclab import IsaacLabEnvConfig

        il = req.backend_config  # IsaacLabConfig or None
        return IsaacLabEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_envs,
            seed=req.seed,
            headless=il.headless if il is not None else True,
            sim_device=il.sim_device if il is not None else "cuda:0",
            obs_mode=req.obs_mode,
            frame_stack=req.frame_stack,
            env_kwargs=(
                json.loads(il.env_kwargs_json) if il is not None and il.env_kwargs_json else {}
            ),
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.isaaclab import make_isaaclab_env

        return make_isaaclab_env(cls._make_cfg(req))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        raise NotImplementedError(
            "IsaacLab backend does not support a separate live eval env in v1 "
            "(AppLauncher only supports one Isaac Sim instance per process). "
            "Pass --eval_freq 0 to skip eval env creation."
        )


register_env_backend("isaaclab", IsaacLabBackend)
