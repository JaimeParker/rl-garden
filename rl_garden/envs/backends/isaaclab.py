"""IsaacLab env backend — registered as ``"isaaclab"``.

Supports both ``ManagerBasedRLEnv`` and ``DirectRLEnv`` tasks (native
``isaaclab_tasks`` or rl-garden's own ``RLGardenDirectRLEnv``-scaffold
tasks), state or image observations. No live eval env (see
``rl_garden.envs.isaaclab.env`` module docstring and
``.agents/rules/adding-env-backend.md``).

One backend for Isaac Lab 3.0 (recommended) and 2.3.x: the generation is
detected at env construction (``rl_garden.envs.isaaclab.detect``); the
``physics``/``renderer`` preset overrides forwarded here are 3.0-only and
rejected on 2.x, as is ``final_obs=True`` (its ``None`` default is resolved
per generation by the runtime, not here).
"""
from __future__ import annotations

import json

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class IsaacLabBackend(EnvBackend):
    api_version = 2
    config_field = "isaaclab"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.isaaclab import IsaacLabEnvConfig
        from rl_garden.observations.schema import ObservationContractError

        obs = req.observation
        if obs.image_size is not None:
            raise ObservationContractError(
                f"isaaclab backend cannot set image_size={obs.image_size!r}; camera "
                "resolution is fixed by the task registration (see "
                "RLGardenDirectRLEnv-scaffold task docstrings)"
            )
        if obs.extra_state:
            raise ObservationContractError("isaaclab has no extra state sources")
        il = req.backend_config  # IsaacLabConfig or None
        return IsaacLabEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_envs,
            seed=req.seed,
            headless=il.headless if il is not None else True,
            sim_device=il.sim_device if il is not None else "cuda:0",
            is_visual=obs.is_visual,
            state=obs.state,
            frame_stack=obs.frame_stack,
            env_kwargs=(
                json.loads(il.env_kwargs_json) if il is not None and il.env_kwargs_json else {}
            ),
            overrides=cls._preset_overrides(il),
            final_obs=il.final_obs if il is not None else None,
        )

    @staticmethod
    def _preset_overrides(il) -> list[str]:
        if il is None:
            return []
        selectors = (("physics", il.physics), ("renderer", il.renderer))
        return [f"{key}={value}" for key, value in selectors if value is not None]

    _make_cfg = resolve_config

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.isaaclab import make_isaaclab_env

        return make_isaaclab_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        raise NotImplementedError(
            "IsaacLab backend does not support a separate live eval env in v1 "
            "(the IsaacLab backend hosts at most one Isaac Sim/Kit app per process). "
            "Pass --eval_freq 0 to skip eval env creation."
        )


register_env_backend("isaaclab", IsaacLabBackend)
