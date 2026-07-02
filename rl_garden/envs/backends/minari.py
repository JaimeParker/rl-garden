"""Minari env backend — registered as ``"minari"``."""
from __future__ import annotations

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class MinariBackend(EnvBackend):
    config_field = "minari"

    @classmethod
    def _make_cfg(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.minari.config import MinariEnvConfig

        mn = req.backend_config  # MinariConfig or None
        return MinariEnvConfig(
            dataset_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            eval_env=is_eval,
            device=mn.device if mn is not None else "cpu",
            download=mn.download if mn is not None else True,
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.minari.env import make_minari_env

        return make_minari_env(cls._make_cfg(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.minari.env import make_minari_env

        return make_minari_env(cls._make_cfg(req, is_eval=True))


register_env_backend("minari", MinariBackend)
