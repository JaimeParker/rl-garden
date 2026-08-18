"""``custom`` env backend — registered as ``"custom"``.

Template backend adapter wired to ``rl_garden.envs.custom``'s
``PointReach-v0`` toy task. Copy this file when adding a new backend; see
``.agents/rules/adding-env-backend.md`` for the full contract this class
must satisfy (``api_version = 2``, ``resolve_config``/``make_train_env``/
``make_eval_env`` implemented directly, side-effect-free
``resolve_config``).
"""
from __future__ import annotations

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class CustomBackend(EnvBackend):
    api_version = 2
    config_field = "custom"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.custom.config import CustomEnvConfig

        custom = req.backend_config  # CustomConfig or None
        return CustomEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=custom.device if custom is not None else "cpu",
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.custom import make_custom_env

        return make_custom_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.custom import make_custom_env

        return make_custom_env(cls.resolve_config(req, is_eval=True))


register_env_backend("custom", CustomBackend)
