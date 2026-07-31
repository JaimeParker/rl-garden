"""Legacy Gym/D4RL backend, registered as ``"d4rl_legacy"``."""
from __future__ import annotations

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class D4RLLegacyBackend(EnvBackend):
    config_field = "d4rl_legacy"

    @classmethod
    def _make_cfg(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.d4rl_legacy import D4RLLegacyEnvConfig

        config = req.backend_config
        return D4RLLegacyEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            device=config.device if config is not None else "cpu",
            reward_scale=1.0 if is_eval else req.reward_scale,
            reward_bias=0.0 if is_eval else req.reward_bias,
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.d4rl_legacy import make_d4rl_legacy_env

        return make_d4rl_legacy_env(cls._make_cfg(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.d4rl_legacy import make_d4rl_legacy_env

        return make_d4rl_legacy_env(cls._make_cfg(req, is_eval=True))


register_env_backend("d4rl_legacy", D4RLLegacyBackend)
