"""robomimic env backend, registered as ``"robomimic"``."""
from __future__ import annotations

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class RobomimicBackend(EnvBackend):
    api_version = 2
    config_field = "robomimic"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.robomimic.config import RobomimicEnvConfig

        config = req.backend_config
        return RobomimicEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            dataset_path=config.dataset_path if config is not None else None,
            device=config.device if config is not None else "cpu",
            horizon=config.horizon if config is not None else 400,
            terminate_on_success=config.terminate_on_success if config is not None else False,
            env_kwargs_json=config.env_kwargs_json if config is not None else "{}",
            reward_scale=1.0 if is_eval else req.reward_scale,
            reward_bias=0.0 if is_eval else req.reward_bias,
        )

    _make_cfg = resolve_config

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.robomimic import make_robomimic_env

        return make_robomimic_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.robomimic import make_robomimic_env

        return make_robomimic_env(cls.resolve_config(req, is_eval=True))


register_env_backend("robomimic", RobomimicBackend)
