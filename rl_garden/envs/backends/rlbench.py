"""RLBench env backend, registered as ``"rlbench"``."""
from __future__ import annotations

import json

from rl_garden.envs.backend_registry import EnvBackend, EnvRequest, register_env_backend


class RLBenchBackend(EnvBackend):
    api_version = 2
    config_field = "rlbench"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.buffers.rlbench_dataset import RLBENCH_CAMERA_NAMES
        from rl_garden.envs.rlbench.config import RLBenchEnvConfig

        config = req.backend_config
        return RLBenchEnvConfig(
            task_name=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=config.device if config is not None else "cpu",
            obs_mode=req.obs_mode,
            cameras=tuple(config.cameras) if config is not None else RLBENCH_CAMERA_NAMES,
            image_size=tuple(config.image_size) if config is not None else (128, 128),
            headless=config.headless if config is not None else True,
            env_kwargs=json.loads(config.env_kwargs_json) if config is not None else {},
            reward_scale=1.0 if is_eval else req.reward_scale,
            reward_bias=0.0 if is_eval else req.reward_bias,
            vectorization=config.vectorization if config is not None else "sync",
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.rlbench import make_rlbench_env

        return make_rlbench_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.rlbench import make_rlbench_env

        return make_rlbench_env(cls.resolve_config(req, is_eval=True))


register_env_backend("rlbench", RLBenchBackend)
