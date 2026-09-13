"""RLBench env backend, registered as ``"rlbench"``."""
from __future__ import annotations

import json

from rl_garden.envs.backend_registry import EnvBackend, EnvRequest, register_env_backend


class RLBenchBackend(EnvBackend):
    api_version = 2
    config_field = "rlbench"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.rlbench.config import RLBenchEnvConfig
        from rl_garden.observations.schema import ObservationContractError

        config = req.backend_config
        obs = req.observation
        if obs.frame_stack > 1 and not (obs.rgb or obs.depth):
            raise ObservationContractError(
                f"rlbench: frame_stack={obs.frame_stack} requires at least "
                "one rgb/depth camera; a state-only observation cannot be "
                "frame-stacked."
            )
        if obs.extra_state:
            raise ObservationContractError("rlbench has no extra state sources")
        return RLBenchEnvConfig(
            task_name=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=config.device if config is not None else "cpu",
            rgb_cameras=obs.rgb,
            depth_cameras=obs.depth,
            state=obs.state,
            image_size=obs.image_size if obs.image_size is not None else (128, 128),
            frame_stack=obs.frame_stack,
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
