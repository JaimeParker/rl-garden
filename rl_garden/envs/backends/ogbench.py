"""OGBench env backend — registered as ``"ogbench"``.

Covers all 8 OGBench task families (locomotion: pointmaze/antmaze/
humanoidmaze/antsoccer; manipulation: cube/scene/puzzle), state and pixel
observations; see ``rl_garden.envs.ogbench.env`` module docstring for the
sync/async vectorization choice and pixel-obs scoping.
"""

from __future__ import annotations

import json

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class OGBenchBackend(EnvBackend):
    api_version = 2
    config_field = "ogbench"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.ogbench.config import OGBenchEnvConfig

        og = req.backend_config  # OGBenchConfig or None
        env_kwargs = (
            json.loads(og.env_kwargs_json)
            if og is not None and og.env_kwargs_json
            else {}
        )
        return OGBenchEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=og.device if og is not None else "cpu",
            env_kwargs=env_kwargs,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            vectorization=og.vectorization if og is not None else "sync",
        )

    _make_cfg = resolve_config

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.ogbench import make_ogbench_env

        return make_ogbench_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.ogbench import make_ogbench_env

        return make_ogbench_env(cls.resolve_config(req, is_eval=True))


register_env_backend("ogbench", OGBenchBackend)
