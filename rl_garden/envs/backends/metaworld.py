"""Meta-World env backend — registered as ``"metaworld"``.

Single-task env ids (any of ``metaworld.MT1.ENV_NAMES``, e.g. ``"reach-v3"``)
support state, rgb/depth camera, and frame-stacked observations; the
``"MT10"``/``"MT50"`` multi-task benchmarks are state observations only (no
per-env construction hook to attach a camera renderer to). See
``rl_garden.envs.metaworld.env`` module docstring for the MT1-vs-MT10/50
vectorization shape.
"""

from __future__ import annotations

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class MetaWorldBackend(EnvBackend):
    api_version = 2
    config_field = "metaworld"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.metaworld.config import MetaWorldEnvConfig
        from rl_garden.observations.schema import ObservationContractError

        mw = req.backend_config  # MetaWorldConfig or None
        obs = req.observation
        if obs.frame_stack > 1 and not (obs.rgb or obs.depth):
            raise ObservationContractError(
                f"metaworld: frame_stack={obs.frame_stack} requires at least "
                "one rgb/depth camera; a state-only observation cannot be "
                "frame-stacked."
            )
        return MetaWorldEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=mw.device if mw is not None else "cpu",
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            vectorization=mw.vectorization if mw is not None else "sync",
            use_one_hot=mw.use_one_hot if mw is not None else True,
            rgb_cameras=obs.rgb,
            depth_cameras=obs.depth,
            state=obs.state,
            image_size=obs.image_size if obs.image_size is not None else (84, 84),
            frame_stack=obs.frame_stack,
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.metaworld import make_metaworld_env

        return make_metaworld_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.metaworld import make_metaworld_env

        return make_metaworld_env(cls.resolve_config(req, is_eval=True))


register_env_backend("metaworld", MetaWorldBackend)
