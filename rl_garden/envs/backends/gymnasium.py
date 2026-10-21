"""Generic ``gymnasium`` env backend — registered as ``"gymnasium"``.

Trains on any registered, single-instance ``gymnasium.Env`` (``--env_id``)
whose registering module is named in ``--gymnasium.import_modules``; see
``rl_garden.envs.gymnasium.env`` for the vectorization, registration, and
observation rules, and ``docs/guides/gymnasium-custom-envs.md`` for usage.
"""

from __future__ import annotations

import json

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)

_VECTORIZATION_MODES = ("sync", "async")


class GymnasiumBackend(EnvBackend):
    api_version = 2
    config_field = "gymnasium"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.gymnasium.config import GymnasiumEnvConfig
        from rl_garden.observations.schema import ObservationContractError

        gc = req.backend_config  # GymnasiumConfig or None
        vectorization = gc.vectorization if gc is not None else "sync"
        if vectorization not in _VECTORIZATION_MODES:
            raise ValueError(
                "gymnasium backend: vectorization must be 'sync' or 'async', "
                f"got {vectorization!r}"
            )
        env_kwargs = (
            json.loads(gc.env_kwargs_json) if gc is not None and gc.env_kwargs_json else {}
        )
        if not isinstance(env_kwargs, dict):
            raise ValueError(  # noqa: TRY004 - invalid configuration value
                "gymnasium backend: env_kwargs_json must encode a JSON object, "
                f"got {type(env_kwargs).__name__}"
            )
        import_modules = tuple(gc.import_modules) if gc is not None else ()
        for name in import_modules:
            if not all(part.isidentifier() for part in name.split(".")):
                raise ValueError(
                    f"gymnasium backend: import_modules entry {name!r} is not a "
                    "dotted Python module path"
                )
        obs = req.observation
        if obs.image_size is not None:
            raise ObservationContractError(
                f"gymnasium backend cannot resize images; image_size={obs.image_size!r} "
                "must be unset (the env's own resolution is used)"
            )
        if obs.frame_stack > 1 and not obs.is_visual:
            raise ObservationContractError(
                f"gymnasium backend: frame_stack={obs.frame_stack} requires an "
                "rgb/depth observation (--obs.rgb/--obs.depth)"
            )
        return GymnasiumEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=gc.device if gc is not None else "cpu",
            import_modules=import_modules,
            env_kwargs=env_kwargs,
            vectorization=vectorization,
            image_keys=obs.image_keys,
            frame_stack=obs.frame_stack,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.gymnasium import make_gymnasium_env

        return make_gymnasium_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.gymnasium import make_gymnasium_env

        return make_gymnasium_env(cls.resolve_config(req, is_eval=True))


register_env_backend("gymnasium", GymnasiumBackend)
