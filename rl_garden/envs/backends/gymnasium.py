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


def _parse_obs_key_map(entries: tuple[str, ...], expected_keys: tuple[str, ...]) -> dict[str, str]:
    from rl_garden.observations.schema import ObservationContractError

    key_map: dict[str, str] = {}
    for entry in entries:
        src, sep, dst = entry.partition("=")
        if not sep or not src or not dst or "=" in dst:
            raise ValueError(f"gymnasium backend: obs_key_map entry {entry!r} must be 'src=dst'")
        duplicate = src if src in key_map else dst if dst in key_map.values() else None
        if duplicate is not None:
            raise ValueError(
                "gymnasium backend: obs_key_map has duplicate source/destination "
                f"key {duplicate!r}"
            )
        if dst not in expected_keys:
            raise ObservationContractError(
                f"gymnasium backend: obs_key_map destination {dst!r} is not an expected "
                f"observation key {sorted(expected_keys)!r}"
            )
        key_map[src] = dst
    return key_map


def _validate_vector_env_factory(spec: str) -> None:
    module, sep, function = spec.partition(":")
    if not (
        sep
        and module
        and all(part.isidentifier() for part in module.split("."))
        and function.isidentifier()
    ):
        raise ValueError(
            f"gymnasium backend: vector_env_factory {spec!r} must be 'pkg.module:function'"
        )


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
        vector_env_factory = gc.vector_env_factory if gc is not None else ""
        vector_env_returns_torch = gc.vector_env_returns_torch if gc is not None else False
        if vector_env_factory:
            _validate_vector_env_factory(vector_env_factory)
            if vectorization != "sync":
                raise ValueError(
                    "gymnasium backend: vectorization is not used with vector_env_factory"
                )
        elif vector_env_returns_torch:
            raise ValueError(
                "gymnasium backend: vector_env_returns_torch requires vector_env_factory"
            )
        obs = req.observation
        obs_key_map = _parse_obs_key_map(
            tuple(gc.obs_key_map) if gc is not None else (), obs.expected_keys
        )
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
            obs_key_map=obs_key_map,
            vector_env_factory=vector_env_factory,
            vector_env_returns_torch=vector_env_returns_torch,
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
