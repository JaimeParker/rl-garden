"""Shared state and serializers for training dry runs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

from rl_garden.common.effective_config import (
    EffectiveConfig,
    effective_config_json,
    json_value,
    resolve_active_environment,
    resolve_effective_config,
    runtime_metadata,
)


@dataclass(frozen=True)
class InspectionSession:
    preflight: EffectiveConfig
    dry_run: bool


_SESSION: ContextVar[InspectionSession | None] = ContextVar(
    "rl_garden_inspection_session", default=None
)
_AGENT_BUILD: ContextVar[dict[str, Any] | None] = ContextVar(
    "rl_garden_agent_build", default=None
)


@contextmanager
def config_session(preflight: EffectiveConfig, *, dry_run: bool) -> Iterator[None]:
    token = _SESSION.set(InspectionSession(preflight, dry_run))
    build_token = _AGENT_BUILD.set(None)
    try:
        yield
    finally:
        _AGENT_BUILD.reset(build_token)
        _SESSION.reset(token)


def is_dry_run() -> bool:
    session = _SESSION.get()
    return session is not None and session.dry_run


def has_config_session() -> bool:
    return _SESSION.get() is not None


def standalone_preflight(
    args: Any,
    *,
    training_phase: str,
    algorithm: str,
    sources: Mapping[str, Any] | None = None,
    derived: Mapping[str, Any] | None = None,
) -> EffectiveConfig:
    return resolve_effective_config(
        args,
        training_phase=training_phase,
        algorithm=algorithm,
        sources=dict(sources or {}),
        active_environment=resolve_active_environment(args),
        algorithm_config={},
        derived=derived,
        runtime=runtime_metadata(),
    )


def prepare_standalone(
    args: Any,
    *,
    registry: Any,
    training_phase: str,
    algorithm: str,
) -> tuple[Any, EffectiveConfig]:
    """Apply the same normalization pipeline used by registry CLI dispatch."""
    sources = {}
    normalized, derived = registry._normalize_runtime(args, sources)
    return normalized, standalone_preflight(
        normalized,
        training_phase=training_phase,
        algorithm=algorithm,
        sources=sources,
        derived=derived,
    )


def current_preflight() -> EffectiveConfig:
    session = _SESSION.get()
    if session is None:
        raise RuntimeError("No active training configuration session.")
    return session.preflight


def run_preflight(derived: Mapping[str, Any]) -> EffectiveConfig:
    preflight = current_preflight()
    return replace(preflight, derived={**preflight.derived, **derived})


def _space_summary(space: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": f"{type(space).__module__}.{type(space).__qualname__}"
    }
    for name in ("shape", "dtype", "n"):
        if hasattr(space, name):
            result[name] = json_value(getattr(space, name))
    if hasattr(space, "spaces"):
        spaces = space.spaces
        if isinstance(spaces, Mapping):
            result["spaces"] = {
                key: _space_summary(value) for key, value in spaces.items()
            }
        else:
            result["spaces"] = [_space_summary(value) for value in spaces]
    return result


def construct_agent(target: Any, /, **kwargs: Any) -> Any:
    """Capture the exact constructor call used by a registered builder."""
    serialized = {
        name: (
            f"<{name}>" if name in {"env", "eval_env", "logger"} else json_value(value)
        )
        for name, value in kwargs.items()
    }
    _AGENT_BUILD.set(
        {
            "target": f"{target.__module__}.{target.__qualname__}",
            "constructor_kwargs": serialized,
        }
    )
    return target(**kwargs)


def _algorithm_summary() -> dict[str, Any]:
    record = _AGENT_BUILD.get()
    if record is None:
        raise RuntimeError(
            "Registered builders must construct agents through construct_agent()."
        )
    return {
        "target": record["target"],
        "constructor_kwargs": record["constructor_kwargs"],
    }


def materialize_config(
    *,
    env_request: Any,
    env: Any,
    eval_env: Any,
    agent: Any,
    derived: Mapping[str, Any] | None = None,
) -> EffectiveConfig:
    session = _SESSION.get()
    if session is None:
        raise RuntimeError(
            "emit_materialized_config() requires an active dry-run session"
        )
    observation_space = getattr(env, "single_observation_space", None)
    action_space = getattr(env, "single_action_space", None)
    runtime = {
        "dry_run": session.dry_run,
        "train_observation_space": (
            _space_summary(observation_space) if observation_space is not None else None
        ),
        "train_action_space": (
            _space_summary(action_space) if action_space is not None else None
        ),
        "eval_environment_created": eval_env is not None,
        "device": json_value(getattr(agent, "device", None)),
    }
    concrete_configs: Any = None
    backend_name = session.preflight.active_environment.get("backend")
    if backend_name is not None and hasattr(env_request, "env_id"):
        from rl_garden.envs.backend_registry import materialize_backend_configs

        concrete_configs = json_value(
            materialize_backend_configs(str(backend_name), env_request)
        )
    active_environment = {
        **session.preflight.active_environment,
        "request": json_value(env_request),
        "materialized_configs": concrete_configs,
    }
    return session.preflight.materialized(
        active_environment=active_environment,
        algorithm=_algorithm_summary(),
        derived=dict(derived or {}),
        runtime=runtime,
    )


def emit_materialized_config(**kwargs: Any) -> EffectiveConfig:
    config = materialize_config(**kwargs)
    print(effective_config_json(config))
    return config
