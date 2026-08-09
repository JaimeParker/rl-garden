"""Shared state and serializers for training dry runs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

from rl_garden.common.effective_config import (
    ConfigError,
    EffectiveConfig,
    default_provenance,
    effective_config_json,
    json_value,
    resolve_effective_config,
    runtime_metadata,
)


@dataclass(frozen=True)
class InspectionSession:
    preflight: EffectiveConfig
    dry_run: bool
    contract: Any | None


_SESSION: ContextVar[InspectionSession | None] = ContextVar(
    "rl_garden_inspection_session", default=None
)
_AGENT_BUILD: ContextVar[dict[str, Any] | None] = ContextVar(
    "rl_garden_agent_build", default=None
)


@contextmanager
def config_session(
    preflight: EffectiveConfig, *, dry_run: bool, contract: Any | None = None
) -> Iterator[None]:
    token = _SESSION.set(InspectionSession(preflight, dry_run, contract))
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
    contract: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
    derived: Mapping[str, Any] | None = None,
) -> EffectiveConfig:
    active_environment: dict[str, Any] = {}
    backend_name = getattr(args, "env_backend", None)
    if backend_name is not None and hasattr(args, backend_name):
        active_environment = {
            "backend": backend_name,
            "config": json_value(getattr(args, backend_name)),
        }
    resolved_provenance = dict(provenance or default_provenance(args))
    algorithm_config: dict[str, Any] = {}
    if contract is not None:
        resolved_provenance = contract.validate_active(
            args, contract.apply(resolved_provenance)
        )
        active_paths = {
            path for path, field in resolved_provenance.items() if field.active
        }
        try:
            implicit_defaults = (
                contract.constructor_defaults() if contract.mode == "strict" else {}
            )
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid config contract: {exc}") from exc
        algorithm_config = {
            "target": contract.target,
            "mode": contract.mode,
            "constructor_kwargs": {},
            "field_mappings": contract.field_mappings(active_paths),
            "implicit_defaults": implicit_defaults,
        }
    return resolve_effective_config(
        args,
        training_phase=training_phase,
        algorithm=algorithm,
        provenance=resolved_provenance,
        active_environment=active_environment,
        algorithm_config=algorithm_config,
        derived=derived,
        runtime=runtime_metadata(),
    )


def prepare_standalone(
    args: Any,
    *,
    registry: Any,
    training_phase: str,
    algorithm: str,
    contract: Any,
) -> tuple[Any, EffectiveConfig]:
    """Apply the same normalization pipeline used by registry CLI dispatch."""
    provenance = default_provenance(args)
    normalized, derived = registry._normalize_runtime(args, provenance)
    return normalized, standalone_preflight(
        normalized,
        training_phase=training_phase,
        algorithm=algorithm,
        contract=contract,
        provenance=provenance,
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
    env = kwargs.get("env")
    observation_space = getattr(env, "single_observation_space", None)
    _AGENT_BUILD.set(
        {
            "target": f"{target.__module__}.{target.__qualname__}",
            "target_type": target,
            "constructor_kwargs": serialized,
            "visual_observation": isinstance(
                getattr(observation_space, "spaces", None), Mapping
            ),
        }
    )
    return target(**kwargs)


def validate_constructor_coverage(
    consumption_by_path: Mapping[str, Any],
    constructor_kwargs: Mapping[str, Any],
    *,
    declared_target: type,
    constructed_target: type,
    visual_observation: bool,
) -> None:
    """Reject a constructed agent that does not satisfy its strict contract."""
    if constructed_target is not declared_target:
        raise ConfigError(
            "Config contract target mismatch: declared "
            f"{declared_target.__module__}.{declared_target.__qualname__}, constructed "
            f"{constructed_target.__module__}.{constructed_target.__qualname__}"
        )
    if not consumption_by_path:
        return
    from rl_garden.training.config_contract import check_constructor_coverage

    inactive_clusters = (
        frozenset() if visual_observation else frozenset({"visual_encoder"})
    )
    violations = check_constructor_coverage(
        consumption_by_path,
        constructor_kwargs,
        inactive_clusters=inactive_clusters,
    )
    if violations:
        raise ConfigError(
            "Config contract / constructor mismatch:\n" + "\n".join(violations)
        )


def _algorithm_summary() -> dict[str, Any]:
    record = _AGENT_BUILD.get()
    if record is None:
        raise RuntimeError(
            "Registered builders must construct agents through construct_agent()."
        )
    preflight = current_preflight()
    session = _SESSION.get()
    assert session is not None
    target = record["target_type"]
    passed = set(record["constructor_kwargs"])
    implicit_defaults: dict[str, Any] = {}
    from rl_garden.training._constructor_introspection import (
        inspect_constructor_parameters,
    )

    for name, parameter in inspect_constructor_parameters(target).items():
        if name not in passed and parameter.default is not parameter.empty:
            implicit_defaults[name] = json_value(parameter.default)
    if session.contract is not None and session.contract.mode == "strict":
        active_paths = {
            path
            for path, field in preflight.provenance.items()
            if getattr(field, "active", True)
        }
        validate_constructor_coverage(
            session.contract.consumption_map(active_paths),
            record["constructor_kwargs"],
            declared_target=session.contract.target_type(),
            constructed_target=record["target_type"],
            visual_observation=record["visual_observation"],
        )
    return {
        "target": record["target"],
        "mode": preflight.algorithm.get("mode", "strict"),
        "constructor_kwargs": record["constructor_kwargs"],
        "field_mappings": preflight.algorithm.get("field_mappings", {}),
        "implicit_defaults": implicit_defaults,
        # Reserved for future composite agents; retained for config JSON
        # compatibility even though current contracts do not populate it.
        "components": preflight.algorithm.get("components", {}),
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
