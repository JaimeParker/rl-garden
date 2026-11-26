"""Shared state and serializers for training dry runs, and the agent-construction
seam every registered builder goes through (``construct_agent``)."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rl_garden.common.effective_config import (
    EffectiveConfig,
    effective_config_json,
    import_path_of,
    json_value,
    resolve_active_environment,
    resolve_effective_config,
    runtime_metadata,
)


_logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class CheckpointEvaluation:
    """The checkpoint an agent is being rebuilt for, and the device it runs on."""

    path: Path
    checkpoint: Mapping[str, Any]
    device: str


_EVALUATION: ContextVar[CheckpointEvaluation | None] = ContextVar(
    "rl_garden_checkpoint_evaluation", default=None
)


@contextmanager
def checkpoint_evaluation(evaluation: CheckpointEvaluation) -> Iterator[None]:
    """Scope in which a registered builder rebuilds the agent of
    ``evaluation``'s checkpoint: ``construct_agent`` and ``get_device("auto")``
    put it on ``evaluation.device`` and ``current_evaluation()`` returns
    ``evaluation``."""
    from rl_garden.common.utils import auto_device

    token = _EVALUATION.set(evaluation)
    try:
        with auto_device(evaluation.device):
            yield
    finally:
        _EVALUATION.reset(token)


def current_evaluation() -> CheckpointEvaluation | None:
    """The checkpoint evaluation the agent is being rebuilt for, ``None`` when
    it is being built to train."""
    return _EVALUATION.get()


def checkpoint_to_restore(args: Any) -> Mapping[str, Any] | None:
    """The checkpoint the agent under construction will load: the evaluated
    one, or ``args.load_checkpoint`` (read on CPU); ``None`` for a fresh run.
    For a builder that needs to know what the checkpoint embeds before the
    agent exists."""
    evaluation = current_evaluation()
    if evaluation is not None:
        return evaluation.checkpoint
    if args.load_checkpoint is None:
        return None
    from rl_garden.common.checkpoint import load_checkpoint_file

    return load_checkpoint_file(args.load_checkpoint, map_location="cpu")


def warn_if_superseded(
    flag: str, path: str | None, embedded: Mapping[str, Any]
) -> None:
    """Warn when the weights ``path`` (an algorithm checkpoint's policy) holds
    differ from the ``embedded`` ones a resumed checkpoint restores instead."""
    if current_evaluation() is not None or not path or not Path(path).exists():
        return
    import torch

    from rl_garden.common.checkpoint import load_checkpoint_file

    policy = load_checkpoint_file(path, map_location="cpu")["state"]["policy"]
    if not all(
        key in policy and torch.equal(policy[key], value.cpu())
        for key, value in embedded.items()
    ):
        _logger.warning(
            "%s %s is not the network embedded in the resumed checkpoint; the embedded one is used",
            flag,
            path,
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


#: Constructor keywords (and ``policy_kwargs`` keys) whose callables the
#: training CLI takes as import paths (``--actor-factory``,
#: ``--actor-extractor``, ``--critic-extractor``; see
#: ``rl_garden.common.cli_args.PolicyModuleArgs``). They are recorded as the
#: ``"pkg.mod:attr"`` path the flag takes, every other value as ``json_value``.
_IMPORT_PATH_KWARGS = frozenset({"actor_factory"})
_IMPORT_PATH_POLICY_KWARGS = frozenset({"actor_extractor_class", "critic_extractor_class"})


def _import_path_or_json(value: Any) -> Any:
    try:
        return import_path_of(value)
    except ValueError:
        return json_value(value)


def _constructor_kwarg_value(name: str, value: Any) -> Any:
    if name in {"env", "eval_env", "logger"}:
        return f"<{name}>"
    if name in _IMPORT_PATH_KWARGS and value is not None:
        return _import_path_or_json(value)
    if name == "policy_kwargs" and isinstance(value, Mapping):
        return {
            str(key): (
                _import_path_or_json(item)
                if key in _IMPORT_PATH_POLICY_KWARGS and item is not None
                else json_value(item)
            )
            for key, item in value.items()
        }
    return json_value(value)


def _accepts_device(target: Any) -> bool:
    """Whether ``target(device=...)`` reaches a ``device`` parameter, following
    ``**kwargs`` constructors to the parent they forward to."""
    if not inspect.isclass(target):
        return "device" in inspect.signature(target).parameters
    for cls in target.__mro__:
        init = vars(cls).get("__init__")
        if init is None:
            continue
        parameters = inspect.signature(init).parameters
        if "device" in parameters:
            return True
        if not any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
            return False
    return False


def construct_agent(target: Any, /, **kwargs: Any) -> Any:
    """Capture the exact constructor call used by a registered builder. While
    a checkpoint is being evaluated (``checkpoint_evaluation``) its device
    replaces the builder's."""
    evaluation = current_evaluation()
    if evaluation is not None and _accepts_device(target):
        kwargs["device"] = evaluation.device
    serialized = {name: _constructor_kwarg_value(name, value) for name, value in kwargs.items()}
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
