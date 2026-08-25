"""Typed training configuration snapshots with sparse override sources."""

from __future__ import annotations

import inspect
import json
import math
import platform
import subprocess
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType, UnionType
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

ConfigStatus = Literal["preflight", "materialized"]
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
SourceKind = Literal[
    "preset",
    "CLI",
    "runtime-derived",
]


class ConfigError(ValueError):
    """Raised when a training configuration cannot be resolved safely."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class FieldSource:
    kind: SourceKind
    detail: str


@dataclass(frozen=True)
class EffectiveConfig:
    """One immutable snapshot of preflight or materialized run configuration."""

    schema_version: int
    status: ConfigStatus
    selection: Mapping[str, Any]
    inputs: Mapping[str, Any]
    active_environment: Mapping[str, Any]
    algorithm: Mapping[str, Any]
    derived: Mapping[str, Any]
    sources: Mapping[str, FieldSource | Mapping[str, Any]]
    runtime: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "selection",
            "inputs",
            "active_environment",
            "algorithm",
            "derived",
            "sources",
            "runtime",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def materialized(
        self,
        *,
        active_environment: Mapping[str, Any],
        algorithm: Mapping[str, Any],
        derived: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> EffectiveConfig:
        return replace(
            self,
            status="materialized",
            active_environment=dict(active_environment),
            algorithm=dict(algorithm),
            derived={**self.derived, **derived},
            runtime={**self.runtime, **runtime},
        )


@dataclass(frozen=True)
class PresetResult:
    values: Mapping[str, Any]
    paths: frozenset[str]
    path: str


def json_value(value: Any) -> Any:
    """Convert supported configuration values to deterministic JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return json_value(value.value)
    if inspect.isclass(value) or inspect.isfunction(value):
        return f"{value.__module__}.{value.__qualname__}"
    value_type = type(value)
    if value_type.__module__ in {"torch", "numpy"} and value_type.__qualname__ in {
        "device",
        "dtype",
    }:
        return str(value)
    return {"type": f"{value_type.__module__}.{value_type.__qualname__}"}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def effective_config_json(config: EffectiveConfig | Mapping[str, Any]) -> str:
    payload = json_value(config)
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def persist_effective_config(config: EffectiveConfig, path: Path) -> None:
    """Atomically write one effective-config snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(effective_config_json(config) + "\n", encoding="utf-8")
    temporary.replace(path)


def override_sources(
    sources: MutableMapping[str, FieldSource],
    paths: set[str] | frozenset[str],
    *,
    kind: SourceKind,
    detail: str,
) -> None:
    source = FieldSource(kind=kind, detail=detail)
    for path in paths:
        sources[path] = source


def _leaf_paths(mapping: Mapping[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            paths.update(_leaf_paths(value, path))
        else:
            paths.add(path)
    return paths


def _coerce_preset_value(value: Any, expected: Any, path: str) -> Any:
    origin = get_origin(expected)
    options = get_args(expected)
    if origin in (Union, UnionType):
        if value is None and type(None) in options:
            return None
        errors = []
        for option in options:
            if option is type(None):
                continue
            try:
                return _coerce_preset_value(value, option, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"Preset field {path!r} does not match {expected!r}.")
    if origin is Literal:
        if value not in options:
            raise ConfigError(
                f"Preset field {path!r} must be one of {options!r}, got {value!r}."
            )
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"Preset field {path!r} must be a list.")
        item_type = options[0] if options else Any
        return [_coerce_preset_value(item, item_type, f"{path}[]") for item in value]
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"Preset field {path!r} must be a YAML list.")
        item_type = options[0] if options else Any
        return tuple(
            _coerce_preset_value(item, item_type, f"{path}[]") for item in value
        )
    if expected is Any:
        return value
    if expected is Path and isinstance(value, str):
        return Path(value)
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if expected in {str, int, bool}:
        if type(value) is expected:
            return value
        raise ConfigError(
            f"Preset field {path!r} must have type {expected.__name__!r}, "
            f"got {type(value).__name__}."
        )
    if isinstance(expected, type) and isinstance(value, expected):
        return value
    raise ConfigError(
        f"Preset field {path!r} must have type {getattr(expected, '__name__', expected)!r}, "
        f"got {type(value).__name__}."
    )


def apply_strict_mapping(
    instance: Any, values: Mapping[str, Any], prefix: str = ""
) -> None:
    """Apply a YAML mapping recursively, rejecting unknown or ill-typed fields."""
    if not is_dataclass(instance) or isinstance(instance, type):
        raise TypeError("apply_strict_mapping() requires a dataclass instance")
    known = {field.name: field for field in fields(instance)}
    hints = get_type_hints(type(instance))
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in known:
            raise ConfigError(
                f"Unknown preset field {path!r} for {type(instance).__name__}."
            )
        current = getattr(instance, key)
        if is_dataclass(current) and not isinstance(current, type):
            if not isinstance(value, Mapping):
                raise ConfigError(f"Preset field {path!r} must be a mapping.")
            apply_strict_mapping(current, value, path)
            continue
        expected = hints.get(key, known[key].type)
        setattr(instance, key, _coerce_preset_value(value, expected, path))


def load_preset(path: str | Path) -> PresetResult:
    preset_path = Path(path)
    try:
        raw = yaml.load(
            preset_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except FileNotFoundError as exc:
        raise ConfigError(f"Preset file not found: {preset_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML preset {preset_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Preset {preset_path} must contain a mapping.")
    if "training_phase" in raw or "algorithm" in raw:
        raise ConfigError(
            "Preset files contain args only; select phase and algorithm on the CLI."
        )
    normalized = {str(key): value for key, value in raw.items()}
    return PresetResult(
        normalized, frozenset(_leaf_paths(normalized)), str(preset_path)
    )


def runtime_metadata(*, argv: list[str] | None = None) -> dict[str, Any]:
    """Return stable, low-cost runtime metadata without importing simulators."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_SOURCE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "argv": list(sys.argv if argv is None else argv),
        "git_commit": commit,
        "python": platform.python_version(),
    }


def _dataclass_leaf_paths(value: Any, prefix: str) -> set[str]:
    result: set[str] = set()
    for field in fields(value):
        path = f"{prefix}.{field.name}"
        item = getattr(value, field.name)
        if is_dataclass(item) and not isinstance(item, type):
            result.update(_dataclass_leaf_paths(item, path))
        else:
            result.add(path)
    return result


def inactive_config_paths(args: Any) -> dict[str, str]:
    """Return inactive Args leaf paths and concise reasons."""
    from rl_garden.common.cli_args import VisionArgs
    from rl_garden.common.env_args import EnvBackendArgs

    inactive: dict[str, str] = {}
    backend_names = {
        field.name for field in fields(EnvBackendArgs) if field.name != "env_backend"
    }
    selected_backend = getattr(args, "env_backend", None)
    for backend_name in backend_names:
        backend = getattr(args, backend_name, None)
        if backend_name != selected_backend and is_dataclass(backend):
            for path in _dataclass_leaf_paths(backend, backend_name):
                inactive[path] = f"environment backend is {selected_backend!r}"

    visual_names = {field.name for field in fields(VisionArgs)} - {"obs_mode"}
    obs_mode = getattr(args, "obs_mode", None)
    encoder = getattr(args, "encoder", None)
    if obs_mode == "state":
        for name in visual_names:
            inactive[name] = "obs_mode is 'state'"
        return inactive

    if encoder is not None and not str(encoder).startswith("resnet"):
        for name in (
            "pretrained_weights",
            "freeze_resnet_encoder",
            "freeze_resnet_backbone",
        ):
            inactive[name] = f"encoder is {encoder!r}"
    if encoder is not None and encoder != "plain_conv":
        for name in (
            "plain_conv_weight_init",
            "plain_conv_last_act",
            "plain_conv_pooling",
        ):
            inactive[name] = f"encoder is {encoder!r}"
    if encoder is not None and encoder != "vit":
        for name in visual_names:
            if name.startswith("vit_"):
                inactive[name] = f"encoder is {encoder!r}"
    if getattr(args, "critic_encoder", None) is None:
        for name in ("critic_image_keys", "critic_include_state"):
            inactive[name] = "critic_encoder is not set"
    return inactive


def resolve_active_environment(args: Any) -> dict[str, Any]:
    """Serialize and statically validate the selected environment backend."""
    backend_name = getattr(args, "env_backend", None)
    if backend_name is None:
        return {}
    try:
        backend_config = getattr(args, backend_name)
    except AttributeError as exc:
        from rl_garden.common.env_args import EnvBackendArgs

        available = sorted(
            field.name
            for field in fields(EnvBackendArgs)
            if field.name != "env_backend"
        )
        raise ConfigError(
            f"Unknown env backend {backend_name!r}. Available: {available}."
        ) from exc
    backend_values = json_value(backend_config)
    if isinstance(backend_values, dict):
        for key, value in backend_values.items():
            if key.endswith("_kwargs_json") and value:
                try:
                    parsed_json = json.loads(value)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ConfigError(
                        f"Invalid {backend_name}.{key}: expected a JSON object."
                    ) from exc
                if not isinstance(parsed_json, dict):
                    raise ConfigError(
                        f"Invalid {backend_name}.{key}: expected a JSON object."
                    )
    return {"backend": backend_name, "config": backend_values}


def _remove_path(mapping: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    current = mapping
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            return
        current = nested
    current.pop(parts[-1], None)


def resolve_effective_config(
    args: Any,
    *,
    training_phase: str,
    algorithm: str,
    sources: Mapping[str, FieldSource | Mapping[str, Any]],
    active_environment: Mapping[str, Any],
    algorithm_config: Mapping[str, Any] | None = None,
    derived: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> EffectiveConfig:
    """Build a deterministic preflight snapshot from resolved training args."""
    inputs = json_value(args)
    if not isinstance(inputs, dict):
        raise TypeError("Training args must serialize to a mapping.")
    inactive = inactive_config_paths(args)
    overridden_inactive = sorted(path for path in sources if path in inactive)
    if overridden_inactive:
        details = "; ".join(
            f"{path!r} is inactive because {inactive[path]}"
            for path in overridden_inactive
        )
        raise ConfigError(f"Explicit override {details}.")
    for path in inactive:
        _remove_path(inputs, path)

    from rl_garden.common.env_args import EnvBackendArgs

    for field in fields(EnvBackendArgs):
        if field.name != "env_backend":
            inputs.pop(field.name, None)
    return EffectiveConfig(
        schema_version=3,
        status="preflight",
        selection={"training_phase": training_phase, "algorithm": algorithm},
        inputs=inputs,
        active_environment=dict(active_environment),
        algorithm=dict(algorithm_config or {}),
        derived=dict(derived or {}),
        sources=dict(sources),
        runtime=dict(runtime or runtime_metadata()),
    )
