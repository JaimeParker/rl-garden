"""Typed, provenance-aware training configuration snapshots."""

from __future__ import annotations

import inspect
import json
import math
import os
import platform
import re
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
    "dataclass",
    "subclass",
    "preset",
    "RLG_*",
    "launcher",
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
class SourceRecord:
    kind: SourceKind
    detail: str


@dataclass(frozen=True)
class FieldProvenance:
    defined_at: str
    owner: str
    source: SourceRecord
    history: tuple[SourceRecord, ...]
    active: bool = True
    active_when: str = "always"
    mapped_to: str = ""


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
    provenance: Mapping[str, FieldProvenance | Mapping[str, Any]]
    runtime: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "selection",
            "inputs",
            "active_environment",
            "algorithm",
            "derived",
            "provenance",
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


def _field_definition(args_cls: type, name: str) -> tuple[str, SourceKind]:
    definitions = [
        cls for cls in args_cls.__mro__ if name in getattr(cls, "__annotations__", {})
    ]
    if not definitions:
        return f"{args_cls.__module__}.{args_cls.__qualname__}", "dataclass"
    owner = definitions[0]
    try:
        source_path = inspect.getsourcefile(owner)
        if source_path is None:
            path = owner.__module__
        else:
            resolved_path = Path(source_path).resolve()
            try:
                path = str(resolved_path.relative_to(_SOURCE_ROOT))
            except ValueError:
                path = str(resolved_path)
        source_lines, class_line = inspect.getsourcelines(owner)
        field_line = next(
            (
                class_line + offset
                for offset, source_line in enumerate(source_lines)
                if re.match(rf"\s+{re.escape(name)}\s*:", source_line)
            ),
            class_line,
        )
        location = f"{path}:{field_line}"
    except (OSError, TypeError):
        location = f"{owner.__module__}.{owner.__qualname__}"
    return location, "subclass" if len(definitions) > 1 else "dataclass"


def default_provenance(args: Any) -> dict[str, FieldProvenance]:
    result: dict[str, FieldProvenance] = {}

    def visit(value: Any, prefix: str, root_cls: type) -> None:
        for field in fields(value):
            path = f"{prefix}.{field.name}" if prefix else field.name
            item = getattr(value, field.name)
            definition, kind = _field_definition(
                root_cls if not prefix else type(value), field.name
            )
            source = SourceRecord(kind=kind, detail=definition)
            result[path] = FieldProvenance(
                defined_at=definition,
                owner="unclassified",
                source=source,
                history=(source,),
            )
            if is_dataclass(item) and not isinstance(item, type):
                visit(item, path, type(item))

    visit(args, "", type(args))
    return result


def override_provenance(
    provenance: MutableMapping[str, FieldProvenance],
    paths: set[str] | frozenset[str],
    *,
    kind: SourceKind,
    detail: str,
) -> None:
    record = SourceRecord(kind=kind, detail=detail)
    for path in paths:
        current = provenance.get(path)
        if current is None:
            continue
        provenance[path] = replace(
            current,
            source=record,
            history=(*current.history, record),
        )


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
        "launcher": os.getenv("RLG_LAUNCHER"),
        "git_commit": commit,
        "python": platform.python_version(),
    }


def resolve_effective_config(
    args: Any,
    *,
    training_phase: str,
    algorithm: str,
    provenance: Mapping[str, FieldProvenance],
    active_environment: Mapping[str, Any],
    algorithm_config: Mapping[str, Any] | None = None,
    derived: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> EffectiveConfig:
    """Build a deterministic preflight snapshot from resolved training args."""
    inputs = json_value(args)
    if not isinstance(inputs, dict):
        raise TypeError("Training args must serialize to a mapping.")
    selected_backend = getattr(args, "env_backend", None)
    if selected_backend is not None:
        backend_names = {
            field.name
            for field in fields(args)
            if is_dataclass(getattr(args, field.name, None)) and field.name != "logging"
        }
        for backend_name in backend_names:
            if backend_name != selected_backend:
                inputs.pop(backend_name, None)

    def remove_path(mapping: dict[str, Any], path: str) -> None:
        parts = path.split(".")
        current = mapping
        for part in parts[:-1]:
            nested = current.get(part)
            if not isinstance(nested, dict):
                return
            current = nested
        current.pop(parts[-1], None)

    for path, field in provenance.items():
        if not field.active:
            remove_path(inputs, path)
    return EffectiveConfig(
        schema_version=2,
        status="preflight",
        selection={"training_phase": training_phase, "algorithm": algorithm},
        inputs=inputs,
        active_environment=dict(active_environment),
        algorithm=dict(algorithm_config or {}),
        derived=dict(derived or {}),
        provenance=dict(provenance),
        runtime=dict(runtime or runtime_metadata()),
    )
