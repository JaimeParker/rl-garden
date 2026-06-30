"""Resolved training configuration serialization and persistence."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def resolved_run_config(
    args: Any,
    *,
    training_phase: str,
    algorithm: str,
    run_name: Optional[str] = None,
) -> dict[str, Any]:
    config = {
        "training_phase": training_phase,
        "algorithm": algorithm,
        "args": _json_value(args),
    }
    if run_name is not None:
        config["run_name"] = run_name
    return config


def resolved_config_json(config: Mapping[str, Any]) -> str:
    return json.dumps(config, indent=2, sort_keys=True, allow_nan=False)


def persist_resolved_config(
    args: Any,
    *,
    training_phase: str,
    algorithm: str,
    run_name: str,
    log_dir: str,
) -> dict[str, Any]:
    config = resolved_run_config(
        args,
        training_phase=training_phase,
        algorithm=algorithm,
        run_name=run_name,
    )
    run_dir = Path(log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        resolved_config_json(config) + "\n",
        encoding="utf-8",
    )
    return config
