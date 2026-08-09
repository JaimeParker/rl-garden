"""Resolved training configuration serialization and persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rl_garden.common.effective_config import json_value


def _json_value(value: Any) -> Any:
    return json_value(value)


def resolved_run_config(
    args: Any,
    *,
    training_phase: str,
    algorithm: str,
    run_name: str | None = None,
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
