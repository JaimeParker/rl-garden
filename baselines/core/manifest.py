"""Reader for ``baselines/baselines.yaml``.

The manifest is the single source of truth for what's registered as a
runnable baseline (git submodule under ``3rd_party/`` + a dedicated venv +
an ``baselines/<name>/`` orchestrator) versus a read-only reference
clone. ``scripts/install_baseline.sh`` reads it through this module rather
than re-parsing YAML in bash.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "baselines" / "baselines.yaml"


@dataclass(frozen=True)
class BaselineEntry:
    name: str
    path: str
    remote_url: str
    python_version: str
    venv_name: str
    requirements_files: list[str]
    extra_pip: list[str]
    d4rl_fork: str | None
    integration_module: str
    invocation_cwd: str | None
    sys_path_insert: bool
    notes: str


def _entry_from_dict(name: str, data: dict[str, Any]) -> BaselineEntry:
    invocation = data.get("invocation", {})
    return BaselineEntry(
        name=name,
        path=data["path"],
        remote_url=data["remote_url"],
        python_version=str(data["python_version"]),
        venv_name=data["venv_name"],
        requirements_files=list(data.get("requirements_files", [])),
        extra_pip=list(data.get("extra_pip", [])),
        d4rl_fork=data.get("d4rl_fork"),
        integration_module=data["integration_module"],
        invocation_cwd=invocation.get("cwd"),
        sys_path_insert=bool(invocation.get("sys_path_insert", False)),
        notes=str(data.get("notes", "")).strip(),
    )


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, BaselineEntry]:
    """Load the ``runnable`` baselines from ``baselines.yaml`` as a name->entry dict."""
    raw = yaml.safe_load(Path(path).read_text())
    runnable = raw.get("runnable", {})
    return {name: _entry_from_dict(name, data) for name, data in runnable.items()}


def get_baseline(name: str, path: Path = DEFAULT_MANIFEST_PATH) -> BaselineEntry:
    """Look up one runnable baseline by name, raising with valid names on a typo."""
    entries = load_manifest(path)
    if name not in entries:
        valid = ", ".join(sorted(entries))
        raise KeyError(f"Unknown baseline {name!r}. Valid names: {valid}")
    return entries[name]
