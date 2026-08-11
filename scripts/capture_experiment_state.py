#!/usr/bin/env python3
"""Capture local source state for experiment reproducibility."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HASH_PATHS = [
    "rl_garden/algorithms/iql.py",
    "rl_garden/algorithms/off2on_iql.py",
    "rl_garden/training/off2on/iql.py",
    "rl_garden/training/off2on/_args.py",
    "rl_garden/training/off2on/_runner.py",
    "scripts/run_iql_fixed_mixing.py",
    "tools/reproductions/iql_fixed_mixing.patch",
]


def _run_git(args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hash-path", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    git_status = _run_git(["status", "--short"])
    git_diff = _run_git(["diff", "--binary"])
    git_diff_cached = _run_git(["diff", "--cached", "--binary"])
    head_output = _run_git(["rev-parse", "HEAD"])
    git_available = head_output is not None
    (output_dir / "git_status.txt").write_text(git_status or "")
    (output_dir / "git_diff.patch").write_text(git_diff or "")
    (output_dir / "git_diff_cached.patch").write_text(git_diff_cached or "")
    if not git_available:
        (output_dir / "git_unavailable.txt").write_text(
            f"No git metadata available under {REPO_ROOT}.\n"
        )
    hash_paths = [*DEFAULT_HASH_PATHS, *args.hash_path]
    hashes = {}
    for rel in hash_paths:
        path = REPO_ROOT / rel
        if path.is_file():
            hashes[rel] = _sha256(path)
    metadata = {
        "repo_root": str(REPO_ROOT),
        "git_available": git_available,
        "head": head_output.strip() if head_output is not None else None,
        "hashes": hashes,
    }
    (output_dir / "source_state.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
