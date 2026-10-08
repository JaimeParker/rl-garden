"""Launch an unmodified official FQL JAX OGBench training recipe.

This module is intentionally a process launcher rather than an adapter for
the upstream agent.  Keeping FQL in its own Python environment prevents its
JAX/OGBench/Gymnasium stack from being mixed with rl-garden's PyTorch stack.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from .presets import FQLPreset, PRESETS


FQL_REFERENCE_REVISION = "e8cd16eb490332924dfa2492097219f181765933"
OGBENCH_VERSION = "1.1.0"
GYMNASIUM_VERSION = "0.29.1"
DEFAULT_FQL_SOURCE = Path(__file__).resolve().parents[2] / "3rd_party" / "fql"


def _git_output(source: Path, *args: str) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source), *args], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
    except FileNotFoundError:
        return 127, ""
    return completed.returncode, completed.stdout.strip()


def inspect_source(source: Path) -> dict[str, object]:
    """Describe source provenance, warning rather than blocking on git drift."""
    if not source.is_dir():
        raise FileNotFoundError(
            f"FQL source is missing: {source}. Run `git clone "
            f"https://github.com/seohongpark/fql.git {source}`; optionally check out "
            f"the reference-tested revision {FQL_REFERENCE_REVISION}. This launcher never clones or installs it."
        )
    if not (source / "main.py").is_file():
        raise FileNotFoundError(f"FQL source has no main.py: {source}")
    top_level_status, top_level = _git_output(source, "rev-parse", "--show-toplevel")
    warnings: list[str] = []
    if top_level_status != 0 or Path(top_level).resolve() != source.resolve():
        warnings.append("source is not an independent git worktree; revision and dirty state are unavailable")
        source_info = {
            "path": str(source), "expected_revision": FQL_REFERENCE_REVISION,
            "revision": None, "dirty": None, "verified": False, "warnings": warnings,
        }
        print(f"WARNING: FQL source provenance: {warnings[0]}", file=sys.stderr)
        return source_info
    revision_status, revision = _git_output(source, "rev-parse", "HEAD")
    dirty_status, dirty = _git_output(source, "status", "--porcelain")
    if revision_status != 0:
        warnings.append("could not read source git revision")
    elif revision != FQL_REFERENCE_REVISION:
        warnings.append(f"source revision differs from reference {FQL_REFERENCE_REVISION}: {revision}")
    if dirty_status != 0:
        warnings.append("could not read source dirty state")
    elif dirty:
        warnings.append("source worktree has local changes")
    source_info = {
        "path": str(source),
        "expected_revision": FQL_REFERENCE_REVISION,
        "revision": revision if revision_status == 0 else None,
        "dirty": dirty if dirty_status == 0 else None,
        "verified": not warnings,
        "warnings": warnings,
    }
    for warning in warnings:
        print(f"WARNING: FQL source provenance: {warning}", file=sys.stderr)
    return source_info


def package_versions() -> dict[str, str | None]:
    """Record installed versions without importing CUDA/JAX packages."""
    names = (
        "jax", "jaxlib", "flax", "optax", "distrax", "ml-collections", "ogbench", "gymnasium",
        "numpy", "wandb", "dm-control", "mujoco", "tensorflow-probability", "orbax-checkpoint",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def build_command(
    preset: FQLPreset, *, seed: int, save_dir: Path, run_group: str, eval_interval: int, save_interval: int
) -> list[str]:
    """Build the full recipe explicitly; never rely on upstream task defaults."""
    return [
        sys.executable,
        "main.py",
        "--agent=agents/fql.py",
        f"--env_name={preset.dataset_id}",
        f"--seed={seed}",
        f"--save_dir={save_dir}",
        f"--run_group={run_group}",
        f"--offline_steps={preset.offline_steps}",
        f"--online_steps={preset.online_steps}",
        "--buffer_size=2000000",
        "--log_interval=5000",
        f"--eval_interval={eval_interval}",
        "--eval_episodes=50",
        f"--save_interval={save_interval}",
        "--video_episodes=0",
        "--balanced_sampling=0",
        "--agent.agent_name=fql",
        "--agent.lr=0.0003",
        "--agent.batch_size=256",
        "--agent.actor_hidden_dims=(512,512,512,512)",
        "--agent.value_hidden_dims=(512,512,512,512)",
        "--agent.layer_norm=True",
        "--agent.actor_layer_norm=False",
        f"--agent.discount={preset.discount:g}",
        "--agent.tau=0.005",
        f"--agent.q_agg={preset.q_agg}",
        f"--agent.alpha={preset.alpha:g}",
        "--agent.flow_steps=10",
        "--agent.normalize_q_loss=False",
    ]


def _gpu_preflight() -> dict[str, object]:
    """Require an actual CUDA JAX device; do not allow an accidental CPU run."""
    code = (
        "import json, jax; "
        "devices=[{'platform': d.platform, 'id': d.id} for d in jax.devices()]; "
        "print(json.dumps(devices)); "
        "raise SystemExit(0 if any(d['platform'] == 'gpu' for d in devices) else 2)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "FQL GPU preflight failed: JAX did not report a CUDA GPU. "
            f"stdout={completed.stdout.strip()!r}; stderr={completed.stderr.strip()!r}"
        )
    try:
        devices = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FQL GPU preflight produced invalid device JSON: {completed.stdout!r}") from exc
    return {"jax_devices": devices}


def _dependency_preflight() -> dict[str, str | None]:
    """Check pinned runtime metadata before source execution imports it."""
    versions = package_versions()
    if versions["ogbench"] != OGBENCH_VERSION or versions["gymnasium"] != GYMNASIUM_VERSION:
        raise RuntimeError(
            "FQL requires its isolated OGBench runtime with "
            f"ogbench=={OGBENCH_VERSION} and gymnasium=={GYMNASIUM_VERSION}; found "
            f"ogbench={versions['ogbench']!r}, gymnasium={versions['gymnasium']!r}."
        )
    missing = [
        name
        for name in ("jax", "jaxlib", "flax", "optax", "distrax", "ml-collections", "wandb")
        if versions[name] is None
    ]
    if missing:
        raise RuntimeError(
            f"FQL runtime is missing required packages: {', '.join(missing)}. "
            "See docs/guides/fql-ogbench-reproduction.md for isolated-environment setup."
        )
    return versions


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fql-source", type=Path, default=DEFAULT_FQL_SOURCE,
        help="Read-only official FQL checkout (default: repository 3rd_party/fql).",
    )
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-group", help="Upstream W&B/run directory group; defaults to the preset name.")
    parser.add_argument("--offline-steps", type=int, help="Override the preset offline-update budget (for isolated smoke runs).")
    parser.add_argument("--online-steps", type=int, help="Override the preset online-step budget (for isolated smoke runs).")
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command only; does not inspect source, runtime, or paths.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.eval_interval < 0 or args.save_interval < 1:
        raise ValueError("--eval-interval must be nonnegative and --save-interval must be positive.")
    if args.offline_steps is not None and args.offline_steps < 0:
        raise ValueError("--offline-steps must be nonnegative.")
    if args.online_steps is not None and args.online_steps < 0:
        raise ValueError("--online-steps must be nonnegative.")
    preset = PRESETS[args.preset]
    preset = replace(
        preset,
        offline_steps=args.offline_steps if args.offline_steps is not None else preset.offline_steps,
        online_steps=args.online_steps if args.online_steps is not None else preset.online_steps,
    )
    source = args.fql_source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    run_group = args.run_group or preset.name
    command = build_command(
        preset, seed=args.seed, save_dir=output_dir / "upstream", run_group=run_group,
        eval_interval=args.eval_interval, save_interval=args.save_interval,
    )
    if args.dry_run:
        print(json.dumps({"preset": asdict(preset), "cwd": str(source), "argv": command}, indent=2, sort_keys=True))
        return 0

    source_info = inspect_source(source)
    versions = _dependency_preflight()
    gpu = _gpu_preflight()
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "preset": asdict(preset), "seed": args.seed, "cwd": str(source), "argv": command,
        "source": source_info, "packages": versions, "gpu_preflight": gpu,
        "official_config_defaults": {"agent.encoder": None, "precision": "float32"},
        "wandb_note": "Upstream main.py calls setup_wandb(project='fql', ...). WANDB_MODE=disabled is not a supported launcher control; use authorized upstream W&B credentials.",
    }
    _write_json(output_dir / "command.json", metadata)
    completed = subprocess.run(command, cwd=source, env=os.environ.copy(), check=False)
    _write_json(output_dir / "result.json", {"returncode": completed.returncode})
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
