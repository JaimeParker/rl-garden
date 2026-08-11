#!/usr/bin/env python3
"""Prepare and run official IQL with rl-garden's fixed-mixing patch.

This script does not modify ``3rd_party/implicit_q_learning``. It copies the
pinned source tree into a run work directory, applies
``tools/reproductions/iql_fixed_mixing.patch``, records source/patch metadata,
and executes ``train_finetune.py`` from the copied tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "3rd_party" / "implicit_q_learning"
DEFAULT_PATCH = REPO_ROOT / "tools" / "reproductions" / "iql_fixed_mixing.patch"
STATE_CAPTURE_SCRIPT = REPO_ROOT / "scripts" / "capture_experiment_state.py"


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


def _copy_source(source: Path, target: Path, *, overwrite: bool) -> None:
    if target.exists():
        if not overwrite:
            raise SystemExit(f"Target source copy already exists: {target}")
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--num-pretraining-steps", type=int, default=1_000_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100_000)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--replay-buffer-size", type=int, default=2_000_000)
    parser.add_argument("--fixed-mixing-ratio", type=float, default=0.5)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--overwrite", action="store_true")
    tqdm_group = parser.add_mutually_exclusive_group()
    tqdm_group.add_argument("--tqdm", dest="tqdm", action="store_true")
    tqdm_group.add_argument("--no-tqdm", dest="tqdm", action="store_false")
    parser.set_defaults(tqdm=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    patch = args.patch.resolve()
    work_dir = args.work_dir.resolve()
    source_copy = work_dir / "implicit_q_learning_fixed_mixing"
    save_dir = args.save_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            str(STATE_CAPTURE_SCRIPT),
            "--output-dir",
            str(save_dir / "repro_state"),
        ]
    )
    _copy_source(source, source_copy, overwrite=args.overwrite)
    _run(["git", "apply", str(patch)], cwd=source_copy)

    metadata = {
        "source": str(source),
        "source_head": _git_head(source),
        "source_copy": str(source_copy),
        "patch": str(patch),
        "patch_sha256": _sha256(patch),
        "env_name": args.env_name,
        "seed": args.seed,
        "fixed_mixing_ratio": args.fixed_mixing_ratio,
        "batch_size": args.batch_size,
        "offline_batch": int(args.batch_size * args.fixed_mixing_ratio),
        "online_batch": args.batch_size - int(args.batch_size * args.fixed_mixing_ratio),
        "max_steps": args.max_steps,
        "num_pretraining_steps": args.num_pretraining_steps,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "value_lr": args.value_lr,
    }
    (save_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    config_path = source_copy / "configs" / "antmaze_finetune_config.py"
    cmd = [
        args.python,
        "-u",
        "train_finetune.py",
        f"--env_name={args.env_name}",
        f"--save_dir={save_dir}",
        f"--seed={args.seed}",
        f"--eval_episodes={args.eval_episodes}",
        f"--eval_interval={args.eval_interval}",
        f"--log_interval={args.log_interval}",
        f"--batch_size={args.batch_size}",
        f"--max_steps={args.max_steps}",
        f"--num_pretraining_steps={args.num_pretraining_steps}",
        f"--replay_buffer_size={args.replay_buffer_size}",
        f"--fixed_mixing_ratio={args.fixed_mixing_ratio}",
        f"--config={config_path}",
        f"--config.actor_lr={args.actor_lr}",
        f"--config.critic_lr={args.critic_lr}",
        f"--config.value_lr={args.value_lr}",
    ]
    if not args.tqdm:
        cmd.append("--notqdm")

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _run(cmd, cwd=source_copy, env=env)


if __name__ == "__main__":
    main()
