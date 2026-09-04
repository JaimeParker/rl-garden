"""Stress RoboTwin reset and tri-camera capture in one Python process."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from rl_garden.envs.robotwin.adapter import RoboTwinTaskAdapter
from rl_garden.envs.robotwin.config import RoboTwinEnvConfig


def _memory_snapshot() -> dict[str, object]:
    status: dict[str, str] = {}
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("VmRSS:", "VmSize:", "VmPeak:", "Threads:")):
                key, value = line.split(":", 1)
                status[key] = value.strip()

    gpu_memory = "unavailable"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_memory = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        **status,
        "open_fds": len(os.listdir("/proc/self/fd")),
        "gpu_memory_mib": gpu_memory,
    }


def _load_seeds(path: Path) -> list[int]:
    seeds = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if not seeds:
        raise ValueError(f"No seeds found in {path}")
    return seeds


def _validate_observation(obs: dict[str, object]) -> None:
    for key in ("rgb", "rgb_left_wrist", "rgb_right_wrist"):
        image = np.asarray(obs[key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise AssertionError(f"Unexpected {key} shape: {image.shape}")
        if not np.isfinite(image).all():
            raise AssertionError(f"Non-finite values in {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--clear-cache-freq", type=int, default=8)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--log-interval", type=int, default=8)
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.log_path.exists() or args.report_path.exists():
        raise FileExistsError("Stress-test output already exists")

    seeds = _load_seeds(args.seed_file)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial_memory = _memory_snapshot()
    completed = 0
    failure: dict[str, object] | None = None

    cfg = RoboTwinEnvConfig(
        task_name="open_laptop",
        device="cpu",
        robotwin_root=str(args.robotwin_root),
        control_mode="delta_ee",
        action_dim=14,
        reward_mode="sparse",
        include_wrist_cameras=True,
        image_size=(240, 320),
        max_episode_steps=700,
        step_lim=700,
        random_background=False,
        cluttered_table=False,
        clean_background_rate=1.0,
        random_head_camera_dis=0.0,
        random_table_height=0.0,
        random_light=False,
        crazy_random_light_rate=0.0,
        clear_cache_freq=args.clear_cache_freq,
    )
    adapter = RoboTwinTaskAdapter(0, cfg, cfg.task_config, env_seed=seeds[0])

    try:
        with args.log_path.open("w", encoding="utf-8", buffering=1) as log:
            for index in range(args.iterations):
                seed = seeds[index % len(seeds)]
                iteration_started = time.monotonic()
                obs = adapter.reset(env_seed=seed)
                _validate_observation(obs)
                completed = index + 1

                if completed == 1 or completed % args.log_interval == 0:
                    event = {
                        "event": "reset_progress",
                        "completed": completed,
                        "target": args.iterations,
                        "seed": seed,
                        "adapter_reset_count": adapter.reset_count,
                        "elapsed_seconds": time.monotonic() - started,
                        "iteration_seconds": time.monotonic() - iteration_started,
                        "memory": _memory_snapshot(),
                    }
                    log.write(json.dumps(event, sort_keys=True) + "\n")
                    print(
                        "RESET_STRESS_PROGRESS "
                        f"completed={completed}/{args.iterations} "
                        f"seed={seed} reset_count={adapter.reset_count}",
                        flush=True,
                    )
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "completed": completed,
        }
        raise
    finally:
        adapter.close(clear_cache=True)
        gc.collect()
        report = {
            "schema_version": 1,
            "completed": completed,
            "target": args.iterations,
            "passed": completed == args.iterations and failure is None,
            "clear_cache_freq": args.clear_cache_freq,
            "seed_file": str(args.seed_file),
            "robotwin_root": str(args.robotwin_root),
            "elapsed_seconds": time.monotonic() - started,
            "initial_memory": initial_memory,
            "final_memory": _memory_snapshot(),
            "failure": failure,
        }
        args.report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print("RESET_STRESS_PASSED", flush=True)


if __name__ == "__main__":
    main()
