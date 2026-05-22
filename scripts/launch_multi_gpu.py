#!/usr/bin/env python3
"""Launch independent RoboTwin training processes, one per GPU.

Each process gets its own CUDA context, SAPIEN scene, policy, and replay
buffer — no shared state.  GPU selection uses CUDA_VISIBLE_DEVICES +
CUDA_DEVICE_ORDER=PCI_BUS_ID so that both PyTorch and SAPIEN/Vulkan land
on the same physical GPU.

Usage (from repo root, inside the Docker container):

    python scripts/launch_multi_gpu.py \\
        --gpus 1 2 \\
        --algo sac \\
        -- \\
        --total-timesteps 1000000 \\
        --log-type tensorboard \\
        --robotwin-root /workspace/RoboTwin

Each GPU spawns its own tmux session named <session-prefix><gpu_id>.
Attach with:  tmux attach -t <session>
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys


def _tmux_session_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch one RoboTwin training process per GPU in separate tmux sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        required=True,
        metavar="GPU_ID",
        help="Physical GPU IDs to use (e.g. --gpus 0 1 2 3).",
    )
    parser.add_argument(
        "--algo",
        choices=["sac", "ppo"],
        default="sac",
        help="Training algorithm (default: sac).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1,
        help="Seed for the first GPU. Each subsequent GPU gets base_seed+rank (default: 1).",
    )
    parser.add_argument(
        "--session-prefix",
        default="robotwin_gpu",
        help="Prefix for tmux session names (default: robotwin_gpu).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Override log directory passed to each process (default: runs/).",
    )
    args, forward = parser.parse_known_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(
        repo_dir, "scripts", f"train_{args.algo}_robotwin_place_empty_cup_rgbd.sh"
    )
    if not os.path.isfile(script):
        print(f"Error: launcher script not found: {script}", file=sys.stderr)
        sys.exit(1)

    launched = []
    for rank, gpu_id in enumerate(args.gpus):
        seed = args.base_seed + rank
        session = f"{args.session_prefix}{gpu_id}"
        exp_name = f"{args.session_prefix}{gpu_id}_s{seed}"

        if _tmux_session_exists(session):
            print(f"[gpu {gpu_id}] tmux session '{session}' already exists — skipping.")
            continue

        extra = list(forward)
        extra += ["--seed", str(seed), "--exp-name", exp_name, "--device", "cuda:0"]
        if args.log_dir:
            extra += ["--log-dir", args.log_dir]

        inner_cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            f"CUDA_DEVICE_ORDER=PCI_BUS_ID "
            f"bash {shlex.quote(script)} "
            + " ".join(shlex.quote(a) for a in extra)
        )

        tmux_cmd = [
            "tmux", "new-session", "-d", "-s", session,
            inner_cmd,
        ]
        subprocess.run(tmux_cmd, check=True)
        print(f"[gpu {gpu_id}] launched → session '{session}'  seed={seed}")
        launched.append(session)

    if launched:
        print("\nAttach to sessions:")
        for s in launched:
            print(f"  tmux attach -t {s}")
    else:
        print("No new sessions were launched.")


if __name__ == "__main__":
    main()
