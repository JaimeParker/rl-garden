"""Convert official Cal-QL Adroit Binary demonstrations to trajectory H5."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

TASKS = ("pen-binary-v0", "door-binary-v0", "relocate-binary-v0")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_observations(values: Any) -> np.ndarray:
    items = list(values)
    if items and isinstance(items[0], dict):
        items = [item["state_observation"] for item in items]
    return np.asarray(items, dtype=np.float32)


def _convert_trajectory(trajectory: dict[str, Any]) -> dict[str, np.ndarray] | None:
    observations = _state_observations(trajectory["observations"])
    next_observations = _state_observations(trajectory["next_observations"])
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    rewards = np.asarray(trajectory["rewards"], dtype=np.float32).squeeze()
    length = min(len(observations), len(next_observations), len(actions), len(rewards))
    rewards = rewards[:length]
    successes = np.flatnonzero(rewards == 0.0)
    if successes.size == 0:
        return None

    length = int(successes[-1]) + 1
    rewards = rewards[:length]
    success = rewards == 0.0
    return {
        "obs": observations[:length],
        "next_obs": next_observations[:length],
        "actions": np.clip(actions[:length], -0.99999, 0.99999),
        "rewards": rewards,
        "terminated": success,
        "truncated": np.zeros(length, dtype=np.bool_),
        "success": success,
        "episode_end": success.copy(),
    }


def convert_adroit_binary_dataset(
    *,
    task: str,
    expert_path: str | Path,
    bc_path: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> int:
    if task not in TASKS:
        raise ValueError(f"Unsupported Adroit Binary task: {task!r}")
    expert_path = Path(expert_path)
    bc_path = Path(bc_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sources = (
        ("expert", expert_path, np.load(expert_path, allow_pickle=True)),
        ("bc", bc_path, np.load(bc_path, allow_pickle=True)),
    )
    converted: list[tuple[str, int, dict[str, np.ndarray]]] = []
    for source_kind, _, trajectories in sources:
        for source_index, trajectory in enumerate(trajectories):
            result = _convert_trajectory(trajectory)
            if result is not None:
                converted.append((source_kind, source_index, result))
    if not converted:
        raise ValueError("No successful trajectories found in the input datasets.")

    with h5py.File(output_path, "w") as handle:
        handle.attrs["schema"] = "rl_garden.trajectory_h5.v1"
        handle.attrs["task"] = task
        handle.attrs["action_clip"] = 0.99999
        handle.attrs["reward_transform"] = "raw"
        handle.attrs["truncation"] = "last_reward_eq_zero_inclusive"
        handle.attrs["expert_file"] = expert_path.name
        handle.attrs["expert_sha256"] = _sha256(expert_path)
        handle.attrs["bc_file"] = bc_path.name
        handle.attrs["bc_sha256"] = _sha256(bc_path)
        for index, (source_kind, source_index, trajectory) in enumerate(converted):
            group = handle.create_group(f"traj_{index}")
            group.attrs["source_kind"] = source_kind
            group.attrs["source_index"] = source_index
            for key, values in trajectory.items():
                group.create_dataset(key, data=values)
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--expert-path", required=True, type=Path)
    parser.add_argument("--bc-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = convert_adroit_binary_dataset(
        task=args.task,
        expert_path=args.expert_path,
        bc_path=args.bc_path,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )
    print(f"Wrote {count} trajectories to {args.output_path}")


if __name__ == "__main__":
    main()
