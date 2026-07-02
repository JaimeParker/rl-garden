"""Replay a ManiSkill H5 demonstration and report replay success.

Usage:
    python examples/replay_maniskill_demo.py \
        demos/pegonly/trajectory.state.pd_ee_twist.physx_cpu.h5 --capture-video
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mani_skill.envs  # noqa: F401  (registers built-in ManiSkill envs)
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

from rl_garden.envs import register_custom_envs


@dataclass
class Args:
    traj_path: Annotated[str, tyro.conf.Positional]
    """Path to the ManiSkill trajectory .h5 file."""

    capture_video: bool = False
    """Whether to save replay videos."""

    output_dir: Optional[str] = None
    """Directory for replay videos and summary.json. Defaults to <traj-dir>/replay."""

    video_fps: int = 30
    """FPS for saved replay videos."""

    count: Optional[int] = None
    """Number of episodes to replay. Defaults to all episodes in the JSON metadata."""

    use_env_states: bool = True
    """Set recorded env states during replay to keep trajectories visually aligned."""


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime env.
        raise ImportError(
            "Replay requires h5py. Use the ManiSkill/RoboTwin environment or install h5py."
        ) from exc
    return h5py


def _json_path_for(traj_path: Path) -> Path:
    return traj_path.with_suffix(".json")


def _output_dir(args: Args, traj_path: Path) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser()
    return traj_path.parent / "replay"


def _normalize_reset_kwargs(episode: dict[str, Any]) -> dict[str, Any]:
    reset_kwargs = dict(episode.get("reset_kwargs") or {})
    seed = reset_kwargs.get("seed", episode.get("episode_seed"))
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError(
                f"Episode {episode.get('episode_id')} has ambiguous reset seed: {seed}."
            )
        seed = seed[0]
    reset_kwargs["seed"] = seed
    return reset_kwargs


def _set_env_state(env: Any, state: dict[str, Any]) -> None:
    target = getattr(env, "base_env", env)
    if hasattr(target, "set_state_dict"):
        target.set_state_dict(state)
        return
    target.unwrapped.set_state_dict(state)


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            return bool(value.reshape(-1)[0].item())
        array = np.asarray(value)
        if array.shape:
            return bool(array.reshape(-1)[0].item())
        return bool(array.item())
    except Exception:
        return None


def _extract_success(info: dict[str, Any]) -> Optional[bool]:
    keys = ("success_at_end", "success_once", "success")
    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        episode = final_info.get("episode")
        if isinstance(episode, dict):
            for key in keys:
                found = _as_bool(episode.get(key))
                if found is not None:
                    return found
        for key in keys:
            found = _as_bool(final_info.get(key))
            if found is not None:
                return found
    for key in keys:
        found = _as_bool(info.get(key))
        if found is not None:
            return found
    return None


def _traj_dataset_success(traj: Any) -> Optional[bool]:
    if "success" not in traj:
        return None
    success = np.asarray(traj["success"])
    if success.size == 0:
        return None
    return bool(success.reshape(-1)[-1])


def _build_env(metadata: dict[str, Any], output_dir: Path, args: Args):
    register_custom_envs()
    env_info = metadata["env_info"]
    env_id = env_info["env_id"]
    env_kwargs = dict(env_info.get("env_kwargs") or {})
    env_kwargs["num_envs"] = 1
    env_kwargs.setdefault("render_mode", "rgb_array")

    env = gym.make(env_id, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if args.capture_video:
        env = RecordEpisode(
            env,
            output_dir=str(output_dir),
            save_trajectory=False,
            save_video=True,
            trajectory_name="replay",
            max_steps_per_video=env_info.get("max_episode_steps"),
            video_fps=args.video_fps,
        )
    return env


def replay(args: Args) -> dict[str, Any]:
    h5py = _require_h5py()
    traj_path = Path(args.traj_path).expanduser()
    json_path = _json_path_for(traj_path)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Trajectory metadata JSON not found: {json_path}")

    metadata = json.loads(json_path.read_text())
    episodes = list(metadata.get("episodes") or [])
    if args.count is not None:
        episodes = episodes[: args.count]
    if not episodes:
        raise ValueError(f"No episodes to replay from metadata: {json_path}")

    output_dir = _output_dir(args, traj_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env(metadata, output_dir, args)
    episode_results: list[dict[str, Any]] = []
    try:
        with h5py.File(traj_path, "r") as h5:
            for episode in tqdm(episodes, desc="Replaying", unit="episode"):
                episode_id = int(episode["episode_id"])
                traj_key = f"traj_{episode_id}"
                if traj_key not in h5:
                    raise KeyError(f"{traj_key} does not exist in {traj_path}")
                traj = h5[traj_key]
                actions = np.asarray(traj["actions"], dtype=np.float32)
                env_states = (
                    trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
                    if args.use_env_states and "env_states" in traj
                    else None
                )

                reset_kwargs = _normalize_reset_kwargs(episode)
                env.reset(**reset_kwargs)
                if env_states:
                    _set_env_state(env, env_states[0])

                last_success: Optional[bool] = None
                for step, action in enumerate(actions):
                    _, _, _, _, info = env.step(action)
                    found = _extract_success(info)
                    if found is not None:
                        last_success = found
                    if env_states and step + 1 < len(env_states):
                        _set_env_state(env, env_states[step + 1])

                dataset_success = episode.get("success")
                if dataset_success is None:
                    dataset_success = _traj_dataset_success(traj)
                replay_success = (
                    bool(last_success)
                    if last_success is not None
                    else bool(dataset_success)
                    if dataset_success is not None
                    else False
                )
                episode_results.append(
                    {
                        "episode_id": episode_id,
                        "elapsed_steps": int(actions.shape[0]),
                        "success": replay_success,
                        "dataset_success": (
                            None if dataset_success is None else bool(dataset_success)
                        ),
                    }
                )
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in episode_results)
    dataset_successes = sum(
        int(item["dataset_success"])
        for item in episode_results
        if item["dataset_success"] is not None
    )
    dataset_count = sum(
        int(item["dataset_success"] is not None) for item in episode_results
    )
    summary = {
        "traj_path": str(traj_path),
        "json_path": str(json_path),
        "output_dir": str(output_dir),
        "episodes": len(episode_results),
        "successes": successes,
        "success_rate": successes / len(episode_results),
        "dataset_successes": dataset_successes,
        "dataset_success_rate": (
            dataset_successes / dataset_count if dataset_count else None
        ),
        "failed_episode_ids": [
            item["episode_id"] for item in episode_results if not item["success"]
        ],
        "episode_results": episode_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== ManiSkill Demo Replay ===", flush=True)
    print(f"episodes: {summary['episodes']}", flush=True)
    print(f"successes: {summary['successes']}", flush=True)
    print(f"success_rate: {summary['success_rate']:.4f}", flush=True)
    if summary["dataset_success_rate"] is not None:
        print(f"dataset_success_rate: {summary['dataset_success_rate']:.4f}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    return summary


def main() -> None:
    replay(tyro.cli(Args))


if __name__ == "__main__":
    main()
