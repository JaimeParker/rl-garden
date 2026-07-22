"""Convert official RoboTwin demonstrations to rl-garden ACT ``delta_ee`` data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from rl_garden.models.act.robotwin_dataset import (
    ROBOTWIN_ACT_CAMERA_NAMES,
    RobotWinEpisode,
    load_official_robotwin_episode,
    normalize_robotwin_task_name,
    read_seed_file,
    resize_images,
    sorted_episode_paths,
    write_act_h5,
)


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--task-name", default="open_laptop")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--robotwin-root", default="3rd_party/RoboTwin")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument(
        "--conversion-mode",
        choices=["auto", "direct", "replay"],
        default="auto",
        help="direct reads /endpose; replay reconstructs EE poses by replaying qpos.",
    )
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=list(ROBOTWIN_ACT_CAMERA_NAMES),
        help="RoboTwin camera group names to export, in ACT camera order.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planner-backend", default="mplib")
    parser.add_argument("--embodiment", nargs="+", default=["aloha-agilex"])
    parser.add_argument("--step-lim", type=int, default=700)
    parser.add_argument("--include-wrist-cameras", type=_str_to_bool, default=True)
    parser.add_argument("--random-background", type=_str_to_bool, default=False)
    parser.add_argument("--cluttered-table", type=_str_to_bool, default=False)
    parser.add_argument("--random-light", type=_str_to_bool, default=False)
    parser.add_argument("--ee-delta-pos-scale", type=float, default=0.03)
    parser.add_argument("--ee-delta-rot-scale", type=float, default=0.15)
    parser.add_argument("--gripper-delta-scale", type=float, default=0.2)
    return parser.parse_args()


def _has_endpose(path: Path) -> bool:
    with h5py.File(path, "r") as root:
        return "endpose" in root


def _load_task_config(robotwin_root: str | os.PathLike[str], task_config: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Replay conversion requires pyyaml.") from exc

    path = Path(robotwin_root).expanduser() / "task_config" / f"{task_config}.yml"
    if not path.exists():
        raise FileNotFoundError(f"RoboTwin task config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.load(f.read(), Loader=yaml.FullLoader)
    if not isinstance(loaded, dict):
        raise TypeError(f"RoboTwin task config must be a mapping: {path}")
    return dict(loaded)


def _joint_targets_from_episode(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as root:
        joint = root["joint_action"]
        if "vector" in joint:
            return np.asarray(joint["vector"][()], dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(joint["left_arm"][()], dtype=np.float32),
                np.asarray(joint["left_gripper"][()], dtype=np.float32).reshape(-1, 1),
                np.asarray(joint["right_arm"][()], dtype=np.float32),
                np.asarray(joint["right_gripper"][()], dtype=np.float32).reshape(-1, 1),
            ],
            axis=1,
        )


def _sample_from_task_obs(raw_obs: dict[str, Any], camera_names: tuple[str, ...]) -> dict[str, Any]:
    observation = raw_obs.get("observation", raw_obs)
    endpose = raw_obs.get("endpose")
    if endpose is None:
        raise KeyError("Replay task observation does not contain /endpose data.")
    joint = raw_obs.get("joint_action", {})
    state = joint.get("vector")
    if state is None:
        raise KeyError("Replay task observation does not contain /joint_action/vector.")
    images = {}
    for camera_name in camera_names:
        camera = observation.get(camera_name, {})
        if "rgb" not in camera:
            raise KeyError(f"Replay task observation missing {camera_name}/rgb.")
        images[camera_name] = np.asarray(camera["rgb"], dtype=np.uint8)
    return {
        "state": np.asarray(state, dtype=np.float32),
        "left_endpose": np.asarray(endpose["left_endpose"], dtype=np.float32),
        "left_gripper": float(endpose["left_gripper"]),
        "right_endpose": np.asarray(endpose["right_endpose"], dtype=np.float32),
        "right_gripper": float(endpose["right_gripper"]),
        "images": images,
    }


def _episode_from_samples(
    samples: list[dict[str, Any]],
    *,
    camera_names: tuple[str, ...],
    image_size: tuple[int, int],
) -> RobotWinEpisode:
    images = {}
    for camera_name in camera_names:
        stacked = np.stack([sample["images"][camera_name] for sample in samples], axis=0)
        images[camera_name] = resize_images(stacked, image_size)
    return RobotWinEpisode(
        states=np.stack([sample["state"] for sample in samples], axis=0),
        left_endpose=np.stack([sample["left_endpose"] for sample in samples], axis=0),
        left_gripper=np.asarray([sample["left_gripper"] for sample in samples], dtype=np.float32),
        right_endpose=np.stack([sample["right_endpose"] for sample in samples], axis=0),
        right_gripper=np.asarray([sample["right_gripper"] for sample in samples], dtype=np.float32),
        images=images,
    )


def _replay_episode(
    episode_path: Path,
    *,
    task_name: str,
    task_config: str,
    robotwin_root: str,
    seed: int,
    camera_names: tuple[str, ...],
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> RobotWinEpisode:
    from rl_garden.envs.robotwin.adapter import RoboTwinTaskAdapter, robotwin_workdir
    from rl_garden.envs.robotwin.config import RoboTwinEnvConfig

    task_args = _load_task_config(robotwin_root, task_config)
    task_args.update(
        {
            "task_name": task_name,
            "step_lim": args.step_lim,
            "planner_backend": args.planner_backend,
            "embodiment": args.embodiment,
            "render_freq": 0,
            "eval_mode": True,
            "collect_data": False,
            "save_data": False,
            "eval_video_log": False,
            "camera": {
                **task_args.get("camera", {}),
                "collect_head_camera": True,
                "collect_wrist_camera": args.include_wrist_cameras,
            },
            "domain_randomization": {
                **task_args.get("domain_randomization", {}),
                "random_background": args.random_background,
                "cluttered_table": args.cluttered_table,
                "random_light": args.random_light,
            },
            "data_type": {
                **task_args.get("data_type", {}),
                "rgb": True,
                "endpose": True,
                "qpos": True,
            },
        }
    )
    cfg = RoboTwinEnvConfig(
        task_name=task_name,
        seed=seed,
        robotwin_root=robotwin_root,
        step_lim=args.step_lim,
        max_episode_steps=args.step_lim,
        task_config=task_args,
        planner_backend=args.planner_backend,
        embodiment=args.embodiment,
        reward_mode="sparse",
        control_mode="joint_pos",
        image_size=image_size,
        include_wrist_cameras=args.include_wrist_cameras,
    )
    adapter = RoboTwinTaskAdapter(0, cfg, task_args, env_seed=seed)
    joint_targets = _joint_targets_from_episode(episode_path)
    samples: list[dict[str, Any]] = []
    try:
        adapter.reset(env_seed=seed)
        assert adapter.task is not None
        samples.append(_sample_from_task_obs(adapter.task.get_obs(), camera_names))
        for target in joint_targets[1:]:
            adapter._install_task_compatibility()
            with robotwin_workdir(robotwin_root):
                adapter.task.take_action(np.asarray(target, dtype=np.float32), action_type="qpos")
            samples.append(_sample_from_task_obs(adapter.task.get_obs(), camera_names))
    finally:
        adapter.close()
    return _episode_from_samples(samples, camera_names=camera_names, image_size=image_size)


def convert(args: argparse.Namespace) -> dict[str, Any]:
    task_name = normalize_robotwin_task_name(args.task_name)
    camera_names = tuple(args.camera_names)
    image_size = (args.camera_height, args.camera_width)
    episode_paths = sorted_episode_paths(args.source_dir)
    if args.num_episodes is not None:
        episode_paths = episode_paths[: args.num_episodes]
    if not episode_paths:
        raise FileNotFoundError(f"No episode*.hdf5 files found under {args.source_dir}.")
    seeds = read_seed_file(args.source_dir)
    episodes = []
    modes = []
    for episode_idx, episode_path in enumerate(episode_paths):
        use_direct = args.conversion_mode in {"auto", "direct"} and _has_endpose(episode_path)
        if args.conversion_mode == "direct" and not use_direct:
            raise KeyError(f"{episode_path} does not contain /endpose.")
        if use_direct:
            episode = load_official_robotwin_episode(
                episode_path,
                camera_names=camera_names,
                image_size=image_size,
            )
            mode = "direct"
        else:
            if args.conversion_mode == "auto":
                print(f"{episode_path}: /endpose missing; replaying joint_action.")
            seed = seeds[episode_idx] if episode_idx < len(seeds) else args.seed + episode_idx
            episode = _replay_episode(
                episode_path,
                task_name=task_name,
                task_config=args.task_config,
                robotwin_root=args.robotwin_root,
                seed=seed,
                camera_names=camera_names,
                image_size=image_size,
                args=args,
            )
            mode = "replay"
        episodes.append(episode)
        modes.append(mode)
        print(f"converted {episode_path.name} with {mode} mode")

    metadata = write_act_h5(
        args.output_path,
        episodes,
        task_name=task_name,
        task_config=args.task_config,
        camera_names=camera_names,
        ee_delta_pos_scale=args.ee_delta_pos_scale,
        ee_delta_rot_scale=args.ee_delta_rot_scale,
        gripper_delta_scale=args.gripper_delta_scale,
    )
    metadata["conversion_modes"] = modes
    json_path = Path(args.output_path).with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = convert(args)
    print(
        "wrote "
        f"{args.output_path} episodes={metadata['num_episodes']} "
        f"transitions={metadata['num_transitions']} json={Path(args.output_path).with_suffix('.json')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
