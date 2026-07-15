#!/usr/bin/env python3
"""Replay RoboTwin expert trajectories and render dense rewards on video.

The runner is task-agnostic: it recreates dataset scenes from ``seed.txt``,
replays planner paths from ``_traj_data``, checks the runtime reward contract,
and renders reward overlays for the dataset videos.

Example:
    python tools/diagnostics/robotwin_reward_replay.py \
        --task stack_bowls_three \
        --robotwin-root /home/RoboTwin \
        --dataset data/aloha-agilex_clean_50 \
        --overlay-dir runs/stack_bowls_three_reward_overlay
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
EPISODE_VIDEO_RE = re.compile(r"^episode_?(\d+)\.mp4$")
VIDEO_CODEC_PREFERENCES = ("libx264", "libopenh264", "mpeg4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Registered RoboTwin reward task.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path(os.getenv("RLG_ROBOTWIN_ROOT", "/home/RoboTwin")),
    )
    parser.add_argument("--assets-path", type=Path, default=None)
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="Defaults to <robotwin-root>/task_config/demo_clean.yml.",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode indices. Defaults to the first --max-episodes.",
    )
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset>/video.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Defaults to runs/<task>_reward_overlay.",
    )
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument(
        "--video-codec",
        default="auto",
        help="ffmpeg video encoder. Defaults to the first available supported codec.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Write outputs but do not fail on reward-contract violations.",
    )
    return parser.parse_args()


def episode_indices(
    episodes: str | None, max_episodes: int, num_seeds: int
) -> list[int]:
    if episodes:
        indices = [int(value.strip()) for value in episodes.split(",") if value.strip()]
    else:
        if max_episodes <= 0:
            raise ValueError("--max-episodes must be positive.")
        indices = list(range(min(max_episodes, num_seeds)))
    if not indices:
        raise ValueError("At least one episode must be selected.")
    invalid = [idx for idx in indices if idx < 0 or idx >= num_seeds]
    if invalid:
        raise ValueError(f"Episode indices outside [0, {num_seeds}): {invalid}")
    if len(indices) != len(set(indices)):
        raise ValueError("Episode indices must be unique.")
    return indices


def load_seeds(dataset: Path) -> list[int]:
    path = dataset / "seed.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset seed file is missing: {path}")
    seeds = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if not seeds:
        raise ValueError(f"Dataset contains no seeds: {path}")
    return seeds


def load_planner_paths(dataset: Path, episode: int) -> dict[str, Any]:
    path = dataset / "_traj_data" / f"episode{episode}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Planner trajectory is missing: {path}")
    # RoboTwin trajectory files are pickle. Only use locally generated datasets.
    with path.open("rb") as file:
        data = pickle.load(file)
    required = {"left_joint_path", "right_joint_path"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError(f"Unexpected planner trajectory schema: {path}")
    return data


def hdf5_frame_count(dataset: Path, episode: int) -> int:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Reward replay validation requires h5py.") from exc
    path = dataset / "data" / f"episode{episode}.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 episode is missing: {path}")
    with h5py.File(path, "r") as file:
        return int(file["joint_action/vector"].shape[0])


def load_task_args(path: Path, task_name: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reward replay requires PyYAML.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"RoboTwin task config is missing: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in task config: {path}")
    task_args = copy.deepcopy(loaded)
    task_args.update(
        {
            "task_name": task_name,
            "task_config": path.stem,
            "eval_mode": False,
            "render_freq": 0,
            "need_plan": False,
            "save_data": False,
            "collect_data": False,
            "eval_video_log": False,
            "use_seed": True,
        }
    )
    task_args.setdefault("save_freq", 15)
    task_args.setdefault("clear_cache_freq", 5)
    return task_args


def task_step_limit(robotwin_root: Path, task_name: str, task_args: dict[str, Any]) -> int:
    if task_args.get("step_lim") is not None:
        return int(task_args["step_lim"])
    path = robotwin_root / "task_config" / "_eval_step_limit.yml"
    try:
        import yaml

        limits = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (ImportError, OSError, ValueError):
        limits = {}
    return int(limits.get(task_name, 2000)) if isinstance(limits, dict) else 2000


def make_config(
    task_name: str,
    seed: int,
    robotwin_root: Path,
    assets_path: Path,
    task_args: dict[str, Any],
):
    from rl_garden.envs.robotwin.config import RoboTwinEnvConfig

    domain = task_args.get("domain_randomization", {})
    camera = task_args.get("camera", {})
    step_lim = task_step_limit(robotwin_root, task_name, task_args)
    task_args["step_lim"] = step_lim
    return RoboTwinEnvConfig(
        task_name=task_name,
        seed=seed,
        robotwin_root=str(robotwin_root),
        assets_path=str(assets_path),
        reward_mode="dense",
        control_mode="joint_pos",
        embodiment=list(task_args.get("embodiment", ["aloha-agilex"])),
        step_lim=step_lim,
        max_episode_steps=step_lim,
        random_background=bool(domain.get("random_background", False)),
        cluttered_table=bool(domain.get("cluttered_table", False)),
        clean_background_rate=float(domain.get("clean_background_rate", 1.0)),
        random_head_camera_dis=float(domain.get("random_head_camera_dis", 0.0)),
        random_table_height=float(domain.get("random_table_height", 0.0)),
        random_light=bool(domain.get("random_light", False)),
        crazy_random_light_rate=float(domain.get("crazy_random_light_rate", 0.0)),
        include_wrist_cameras=bool(camera.get("collect_wrist_camera", True)),
        head_camera_type=str(camera.get("head_camera_type", "D435")),
        wrist_camera_type=str(camera.get("wrist_camera_type", "D435")),
        clear_cache_freq=int(task_args.get("clear_cache_freq", 5)),
        task_config=task_args,
    )


def set_stationary_reward_trace(task: Any) -> None:
    robot = task.robot
    task.episode_left_eef_poses = [
        np.asarray(robot.get_left_ee_pose(), dtype=np.float32)
    ]
    task.episode_right_eef_poses = [
        np.asarray(robot.get_right_ee_pose(), dtype=np.float32)
    ]
    task.episode_left_joint_states = [
        np.asarray(robot.get_left_arm_jointState(), dtype=np.float32)
    ]
    task.episode_right_joint_states = [
        np.asarray(robot.get_right_arm_jointState(), dtype=np.float32)
    ]


def runtime_reward(task: Any) -> tuple[float, bool]:
    set_stationary_reward_trace(task)
    success = bool(task.check_success())
    if success:
        return 1.0, True
    task.reward.update()
    if task.reward.is_fail():
        return 0.0, False
    return float(task.reward.compute_reward()), False


def reward_contract_violations(
    rewards: np.ndarray,
    *,
    native_success: bool,
    replay_frames: int,
    hdf5_frames: int,
) -> list[str]:
    violations: list[str] = []
    if not native_success:
        violations.append("expert replay did not reach native success")
    if not np.isfinite(rewards).all():
        violations.append("reward contains NaN or infinity")
    finite_rewards = rewards[np.isfinite(rewards)]
    if (
        finite_rewards.min(initial=0.0) < -1e-8
        or finite_rewards.max(initial=0.0) > 1.0 + 1e-8
    ):
        violations.append("reward is outside [0, 1]")
    if replay_frames != hdf5_frames:
        violations.append(
            f"replay frame count {replay_frames} != HDF5 frame count {hdf5_frames}"
        )
    return violations


def replay_episode(
    *,
    task_name: str,
    dataset: Path,
    robotwin_root: Path,
    assets_path: Path,
    task_args: dict[str, Any],
    episode: int,
    seed: int,
) -> dict[str, Any]:
    from rl_garden.envs.robotwin.adapter import RoboTwinTaskAdapter

    os.environ["ASSETS_PATH"] = str(assets_path)
    episode_args = copy.deepcopy(task_args)
    cfg = make_config(
        task_name, seed, robotwin_root, assets_path, episode_args
    )
    adapter = RoboTwinTaskAdapter(0, cfg, cfg.task_config, env_seed=seed)
    samples: list[dict[str, Any]] = []
    frame_samples = 0
    try:
        adapter.reset(env_seed=seed)
        if adapter.env_seed != seed:
            raise RuntimeError(
                f"Episode {episode}: requested seed {seed}, but reset used "
                f"{adapter.env_seed}; dataset scene would not match."
            )
        task = adapter.task
        assert task is not None
        paths = load_planner_paths(dataset, episode)
        task.set_path_lst(
            {
                "need_plan": False,
                "left_joint_path": paths["left_joint_path"],
                "right_joint_path": paths["right_joint_path"],
            }
        )

        def record_sample(kind: str = "frame", *_args: Any, **_kwargs: Any) -> None:
            nonlocal frame_samples
            reward, success = runtime_reward(task)
            sample: dict[str, Any] = {
                "sample": len(samples),
                "kind": kind,
                "reward": reward,
                "success": success,
            }
            samples.append(sample)
            if kind == "frame":
                frame_samples += 1

        def capture_frame(*_args: Any, **_kwargs: Any) -> None:
            record_sample("frame")

        task._take_picture = capture_frame
        record_sample("initial")
        try:
            task.play_once()
        except IndexError as exc:
            raise RuntimeError(
                f"Episode {episode}: planner path exhausted; replay initialization "
                "does not match dataset generation "
                f"(left={task.left_cnt}/{len(paths['left_joint_path'])}, "
                f"right={task.right_cnt}/{len(paths['right_joint_path'])})."
            ) from exc
        consumed = (task.left_cnt, task.right_cnt)
        available = (
            len(paths["left_joint_path"]),
            len(paths["right_joint_path"]),
        )
        if consumed != available:
            raise RuntimeError(
                f"Episode {episode}: replay consumed planner paths "
                f"left={consumed[0]}/{available[0]}, "
                f"right={consumed[1]}/{available[1]}."
            )
        record_sample("final")

        rewards = np.asarray([sample["reward"] for sample in samples], dtype=float)
        expected_frames = hdf5_frame_count(dataset, episode)
        violations = reward_contract_violations(
            rewards,
            native_success=bool(task.check_success()),
            replay_frames=frame_samples,
            hdf5_frames=expected_frames,
        )
        return {
            "episode": episode,
            "seed": seed,
            "hdf5_frames": expected_frames,
            "replay_frames": frame_samples,
            "native_success": bool(task.check_success()),
            "reward_start": float(rewards[0]),
            "reward_final": float(rewards[-1]),
            "reward_max": float(rewards.max()),
            "reward_drop_count": int(np.sum(np.diff(rewards) < -1e-6)),
            "finite": bool(np.isfinite(rewards).all()),
            "violations": violations,
            "samples": samples,
        }
    finally:
        adapter.close(clear_cache=((episode + 1) % cfg.clear_cache_freq == 0))


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.asarray(
        [sample["reward"] for episode in episodes for sample in episode["samples"]],
        dtype=float,
    )
    violations = [
        f"episode {episode['episode']}: {message}"
        for episode in episodes
        for message in episode["violations"]
    ]
    return {
        "episodes": len(episodes),
        "successful_replays": sum(bool(episode["native_success"]) for episode in episodes),
        "samples": int(rewards.size),
        "reward_quantiles": {
            str(q): float(np.quantile(rewards, q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "episodes_with_reward_drops": sum(
            episode["reward_drop_count"] > 0 for episode in episodes
        ),
        "violations": violations,
        "passed": not violations,
    }


def record_index(frame: int, frames: int, records: int) -> int:
    if frames <= 1 or records <= 1:
        return 0
    return min(round(frame * (records - 1) / (frames - 1)), records - 1)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def overlay_frame(
    raw: bytes,
    *,
    width: int,
    height: int,
    episode: int,
    episode_count: int,
    seed: int,
    frame: int,
    frame_count: int,
    samples: list[dict[str, Any]],
    returns: list[float],
) -> bytes:
    idx = record_index(frame, frame_count, len(samples))
    sample = samples[idx]
    succeeded = any(bool(row["success"]) for row in samples[: idx + 1])
    texts = [
        f"Episode {episode + 1}/{episode_count}  Seed {seed}",
        f"Frame {idx + 1}/{len(samples)}  Reward {float(sample['reward']):.4f}",
        f"Return {returns[idx]:.3f}  Success {'YES' if succeeded else 'NO'}",
    ]
    colors = [
        (245, 247, 250, 255),
        (255, 218, 102, 255),
        (91, 227, 137, 255) if succeeded else (230, 234, 240, 255),
    ]
    image = Image.frombytes("RGB", (width, height), raw).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    margin = max(4, round(width * 0.015))
    padding_x = max(5, round(width * 0.017))
    padding_y = max(4, round(height * 0.017))
    gap = max(2, round(height * 0.008))
    size = max(9, round(height * 0.038))
    max_width = width - 2 * margin - 2 * padding_x
    while size > 8:
        selected = font(size)
        boxes = [draw.textbbox((0, 0), text, font=selected) for text in texts]
        if max(box[2] - box[0] for box in boxes) <= max_width:
            break
        size -= 1
    else:
        selected = font(8)
        boxes = [draw.textbbox((0, 0), text, font=selected) for text in texts]
    line_heights = [box[3] - box[1] for box in boxes]
    box_width = max(box[2] - box[0] for box in boxes) + 2 * padding_x
    box_height = sum(line_heights) + gap * (len(texts) - 1) + 2 * padding_y
    draw.rounded_rectangle(
        (margin, margin, margin + box_width, margin + box_height),
        radius=4,
        fill=(0, 0, 0, 155),
        outline=(255, 255, 255, 80),
        width=1,
    )
    y = margin + padding_y
    for text, color, box, line_height in zip(
        texts, colors, boxes, line_heights, strict=True
    ):
        draw.text(
            (margin + padding_x - box[0], y - box[1]),
            text,
            font=selected,
            fill=color,
        )
        y += line_height + gap
    return Image.alpha_composite(image, layer).convert("RGB").tobytes()


def probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,nb_read_frames",
        "-of",
        "json",
        os.fspath(path),
    ]
    streams = json.loads(subprocess.check_output(command, text=True))["streams"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    return streams[0]


def episode_videos(video_dir: Path) -> dict[int, Path]:
    videos: dict[int, Path] = {}
    for path in video_dir.glob("episode*.mp4"):
        match = EPISODE_VIDEO_RE.match(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in videos:
            raise ValueError(f"Duplicate videos for episode {index}: {video_dir}")
        videos[index] = path
    return videos


def available_video_encoders(ffmpeg: str) -> set[str]:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Could not query ffmpeg encoders (exit={result.returncode}): {detail}"
        )
    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*V\S*\s+(\S+)", line)
        if match is not None and match.group(1) != "=":
            encoders.add(match.group(1))
    return encoders


def resolve_video_codec(ffmpeg: str, requested: str) -> str:
    encoders = available_video_encoders(ffmpeg)
    if requested != "auto":
        if requested not in encoders:
            alternatives = [
                codec for codec in VIDEO_CODEC_PREFERENCES if codec in encoders
            ]
            hint = ", ".join(alternatives) if alternatives else "auto"
            raise ValueError(
                f"ffmpeg encoder {requested!r} is unavailable; pass --video-codec "
                f"with one of: {hint}"
            )
        return requested
    for codec in VIDEO_CODEC_PREFERENCES:
        if codec in encoders:
            return codec
    raise RuntimeError(
        "ffmpeg has none of the supported video encoders: "
        + ", ".join(VIDEO_CODEC_PREFERENCES)
    )


def _stderr(process: subprocess.Popen) -> str:
    if process.stderr is None:
        return ""
    return process.stderr.read().decode(errors="replace").strip()


def _process_error(label: str, process: subprocess.Popen) -> RuntimeError:
    returncode = process.wait()
    detail = _stderr(process)
    suffix = f": {detail}" if detail else ""
    return RuntimeError(f"{label} exited with code {returncode}{suffix}")


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def encode_overlay(
    source: Path,
    destination: Path,
    report: dict[str, Any],
    *,
    episode_order: int,
    episode_count: int,
    ffmpeg: str,
    ffprobe: str,
    codec: str,
) -> None:
    info = probe_video(source, ffprobe)
    width, height = int(info["width"]), int(info["height"])
    count_value = info.get("nb_frames")
    if not count_value or count_value == "N/A":
        count_value = info.get("nb_read_frames")
    if not count_value or count_value == "N/A":
        raise ValueError(f"Could not determine frame count: {source}")
    frames = int(count_value)
    fps = Fraction(info["r_frame_rate"])
    samples = [sample for sample in report["samples"] if sample["kind"] == "frame"]
    if not samples:
        raise ValueError(f"Episode {report['episode']} has no replay frame samples")
    returns = np.cumsum([float(sample["reward"]) for sample in samples]).tolist()
    decoder = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            os.fspath(source),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps.numerator}/{fps.denominator}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            codec,
            "-b:v",
            "2M",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            os.fspath(destination),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert decoder.stdout is not None and encoder.stdin is not None
    frame_bytes = width * height * 3
    try:
        for frame_idx in range(frames):
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                raise RuntimeError(
                    f"Short decoded frame: {source} frame={frame_idx}"
                )
            try:
                encoder.stdin.write(
                    overlay_frame(
                        raw,
                        width=width,
                        height=height,
                        episode=episode_order,
                        episode_count=episode_count,
                        seed=int(report["seed"]),
                        frame=frame_idx,
                        frame_count=frames,
                        samples=samples,
                        returns=returns,
                    )
                )
            except BrokenPipeError as exc:
                raise _process_error(
                    f"ffmpeg encoder for {destination} (codec={codec})", encoder
                ) from exc
        try:
            encoder.stdin.close()
        except BrokenPipeError as exc:
            raise _process_error(
                f"ffmpeg encoder for {destination} (codec={codec})", encoder
            ) from exc
        decoder.stdout.close()
        decode_error = _stderr(decoder)
        encode_error = _stderr(encoder)
        decode_returncode = decoder.wait()
        encode_returncode = encoder.wait()
        if decode_returncode != 0:
            raise RuntimeError(
                f"ffmpeg decoder for {source} exited with code "
                f"{decode_returncode}: {decode_error}"
            )
        if encode_returncode != 0:
            raise RuntimeError(
                f"ffmpeg encoder for {destination} (codec={codec}) exited with "
                f"code {encode_returncode}: {encode_error}"
            )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        for stream in (decoder.stdout, encoder.stdin):
            try:
                stream.close()
            except (BrokenPipeError, OSError):
                pass
        _stop_process(decoder)
        _stop_process(encoder)


def render_overlays(
    reports: list[dict[str, Any]],
    *,
    video_dir: Path,
    output_dir: Path,
    ffmpeg: str,
    ffprobe: str,
    codec: str,
) -> None:
    videos = episode_videos(video_dir)
    missing = [report["episode"] for report in reports if report["episode"] not in videos]
    if missing:
        raise FileNotFoundError(f"Missing videos for episodes: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for order, report in enumerate(reports):
        source = videos[report["episode"]]
        destination = output_dir / f"{source.stem}_reward_overlay.mp4"
        encode_overlay(
            source,
            destination,
            report,
            episode_order=order,
            episode_count=len(reports),
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            codec=codec,
        )
        print(f"episode={report['episode']} overlay={destination}", flush=True)


def executable(value: str | None, name: str) -> str:
    selected = value or shutil.which(name)
    if selected is None:
        raise FileNotFoundError(f"Could not find {name}; pass --{name} explicitly.")
    return selected


def main() -> int:
    options = parse_args()
    dataset = options.dataset.resolve()
    robotwin_root = options.robotwin_root.resolve()
    assets_path = (options.assets_path or options.robotwin_root).resolve()
    task_config_path = (
        options.task_config or robotwin_root / "task_config" / "demo_clean.yml"
    ).resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory is missing: {dataset}")
    if not (robotwin_root / "envs" / f"{options.task}.py").is_file():
        raise FileNotFoundError(f"Task {options.task!r} is missing under {robotwin_root}")

    from rl_garden.envs.robotwin.rewards import supported_reward_tasks

    supported = supported_reward_tasks()
    if options.task not in supported:
        raise ValueError(
            f"No dense reward registered for {options.task!r}; supported tasks: "
            f"{', '.join(supported)}"
        )
    ffmpeg = executable(options.ffmpeg, "ffmpeg")
    ffprobe = executable(options.ffprobe, "ffprobe")
    codec = resolve_video_codec(ffmpeg, options.video_codec)
    print(f"video_encoder={codec}", flush=True)
    task_args = load_task_args(task_config_path, options.task)
    seeds = load_seeds(dataset)
    selected = episode_indices(options.episodes, options.max_episodes, len(seeds))
    reports = []
    for order, episode in enumerate(selected, start=1):
        print(
            f"[{order}/{len(selected)}] replay task={options.task} "
            f"episode={episode} seed={seeds[episode]}",
            flush=True,
        )
        report = replay_episode(
            task_name=options.task,
            dataset=dataset,
            robotwin_root=robotwin_root,
            assets_path=assets_path,
            task_args=task_args,
            episode=episode,
            seed=seeds[episode],
        )
        reports.append(report)
        print(
            f"  success={report['native_success']} frames="
            f"{report['replay_frames']}/{report['hdf5_frames']} "
            f"start={report['reward_start']:.4f} "
            f"max={report['reward_max']:.4f} "
            f"final={report['reward_final']:.4f}",
            flush=True,
        )

    summary = summarize(reports)
    overlay_dir = (
        options.overlay_dir or Path("runs") / f"{options.task}_reward_overlay"
    ).resolve()
    render_overlays(
        reports,
        video_dir=(options.video_dir or dataset / "video").resolve(),
        output_dir=overlay_dir,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        codec=codec,
    )
    print(
        f"summary passed={summary['passed']} episodes={summary['episodes']} "
        f"successful={summary['successful_replays']} output={overlay_dir}",
        flush=True,
    )
    for violation in summary["violations"]:
        print(f"violation: {violation}", flush=True)
    if not summary["passed"] and not options.no_strict:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
