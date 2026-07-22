"""RoboTwin ACT dataset utilities.

The converter writes ManiSkill-style trajectory HDF5 files because the vendored
ACT loader already understands that structure.  The dataset class below keeps
RoboTwin camera ordering explicit so training matches ``ACTBaseActionProvider``.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal CLI envs
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass


ROBOTWIN_ACT_CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")
ROBOTWIN_EVAL_RGB_KEYS = ("rgb", "rgb_left_wrist", "rgb_right_wrist")


@dataclass(frozen=True)
class RobotWinEpisode:
    states: np.ndarray
    left_endpose: np.ndarray
    left_gripper: np.ndarray
    right_endpose: np.ndarray
    right_gripper: np.ndarray
    images: dict[str, np.ndarray]


def normalize_robotwin_task_name(task_name: str) -> str:
    """Convert user-facing task names such as ``open-laptop`` to RoboTwin ids."""

    return task_name.replace("-", "_")


def robotwin_json_path(h5_path: str | Path) -> Path:
    return Path(h5_path).with_suffix(".json")


def sorted_episode_paths(source_dir: str | Path) -> list[Path]:
    source = Path(source_dir).expanduser()
    data_dir = source / "data" if (source / "data").is_dir() else source
    paths = list(data_dir.glob("episode*.hdf5"))

    def _episode_index(path: Path) -> int:
        stem = path.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return int(digits) if digits else -1

    return sorted(paths, key=_episode_index)


def read_seed_file(source_dir: str | Path) -> list[int]:
    source = Path(source_dir).expanduser()
    candidates = [source / "seed.txt", source.parent / "seed.txt"]
    for path in candidates:
        if path.exists():
            return [int(item) for item in path.read_text(encoding="utf-8").split()]
    return []


def decode_image_sequence(data: np.ndarray) -> np.ndarray:
    """Return ``T,H,W,3`` uint8 images from raw arrays or RoboTwin JPEG bytes."""

    array = np.asarray(data)
    if array.ndim == 4 and array.dtype == np.uint8 and array.shape[-1] == 3:
        return array
    if array.ndim == 0:
        array = array.reshape(1)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Decoding RoboTwin JPEG image datasets requires pillow."
        ) from exc

    images = []
    for item in array.reshape(-1):
        if isinstance(item, np.void):
            payload = bytes(item)
        elif isinstance(item, (bytes, bytearray)):
            payload = bytes(item)
        elif isinstance(item, np.ndarray) and item.dtype == np.uint8:
            payload = item.tobytes()
        else:
            raise TypeError(f"Unsupported RoboTwin image item type: {type(item)!r}.")
        payload = payload.rstrip(b"\0")
        with Image.open(io.BytesIO(payload)) as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return np.stack(images, axis=0)


def resize_images(images: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize ``T,H,W,3`` uint8 images to ``(height, width)``."""

    height, width = image_size
    if images.shape[1:3] == (height, width):
        return np.asarray(images, dtype=np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Resizing RoboTwin RGB images requires pillow.") from exc
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize((width, height), resample=Image.BILINEAR),
                dtype=np.uint8,
            )
            for frame in images
        ],
        axis=0,
    )


def load_official_robotwin_episode(
    episode_path: str | Path,
    *,
    camera_names: Sequence[str] = ROBOTWIN_ACT_CAMERA_NAMES,
    image_size: Optional[tuple[int, int]] = None,
) -> RobotWinEpisode:
    """Load one official RoboTwin HDF5 episode with qpos, endpose, and RGB."""

    path = Path(episode_path).expanduser()
    with h5py.File(path, "r") as root:
        if "joint_action" not in root:
            raise KeyError(f"{path} does not contain /joint_action.")
        joint = root["joint_action"]
        if "vector" in joint:
            states = np.asarray(joint["vector"][()], dtype=np.float32)
        else:
            states = np.concatenate(
                [
                    np.asarray(joint["left_arm"][()], dtype=np.float32),
                    np.asarray(joint["left_gripper"][()], dtype=np.float32).reshape(-1, 1),
                    np.asarray(joint["right_arm"][()], dtype=np.float32),
                    np.asarray(joint["right_gripper"][()], dtype=np.float32).reshape(-1, 1),
                ],
                axis=1,
            )
        if "endpose" not in root:
            raise KeyError(f"{path} does not contain /endpose.")
        endpose = root["endpose"]
        images = {}
        for camera_name in camera_names:
            if f"observation/{camera_name}/rgb" not in root:
                raise KeyError(f"{path} does not contain /observation/{camera_name}/rgb.")
            decoded = decode_image_sequence(root[f"observation/{camera_name}/rgb"][()])
            images[camera_name] = resize_images(decoded, image_size) if image_size else decoded
        return RobotWinEpisode(
            states=states,
            left_endpose=np.asarray(endpose["left_endpose"][()], dtype=np.float32),
            left_gripper=np.asarray(endpose["left_gripper"][()], dtype=np.float32).reshape(-1),
            right_endpose=np.asarray(endpose["right_endpose"][()], dtype=np.float32),
            right_gripper=np.asarray(endpose["right_gripper"][()], dtype=np.float32).reshape(-1),
            images=images,
        )


def quaternion_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    cos_angle = (np.trace(matrix) - 1.0) * 0.5
    angle = math.acos(float(np.clip(cos_angle, -1.0, 1.0)))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * math.sin(angle)
    return (axis * angle).astype(np.float32)


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
    matrix[:3, :3] = quaternion_wxyz_to_matrix(np.asarray(pose[3:7], dtype=np.float64))
    return matrix


def compute_delta_ee_actions(
    left_endpose: np.ndarray,
    left_gripper: np.ndarray,
    right_endpose: np.ndarray,
    right_gripper: np.ndarray,
    *,
    ee_delta_pos_scale: float = 0.03,
    ee_delta_rot_scale: float = 0.15,
    gripper_delta_scale: float = 0.2,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute normalized 14D RoboTwin ``delta_ee`` actions from EE poses."""

    length = min(
        len(left_endpose), len(left_gripper), len(right_endpose), len(right_gripper)
    )
    if length < 2:
        raise ValueError("At least two endpose samples are required.")
    actions = np.zeros((length - 1, 14), dtype=np.float32)
    raw_abs_max = np.zeros(14, dtype=np.float32)
    for i in range(length - 1):
        left_delta = np.linalg.inv(pose_to_matrix(left_endpose[i])) @ pose_to_matrix(left_endpose[i + 1])
        right_delta = np.linalg.inv(pose_to_matrix(right_endpose[i])) @ pose_to_matrix(right_endpose[i + 1])
        actions[i, 0:3] = left_delta[:3, 3] / ee_delta_pos_scale
        actions[i, 3:6] = matrix_to_rotvec(left_delta[:3, :3]) / ee_delta_rot_scale
        actions[i, 6] = (left_gripper[i + 1] - left_gripper[i]) / gripper_delta_scale
        actions[i, 7:10] = right_delta[:3, 3] / ee_delta_pos_scale
        actions[i, 10:13] = matrix_to_rotvec(right_delta[:3, :3]) / ee_delta_rot_scale
        actions[i, 13] = (right_gripper[i + 1] - right_gripper[i]) / gripper_delta_scale
        raw_abs_max = np.maximum(raw_abs_max, np.abs(actions[i]))

    clipped = np.clip(actions, -1.0, 1.0).astype(np.float32)
    stats = {
        "action_abs_max_before_clip": float(raw_abs_max.max()),
        "action_clip_fraction": float(np.mean(np.abs(actions) > 1.0)),
    }
    return clipped, stats


def write_act_h5(
    output_path: str | Path,
    episodes: Iterable[RobotWinEpisode],
    *,
    task_name: str,
    task_config: Optional[str],
    camera_names: Sequence[str] = ROBOTWIN_ACT_CAMERA_NAMES,
    ee_delta_pos_scale: float = 0.03,
    ee_delta_rot_scale: float = 0.15,
    gripper_delta_scale: float = 0.2,
) -> dict[str, Any]:
    """Write converted RoboTwin episodes into rl-garden ACT HDF5 format."""

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "env_info": {
            "env_id": task_name,
            "env_kwargs": {
                "control_mode": "delta_ee",
                "obs_mode": "rgb",
                "camera_names": list(camera_names),
                "ee_delta_pos_scale": ee_delta_pos_scale,
                "ee_delta_rot_scale": ee_delta_rot_scale,
                "gripper_delta_scale": gripper_delta_scale,
            },
        },
        "task_config": task_config,
        "episodes": [],
    }
    with h5py.File(output, "w") as root:
        for traj_idx, episode in enumerate(episodes):
            actions, stats = compute_delta_ee_actions(
                episode.left_endpose,
                episode.left_gripper,
                episode.right_endpose,
                episode.right_gripper,
                ee_delta_pos_scale=ee_delta_pos_scale,
                ee_delta_rot_scale=ee_delta_rot_scale,
                gripper_delta_scale=gripper_delta_scale,
            )
            obs_len = actions.shape[0] + 1
            traj = root.create_group(f"traj_{traj_idx}")
            traj.create_dataset("actions", data=actions, dtype=np.float32)
            obs = traj.create_group("obs")
            sensor_data = obs.create_group("sensor_data")
            for camera_name in camera_names:
                sensor_data.create_group(camera_name).create_dataset(
                    "rgb",
                    data=np.asarray(episode.images[camera_name][:obs_len], dtype=np.uint8),
                    dtype=np.uint8,
                )
            obs.create_group("sensor_param")
            extra = obs.create_group("extra")
            extra.create_dataset(
                "state",
                data=np.asarray(episode.states[:obs_len], dtype=np.float32),
                dtype=np.float32,
            )
            metadata["episodes"].append(
                {
                    "episode_id": traj_idx,
                    "control_mode": "delta_ee",
                    "length": int(actions.shape[0]),
                    **stats,
                }
            )
    metadata["num_episodes"] = len(metadata["episodes"])
    metadata["num_transitions"] = int(sum(ep["length"] for ep in metadata["episodes"]))
    robotwin_json_path(output).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class RoboTwinACTDataset(Dataset):
    """In-memory ACT dataset for converted RoboTwin RGB demonstrations."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        num_queries: int,
        num_traj: Optional[int] = None,
        camera_names: Sequence[str] = ROBOTWIN_ACT_CAMERA_NAMES,
        image_size: tuple[int, int] = (224, 224),
        control_mode: str = "delta_ee",
    ) -> None:
        if torch is None:
            raise RuntimeError("RoboTwinACTDataset requires torch.")
        self.data_path = Path(data_path).expanduser()
        self.num_queries = int(num_queries)
        self.camera_names = tuple(camera_names)
        self.image_size = image_size
        self.control_mode = control_mode
        self.delta_control = control_mode == "delta_ee" or "delta" in control_mode
        self.trajectories = self._load(num_traj=num_traj)
        self.slices: list[tuple[int, int]] = []
        for traj_idx, traj in enumerate(self.trajectories):
            for ts in range(traj["actions"].shape[0]):
                self.slices.append((traj_idx, ts))
        if not self.slices:
            raise ValueError(f"No transitions found in {self.data_path}.")
        self.state_dim = int(self.trajectories[0]["state"].shape[1])
        self.action_dim = int(self.trajectories[0]["actions"].shape[1])
        self.num_traj = len(self.trajectories)
        self.norm_stats = None if self.delta_control else self._compute_norm_stats()

    def _load(self, *, num_traj: Optional[int]) -> list[dict[str, torch.Tensor]]:
        with h5py.File(self.data_path, "r") as root:
            keys = sorted(root.keys(), key=lambda key: int(key.split("_")[-1]))
            if num_traj is not None:
                keys = keys[: int(num_traj)]
            trajectories = []
            for key in keys:
                traj = root[key]
                obs = traj["obs"]
                state = torch.as_tensor(obs["extra"]["state"][()], dtype=torch.float32)
                actions = torch.as_tensor(traj["actions"][()], dtype=torch.float32)
                images = []
                for camera_name in self.camera_names:
                    rgb = np.asarray(obs["sensor_data"][camera_name]["rgb"][()], dtype=np.uint8)
                    rgb = resize_images(rgb, self.image_size)
                    images.append(torch.as_tensor(rgb, dtype=torch.uint8).permute(0, 3, 1, 2))
                rgb_tensor = torch.stack(images, dim=1)
                trajectories.append({"state": state, "actions": actions, "rgb": rgb_tensor})
        return trajectories

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        traj_idx, ts = self.slices[index]
        traj = self.trajectories[traj_idx]
        state = traj["state"][ts]
        action_seq = traj["actions"][ts : ts + self.num_queries]
        action_len = action_seq.shape[0]
        if action_len < self.num_queries:
            if self.delta_control:
                pad = torch.zeros(
                    (self.num_queries - action_len, traj["actions"].shape[1]),
                    dtype=action_seq.dtype,
                )
            else:
                pad = action_seq[-1:].repeat(self.num_queries - action_len, 1)
            action_seq = torch.cat([action_seq, pad], dim=0)
        if self.norm_stats is not None:
            state = (state - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
            action_seq = (
                action_seq - self.norm_stats["action_mean"]
            ) / self.norm_stats["action_std"]
        return {
            "observations": {"state": state, "rgb": traj["rgb"][ts]},
            "actions": action_seq,
        }

    def _compute_norm_stats(self) -> dict[str, torch.Tensor]:
        states = []
        actions = []
        for traj_idx, ts in self.slices:
            traj = self.trajectories[traj_idx]
            states.append(traj["state"][ts])
            action_seq = traj["actions"][ts : ts + self.num_queries]
            if action_seq.shape[0] < self.num_queries:
                action_seq = torch.cat(
                    [
                        action_seq,
                        action_seq[-1:].repeat(self.num_queries - action_seq.shape[0], 1),
                    ],
                    dim=0,
                )
            actions.append(action_seq)
        state_data = torch.stack(states)
        action_data = torch.cat(actions, dim=0)
        return {
            "action_mean": action_data.mean(dim=0, keepdim=True),
            "action_std": action_data.std(dim=0, keepdim=True).clamp_min(1e-2),
            "state_mean": state_data.mean(dim=0, keepdim=True),
            "state_std": state_data.std(dim=0, keepdim=True).clamp_min(1e-2),
            "example_state": states[0],
        }
