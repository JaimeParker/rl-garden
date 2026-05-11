"""Datasets for the color reward classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from rl_garden.models.reward.classifiers.base.io import decode_jpeg_crop
from rl_garden.models.reward.classifiers.base.transforms import build_transforms, empty_image


class ColorRewardDataset(Dataset):
    """Color classification dataset from NPZ labels + HDF5 episodes."""

    def __init__(
        self,
        npz_file: Path,
        data_dir: Path,
        image_size: Tuple[int, int] = (128, 128),
        normalize: bool = True,
    ) -> None:
        self.npz_file = Path(npz_file)
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self._cam_key = "cam_high_rgb"
        self._crop_region = (100, 200, 300, 400)

        with np.load(self.npz_file) as data:
            self.labels = data["color_labels"].astype(np.float32)
            self.frame_indices = data["frame_indices"].astype(np.int32)
            self.episode_ids = data["episode_ids"]

        self._episode_handles: dict[str, h5py.File] = {}
        self.transforms = build_transforms(image_size, normalize)

    def _get_episode_handle(self, episode_id: str) -> Optional[h5py.File]:
        handle = self._episode_handles.get(episode_id)
        if handle is not None:
            return handle

        episode_file = self.data_dir / f"{episode_id}.hdf5"
        if not episode_file.exists():
            return None

        handle = h5py.File(episode_file, "r")

        self._episode_handles[episode_id] = handle
        return handle

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        episode_id = self.episode_ids[idx]
        if isinstance(episode_id, bytes):
            episode_id = episode_id.decode()

        handle = self._get_episode_handle(str(episode_id))
        if handle is None:
            return empty_image(self.image_size), label

        compressed = handle[f"obs/{self._cam_key}"][self.frame_indices[idx]]
        img = decode_jpeg_crop(compressed, self._crop_region)
        if img is None:
            return empty_image(self.image_size), label
        return self.transforms(img), label

    def __del__(self) -> None:
        for handle in self._episode_handles.values():
            if handle is not None:
                handle.close()


class CombinedColorRewardDataset(Dataset):
    """Combine multiple color reward datasets from .npz files."""

    def __init__(
        self,
        npz_files: Sequence[Path],
        data_dirs: Sequence[Path],
        image_size: Tuple[int, int] = (128, 128),
        normalize: bool = True,
    ) -> None:
        if len(npz_files) != len(data_dirs):
            raise ValueError("npz_files must match data_dirs length")

        datasets = [
            ColorRewardDataset(npz, data_dir, image_size, normalize)
            for npz, data_dir in zip(npz_files, data_dirs)
        ]
        self._dataset = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dataset[idx]


__all__ = ["ColorRewardDataset", "CombinedColorRewardDataset"]
