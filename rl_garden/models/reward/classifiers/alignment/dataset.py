"""Datasets for the alignment reward classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from rl_garden.models.reward.classifiers.base.io import decode_jpeg
from rl_garden.models.reward.classifiers.base.transforms import build_transforms, empty_image


class AlignmentRewardDataset(Dataset):
    """Alignment classification dataset from HDF5 labels."""

    def __init__(
        self,
        label_file: Path,
        image_size: Tuple[int, int] = (128, 128),
        normalize: bool = True,
    ) -> None:
        self.label_file = Path(label_file)
        self._handle = h5py.File(self.label_file, "r")
        self.image_size = image_size
        self.transforms = build_transforms(image_size, normalize)

        labels = self._handle["labels"][()]
        self.labels = (labels - 1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        compressed = self._handle["images/cam_right_wrist_rgb"][idx]
        img = decode_jpeg(compressed)
        if img is None:
            return empty_image(self.image_size), label
        return self.transforms(img), label

    def __del__(self) -> None:
        if hasattr(self, "_handle") and self._handle is not None:
            self._handle.close()


class CombinedAlignmentDataset(Dataset):
    """Combine multiple alignment datasets."""

    def __init__(
        self,
        label_files: Sequence[Path],
        image_size: Tuple[int, int] = (128, 128),
        normalize: bool = True,
    ) -> None:
        datasets = [AlignmentRewardDataset(path, image_size, normalize) for path in label_files]
        self._dataset = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dataset[idx]


__all__ = ["AlignmentRewardDataset", "CombinedAlignmentDataset"]
