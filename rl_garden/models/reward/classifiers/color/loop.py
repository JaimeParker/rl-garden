"""Training loop for the color reward classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.data import Dataset

from rl_garden.models.reward.classifiers.color.dataset import (
    ColorRewardDataset,
    CombinedColorRewardDataset,
)
from rl_garden.models.reward.classifiers.base.loop import BaseBinaryClassifierTrainer, BaseTrainConfig
from rl_garden.models.reward.classifiers.base.model import ResNetBinaryClassifier


@dataclass(frozen=True)
class ColorClassifierConfig(BaseTrainConfig):
    npz_files: Sequence[Path] = ()
    data_dirs: Sequence[Path] = ()


class ColorClassifierTrainer(BaseBinaryClassifierTrainer):
    """Training wrapper for the color reward classifier."""

    def _validate_paths(self, paths: Iterable[Path]) -> list[Path]:
        out = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(path)
            out.append(path)
        return out

    def build_dataset(self) -> Dataset:
        npz_files = self._validate_paths(self.config.npz_files)
        data_dirs = self._validate_paths(self.config.data_dirs)
        if not npz_files:
            raise ValueError("npz_files is required for color classifiers")
        if not data_dirs:
            raise ValueError("data_dirs is required for color classifiers")
        if len(data_dirs) == 1 and len(npz_files) > 1:
            data_dirs = data_dirs * len(npz_files)
        if len(npz_files) != len(data_dirs):
            raise ValueError("npz_files must match data_dirs length")

        if len(npz_files) == 1:
            return ColorRewardDataset(
                npz_file=npz_files[0],
                data_dir=data_dirs[0],
                image_size=(self.config.image_size, self.config.image_size),
                normalize=self.config.normalize,
            )
        return CombinedColorRewardDataset(
            npz_files=npz_files,
            data_dirs=data_dirs,
            image_size=(self.config.image_size, self.config.image_size),
            normalize=self.config.normalize,
        )

    def build_model(self):
        return ResNetBinaryClassifier(pretrained=self.config.resnet_pretrained, dropout=0.1).to(
            self.device
        )

    def train(self) -> None:
        super().train(config_payload={"classifier_type": "color", "resnet": "resnet18"})
