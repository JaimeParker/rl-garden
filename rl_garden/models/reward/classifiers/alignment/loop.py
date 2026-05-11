"""Training loop for the alignment reward classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from torch.utils.data import Dataset

from rl_garden.models.reward.classifiers.alignment.dataset import (
    AlignmentRewardDataset,
    CombinedAlignmentDataset,
)
from rl_garden.models.reward.classifiers.shared.loop import BaseBinaryClassifierTrainer, BaseTrainConfig
from rl_garden.models.reward.classifiers.shared.model import ResNetBinaryClassifier


@dataclass(frozen=True)
class AlignmentClassifierConfig(BaseTrainConfig):
    label_files: Sequence[Path] = ()


class AlignmentClassifierTrainer(BaseBinaryClassifierTrainer):
    """Training wrapper for the alignment reward classifier."""

    def _validate_paths(self, paths: Iterable[Path]) -> list[Path]:
        out = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(path)
            out.append(path)
        return out

    def build_dataset(self) -> Dataset:
        label_files = self._validate_paths(self.config.label_files)
        if not label_files:
            raise ValueError("label_files is required for alignment classifiers")
        if len(label_files) == 1:
            return AlignmentRewardDataset(
                label_file=label_files[0],
                image_size=(self.config.image_size, self.config.image_size),
                normalize=self.config.normalize,
            )
        return CombinedAlignmentDataset(
            label_files=label_files,
            image_size=(self.config.image_size, self.config.image_size),
            normalize=self.config.normalize,
        )

    def build_model(self):
        return ResNetBinaryClassifier(pretrained=self.config.resnet_pretrained, dropout=0.1).to(
            self.device
        )

    def train(self) -> None:
        super().train(config_payload={"classifier_type": "alignment", "resnet": "resnet18"})
