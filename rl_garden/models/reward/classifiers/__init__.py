"""Classifier-specific helpers for reward models."""

from rl_garden.models.reward.classifiers.alignment import (
    AlignmentClassifierConfig,
    AlignmentClassifierTrainer,
    AlignmentRewardDataset,
    CombinedAlignmentDataset,
)
from rl_garden.models.reward.classifiers.color import (
    ColorClassifierConfig,
    ColorClassifierTrainer,
    ColorRewardDataset,
    CombinedColorRewardDataset,
)
from rl_garden.models.reward.classifiers.shared.model import ResNetBinaryClassifier

__all__ = [
    "AlignmentClassifierConfig",
    "AlignmentClassifierTrainer",
    "AlignmentRewardDataset",
    "CombinedAlignmentDataset",
    "ColorClassifierConfig",
    "ColorClassifierTrainer",
    "ColorRewardDataset",
    "CombinedColorRewardDataset",
    "ResNetBinaryClassifier",
]
