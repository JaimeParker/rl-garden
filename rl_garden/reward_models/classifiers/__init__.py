"""Classifier-specific helpers for reward models."""

from rl_garden.reward_models.classifiers.alignment import (
    AlignmentClassifierConfig,
    AlignmentClassifierTrainer,
    AlignmentRewardDataset,
    CombinedAlignmentDataset,
)
from rl_garden.reward_models.classifiers.color import (
    ColorClassifierConfig,
    ColorClassifierTrainer,
    ColorRewardDataset,
    CombinedColorRewardDataset,
)
from rl_garden.reward_models.classifiers.shared.model import ResNetBinaryClassifier

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
