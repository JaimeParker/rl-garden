"""Alignment reward classifier (ResNet)."""

from rl_garden.reward_models.classifiers.alignment.dataset import (
	AlignmentRewardDataset,
	CombinedAlignmentDataset,
)
from rl_garden.reward_models.classifiers.alignment.loop import (
	AlignmentClassifierConfig,
	AlignmentClassifierTrainer,
)

__all__ = [
	"AlignmentRewardDataset",
	"CombinedAlignmentDataset",
	"AlignmentClassifierConfig",
	"AlignmentClassifierTrainer",
]
