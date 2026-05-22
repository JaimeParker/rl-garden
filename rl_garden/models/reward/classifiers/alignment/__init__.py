"""Alignment reward classifier (ResNet)."""

from rl_garden.models.reward.classifiers.alignment.dataset import (
	AlignmentRewardDataset,
	CombinedAlignmentDataset,
)
from rl_garden.models.reward.classifiers.alignment.loop import (
	AlignmentClassifierConfig,
	AlignmentClassifierTrainer,
)

__all__ = [
	"AlignmentRewardDataset",
	"CombinedAlignmentDataset",
	"AlignmentClassifierConfig",
	"AlignmentClassifierTrainer",
]
