"""Color reward classifier (ResNet)."""

from rl_garden.models.reward.classifiers.color.dataset import (
    ColorRewardDataset,
    CombinedColorRewardDataset,
)
from rl_garden.models.reward.classifiers.color.loop import (
    ColorClassifierConfig,
    ColorClassifierTrainer,
)

__all__ = [
    "ColorRewardDataset",
    "CombinedColorRewardDataset",
    "ColorClassifierConfig",
    "ColorClassifierTrainer",
]
