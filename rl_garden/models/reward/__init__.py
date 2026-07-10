"""Reward model datasets, networks, and training helpers.

Two kinds of reward model live here today: ``classifiers/`` (offline,
HDF5-labeled binary classifiers trained from pre-collected episodes) and
``success/`` (an online real-robot success classifier, HIL-SERL-style). Future
reward-model kinds (VLM-based, rule-based, ...) are expected to land as
further siblings, not inside either of these two.
"""

from rl_garden.models.reward.classifiers import (
    AlignmentClassifierConfig,
    AlignmentClassifierTrainer,
    AlignmentRewardDataset,
    ColorClassifierConfig,
    ColorClassifierTrainer,
    ColorRewardDataset,
    CombinedAlignmentDataset,
    CombinedColorRewardDataset,
    ResNetBinaryClassifier,
)
from rl_garden.models.reward.success import SuccessClassifier, load_classifier_fn

__all__ = [
    "AlignmentClassifierConfig",
    "AlignmentClassifierTrainer",
    "AlignmentRewardDataset",
    "ColorClassifierConfig",
    "ColorClassifierTrainer",
    "ColorRewardDataset",
    "CombinedAlignmentDataset",
    "CombinedColorRewardDataset",
    "ResNetBinaryClassifier",
    "SuccessClassifier",
    "load_classifier_fn",
]
