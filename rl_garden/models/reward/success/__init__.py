"""Online real-robot success classifier (HIL-SERL-style reward model)."""

from rl_garden.models.reward.success.model import SuccessClassifier, load_classifier_fn

__all__ = [
    "SuccessClassifier",
    "load_classifier_fn",
]
