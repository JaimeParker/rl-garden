"""Base class for feature extractors.

API-compatible with Stable-Baselines3 ``BaseFeaturesExtractor``: exposes a
``features_dim`` property that the policy heads rely on to size their MLPs.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


def image_needs_normalization(space: spaces.Box) -> bool:
    """Return True when the image space stores raw [0, 255] pixel data.

    Covers both uint8 spaces (ManiSkill default) and float32 spaces whose
    declared upper bound exceeds 1.0 (un-normalized pixels from other sources).
    A float32 space with high=1.0 is assumed already normalized.
    """
    return space.dtype == np.uint8 or float(space.high.max()) > 1.0


class TokenAndPropFeatureConfig(TypedDict):
    """Structured feature layout declaration for token-based extractors.

    Invariant: ``num_patches * patch_dim + prop_dim == features_dim``.
    """

    layout: Literal["token_and_prop"]
    num_patches: int
    patch_dim: int
    prop_dim: int


class BaseFeaturesExtractor(nn.Module):
    def __init__(self, observation_space: gym.Space, features_dim: int) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def extract(self, obs, stop_gradient: bool = False) -> torch.Tensor:
        """Extract features with optional stop-gradient semantics.

        Visual extractors can override this to stop gradients only for image
        branches. The default covers state-only and simple custom extractors.
        """
        features = self.forward(obs)
        return features.detach() if stop_gradient else features

    def structured_feature_config(self) -> Optional[TokenAndPropFeatureConfig]:
        """Return structured feature layout, or None for flat features.

        Override in extractors whose output has token-and-prop structure so
        that downstream policy heads (e.g. spatial Q-critics, actor adapters)
        can self-configure without inspecting the extractor type.
        """
        return None

    def prepare_batch(
        self,
        obs: dict,
        next_obs: Optional[dict] = None,
    ) -> None:
        """Pre-process a training batch before the update step.

        Default is a no-op. Override to cache expensive encodings or apply
        data augmentation once per batch (e.g. ViT feature caching).

        Both ``obs`` and ``next_obs`` dicts may be mutated in-place to store
        cache keys. Each is a fresh dict returned by the replay buffer's
        ``sample()`` call, so mutation does not pollute replay storage.
        """
