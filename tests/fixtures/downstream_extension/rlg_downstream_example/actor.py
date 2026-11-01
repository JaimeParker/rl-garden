"""Split-mean Gaussian actor whose heads all read one features tensor."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from rl_garden.networks import SplitDiagGaussianActor


class SharedInputSplitActor(SplitDiagGaussianActor):
    """``SplitDiagGaussianActor`` for callers that pass a single features
    tensor, as every ``RecurrentPPOPolicy`` method does: each head reads it."""

    def _mean(self, features: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            features = [features] * len(self.heads)
        return super()._mean(features)
