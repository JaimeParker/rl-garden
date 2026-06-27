"""Distributions used by rl-garden algorithms.

``TruncatedNormal`` is ported from DrQ-v2's ``utils.py`` and used by the DDPG
actor for exploration noise with straight-through clamping.

``safe_tanh_log_det_jacobian`` is ported from FlashSAC's ``distribution.py``
and used by ``NormalTanhPolicy`` for numerically stable tanh log-prob correction.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


def safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log|det J| of tanh: 2*(log2 - x - softplus(-2x)).

    Ported from 3rd_party/FlashSAC/flash_rl/agents/utils/distribution.py.
    Avoids the log(1 - tanh^2(x)) form which loses precision near |x| >> 1.
    """
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class TruncatedNormal(pyd.Normal):
    """Normal distribution truncated to ``[low+eps, high-eps]`` via straight-through.

    Ported from ``3rd_party/drqv2/utils.py:105-126``.  The reparameterized
    sample is clamped and the gradient is passed through unchanged (STE).
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(
        self, clip: float | None = None, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
