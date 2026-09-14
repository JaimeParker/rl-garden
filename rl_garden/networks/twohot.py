"""Two-hot scalar<->distribution encoding, ported from
``3rd_party/tdmpc2/tdmpc2/common/math.py``.

Shared numeric primitive: TD-MPC2's reward/value heads and DreamerV3's
reward/critic heads both predict a discrete distribution over ``num_bins``
symlog-spaced bins instead of a scalar, trained with ``soft_ce`` against a
soft two-hot target, and decoded back to a scalar with ``two_hot_inv``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_garden.networks.symlog import symexp, symlog


def two_hot(
    x: torch.Tensor, num_bins: int, vmin: float, vmax: float, bin_size: float
) -> torch.Tensor:
    """Converts a batch of scalars to soft two-hot encoded targets."""
    if num_bins == 0:
        return x
    if num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), vmin, vmax).squeeze(-1)
    bin_idx = torch.floor((x - vmin) / bin_size)
    bin_offset = ((x - vmin) / bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Converts a batch of soft two-hot encoded vectors back to scalars."""
    if num_bins == 0:
        return x
    if num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(vmin, vmax, num_bins, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)


def soft_ce(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_bins: int,
    vmin: float,
    vmax: float,
    bin_size: float,
) -> torch.Tensor:
    """Cross-entropy loss between predicted bin logits and soft two-hot targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, num_bins, vmin, vmax, bin_size)
    return -(target * pred).sum(-1, keepdim=True)
