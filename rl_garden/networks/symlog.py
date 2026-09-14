"""Symmetric log transform (DreamerV3-style), ported from
``3rd_party/tdmpc2/tdmpc2/common/math.py``.

Shared numeric primitive: TD-MPC2 uses it to build two-hot reward/value bin
targets (``rl_garden.networks.twohot``); DreamerV3 uses the same transform for
symlog-MSE decoder heads and its own two-hot reward/critic targets.
"""
from __future__ import annotations

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm: ``sign(x) * log(1 + |x|)``."""
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of ``symlog``."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
