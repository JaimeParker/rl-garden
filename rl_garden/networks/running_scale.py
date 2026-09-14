"""Running trimmed-range scale estimator, ported from
``3rd_party/tdmpc2/tdmpc2/common/scale.py``.

Rescales Q-values before the actor loss so the entropy/value terms stay
comparable in magnitude across tasks with different reward scales.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RunningScale(nn.Module):
    """Running trimmed-range scale estimator (5th-95th percentile spread)."""

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.tau = tau
        self.register_buffer("value", torch.ones(1, dtype=torch.float32))
        self.register_buffer("_percentiles", torch.tensor([5.0, 95.0], dtype=torch.float32))

    def _positions(self, x_shape: int):
        positions = self._percentiles * (x_shape - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled = torch.where(ceiled > x_shape - 1, torch.full_like(ceiled, x_shape - 1), ceiled)
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        return floored.long(), ceiled.long(), weight_floored.unsqueeze(1), weight_ceiled.unsqueeze(1)

    def _percentile(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype, x_shape = x.dtype, x.shape
        x = x.flatten(1, x.ndim - 1)
        in_sorted = torch.sort(x, dim=0).values
        floored, ceiled, weight_floored, weight_ceiled = self._positions(x.shape[0])
        d0 = in_sorted[floored] * weight_floored
        d1 = in_sorted[ceiled] * weight_ceiled
        return (d0 + d1).reshape(-1, *x_shape[1:]).to(x_dtype)

    def update(self, x: torch.Tensor) -> None:
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self.value.data.lerp_(value, self.tau)

    def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update:
            self.update(x)
        return x / self.value

    def __repr__(self) -> str:
        return f"RunningScale(S: {self.value})"
