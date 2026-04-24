"""FiLM conditioning layer — placeholder for future task-conditioned encoders.

Not wired into any algorithm yet; kept as a stub so the design hook exists.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond)[:, :, None, None]
        beta = self.to_beta(cond)[:, :, None, None]
        return gamma * x + beta
