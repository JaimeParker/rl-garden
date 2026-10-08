"""Private network blocks for FQL's state-action flow policy.

FQL uses ``Linear -> GELU -> LayerNorm`` in its critic hidden layers, which
differs from the repository-wide MLP builder.  Keep these blocks beside the
only policy that consumes them so that this algorithm-specific operation order
does not become an accidental general network API.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FQLMLP(nn.Module):
    """FQL dense stack with an unnormalised output layer."""

    def __init__(
        self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, *, layer_norm: bool
    ) -> None:
        super().__init__()
        widths = [input_dim, *hidden_dims, output_dim]
        self.linears = nn.ModuleList(
            [nn.Linear(widths[index], widths[index + 1]) for index in range(len(widths) - 1)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(width, eps=1e-6) for width in hidden_dims] if layer_norm else []
        )
        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for index, linear in enumerate(self.linears):
            x = linear(x)
            if index + 1 < len(self.linears):
                x = F.gelu(x, approximate="tanh")
                if self.layer_norms:
                    x = self.layer_norms[index](x)
        return x


class FQLVectorField(nn.Module):
    def __init__(
        self, features_dim: int, action_dim: int, hidden_dims: Sequence[int], *, time_conditioned: bool
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.time_conditioned = time_conditioned
        self.mlp = FQLMLP(
            features_dim + action_dim + int(time_conditioned), hidden_dims, action_dim, layer_norm=False
        )

    def forward(
        self, features: torch.Tensor, actions: torch.Tensor, times: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.time_conditioned != (times is not None):
            raise ValueError("FQL vector field received incompatible time conditioning.")
        inputs = [features, actions] if times is None else [features, actions, times]
        return self.mlp(torch.cat(inputs, dim=-1))


class FQLEnsembleCritic(nn.Module):
    def __init__(self, features_dim: int, action_dim: int, hidden_dims: Sequence[int], n_critics: int) -> None:
        super().__init__()
        self.critics = nn.ModuleList(
            [FQLMLP(features_dim + action_dim, hidden_dims, 1, layer_norm=True) for _ in range(n_critics)]
        )

    def forward_all(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([features, actions], dim=-1)
        return torch.stack([critic(inputs).squeeze(-1) for critic in self.critics], dim=0)
