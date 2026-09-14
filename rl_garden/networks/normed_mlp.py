"""LayerNorm+Mish MLP building blocks, ported from
``3rd_party/tdmpc2/tdmpc2/common/layers.py``.

Shared numeric primitive: TD-MPC2's dynamics/reward/actor/critic heads are all
built from ``mlp()``/``NormedLinear``; DreamerV3 (r2dreamer) uses the same
LayerNorm-normalized-hidden-layer pattern for its MLP heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimNorm(nn.Module):
    """Simplicial normalization (https://arxiv.org/abs/2204.00616)."""

    def __init__(self, simnorm_dim: int) -> None:
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = torch.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self) -> str:
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """Linear layer with LayerNorm, activation, and optional dropout."""

    def __init__(self, *args, dropout: float = 0.0, act: nn.Module | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.ln(x))


def mlp(
    in_dim: int,
    mlp_dims: int | list[int],
    out_dim: int,
    act: nn.Module | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    """MLP with LayerNorm + Mish hidden layers, matching upstream's ``common.layers.mlp``."""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + list(mlp_dims) + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    if act is not None:
        layers.append(NormedLinear(dims[-2], dims[-1], act=act))
    else:
        layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)
