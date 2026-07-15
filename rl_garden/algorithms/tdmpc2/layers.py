"""Small nn.Module building blocks ported from
``3rd_party/tdmpc2/tdmpc2/common/layers.py`` / ``common/scale.py``.

``Ensemble`` in the upstream code uses ``torch.vmap`` over
``tensordict.nn.TensorDictParams`` for a single fused forward pass across the
Q-ensemble. That pulls in ``tensordict``/functorch machinery this port
deliberately avoids (see the "no new dependencies" decision in
``docs/hil_serl_roadmap.md``-style scoping notes for this port). ``QEnsemble``
here is a plain ``nn.ModuleList`` looped in Python instead -- slower per step,
but a plain ``nn.Module`` that round-trips through ``state_dict()`` with zero
special-casing, which is what the checkpoint design in this port relies on.
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


class QEnsemble(nn.Module):
    """``num_q`` independent MLP heads, each predicting two-hot bin logits."""

    def __init__(
        self,
        in_dim: int,
        mlp_dims: int | list[int],
        num_bins: int,
        num_q: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.qs = nn.ModuleList(
            [mlp(in_dim, mlp_dims, max(num_bins, 1), dropout=dropout) for _ in range(num_q)]
        )

    def __len__(self) -> int:
        return len(self.qs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns ``(num_q, *z.shape[:-1], max(num_bins, 1))``."""
        return torch.stack([q(z) for q in self.qs], dim=0)


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
