"""State-only DDPM epsilon-predictor MLP for DPPO / diffusion BC.

Ported from ``3rd_party/dppo/model/diffusion/mlp_diffusion.py::DiffusionMLP``,
verified against source directly (image conditioning / ``VisionDiffusionMLP``
is out of scope -- state-only). The time embedding matches
``model/diffusion/modules.py::SinusoidalPosEmb`` plus the reference's own
``Linear -> Mish -> Linear`` projection, which is hardcoded to Mish in the
reference regardless of the trunk's own activation choice -- kept hardcoded
here too, for fidelity.

``residual_style=True`` reproduces
``model/common/mlp.py::ResidualMLP``/``TwoLayerPreActivationResNetLinear``
exactly (pre-activation, two same-width linear layers per block, no width
expansion). This is a different convention from
``rl_garden.networks.mlp.MLPResNet`` (WSRL's 4x-expansion, post-activation
block), so it is reimplemented here rather than reused -- reusing
``MLPResNet`` would silently diverge from the DPPO reference architecture
that ``residual_style=True`` is meant to reproduce. ``build_diffusion_mlp_head``
is exported so ``DPPOPolicy``'s obs-only critic (``CriticObs`` in the
reference, also built from this same ``ResidualMLP``/``MLP`` pair) can share
it without duplicating the block implementation.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from rl_garden.networks.mlp import Activation, KernelInit, _apply_kernel_init, resolve_activation


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=x.device) * -scale)
        emb = x[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class _PreActivationResidualBlock(nn.Module):
    """Matches ``model/common/mlp.py::TwoLayerPreActivationResNetLinear``."""

    def __init__(self, hidden_dim: int, activation_fn: type[nn.Module]) -> None:
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.l1(self.act(x))
        h = self.l2(self.act(h))
        return residual + h


def build_diffusion_mlp_head(
    dim_list: Sequence[int],
    *,
    activation_fn: type[nn.Module],
    residual_style: bool,
) -> nn.Module:
    """``dim_list = [input_dim, *mlp_dims, output_dim]``. Matches the
    reference's ``MLP``/``ResidualMLP`` with ``out_activation_type="Identity"``
    (no activation after the final linear layer)."""
    if not residual_style:
        layers: list[nn.Module] = []
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            if i < len(dim_list) - 2:
                layers.append(activation_fn())
        return nn.Sequential(*layers)

    hidden_dim = dim_list[1]
    num_hidden_layers = len(dim_list) - 3
    if num_hidden_layers % 2 != 0:
        raise ValueError(
            "residual_style requires len(mlp_dims) to be odd "
            f"(got {len(dim_list) - 2} mlp_dims)."
        )
    modules: list[nn.Module] = [nn.Linear(dim_list[0], hidden_dim)]
    modules.extend(
        _PreActivationResidualBlock(hidden_dim, activation_fn)
        for _ in range(0, num_hidden_layers, 2)
    )
    modules.append(nn.Linear(hidden_dim, dim_list[-1]))
    return nn.Sequential(*modules)


class DiffusionMLP(nn.Module):
    """``eps_theta(x_t, t, obs) -> (B, horizon_steps, action_dim)``."""

    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        cond_dim: int,
        *,
        time_dim: int = 16,
        mlp_dims: Sequence[int] = (256, 256),
        activation_fn: Optional[Activation] = "mish",
        residual_style: bool = False,
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.time_dim = time_dim
        act = resolve_activation(activation_fn, default=nn.Mish)

        self.time_embedding = nn.Sequential(
            _SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        output_dim = action_dim * horizon_steps
        input_dim = time_dim + output_dim + cond_dim
        self.mlp_mean = build_diffusion_mlp_head(
            [input_dim] + list(mlp_dims) + [output_dim],
            activation_fn=act,
            residual_style=residual_style,
        )
        _apply_kernel_init(self, kernel_init)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict) -> torch.Tensor:
        batch, horizon, action_dim = x.shape
        x = x.reshape(batch, -1)
        state = cond["state"].reshape(batch, -1)
        time_emb = self.time_embedding(time.view(batch, 1)).view(batch, self.time_dim)
        h = torch.cat([x, time_emb, state], dim=-1)
        out = self.mlp_mean(h)
        return out.view(batch, horizon, action_dim)
