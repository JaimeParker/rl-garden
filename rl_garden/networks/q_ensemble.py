"""``QEnsemble``: ``num_q`` independent two-hot-logit Q-heads, ported from
``3rd_party/tdmpc2/tdmpc2/common/layers.py::Ensemble``.

Upstream's ``Ensemble`` uses ``torch.vmap`` over
``tensordict.nn.TensorDictParams`` for a single fused forward pass across the
ensemble. That pulls in ``tensordict``/functorch machinery this port
deliberately avoids (a "no new dependencies" scoping decision, see
``rl_garden.world_models.latent_consistency`` module docstring).
``QEnsemble`` here is a plain ``nn.ModuleList`` looped in Python instead --
slower per step, but a plain ``nn.Module`` that round-trips through
``state_dict()`` with zero special-casing.

Generic enough to live in ``rl_garden.networks`` (not TD-MPC2-specific): any
algorithm wanting an ensemble of two-hot-bin Q-heads over a flat
``[features, action]`` input can reuse it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rl_garden.networks.normed_mlp import mlp


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
