"""Discrete Double-DQN Q-head used by :class:`~rl_garden.algorithms.rlpd_hybrid.RLPDHybrid`
for a hybrid continuous-arm + discrete-gripper action space (HIL-SERL's
``GraspCritic``). Unlike the continuous critic, this network is not an
ensemble and does not take an action as input -- it maps state features
directly to one Q-value per discrete gripper action (open/hold/close).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rl_garden.networks.mlp import KernelInit, create_mlp


class DiscreteCritic(nn.Module):
    """``Q(features) -> (batch, n_actions)`` Q-values for a discrete action set."""

    def __init__(
        self,
        features_dim: int,
        hidden_dims: Sequence[int],
        *,
        n_actions: int = 3,
        use_layer_norm: bool = False,
        dropout_rate: float | None = None,
        kernel_init: KernelInit | None = None,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.net = create_mlp(
            input_dim=features_dim,
            output_dim=n_actions,
            net_arch=hidden_dims,
            use_layer_norm=use_layer_norm,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
