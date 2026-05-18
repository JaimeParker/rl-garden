"""Value-function networks for offline RL algorithms."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from rl_garden.networks.actor_critic import BackboneType, _build_trunk
from rl_garden.networks.mlp import KernelInit


class ValueNetwork(nn.Module):
    """State value network V(s) used by IQL-style offline algorithms."""

    def __init__(
        self,
        features_dim: int,
        hidden_dims: Sequence[int],
        *,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 32,
        dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
    ) -> None:
        super().__init__()
        self.trunk, trunk_dim = _build_trunk(
            features_dim,
            hidden_dims,
            backbone_type=backbone_type,
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
        )
        self.head = nn.Linear(trunk_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(features))
