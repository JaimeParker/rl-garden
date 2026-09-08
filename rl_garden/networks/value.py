"""Value-function networks for offline RL algorithms."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from rl_garden.networks.actor_critic import BackboneType, _build_trunk
from rl_garden.networks.mlp import Activation, KernelInit, create_mlp


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
        activation_fn: Optional[Activation] = None,
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
            activation_fn=activation_fn,
        )
        self.head = nn.Linear(trunk_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(features))


class ScalarQNetwork(nn.Module):
    """Standalone ``Q(s, a) -> scalar`` network for a single reference/target
    value, not an ensemble critic. Used where a full ``SACPolicy``/
    ``EnsembleQCritic`` (vmap, subsampling, RGBD encoders) is unneeded
    overhead for one network -- e.g. Cal-QL's SARSA/FQE reference value
    (``CalQLCore``) and BPPO's SARSA-fit Q (``BPPOCore``). Flat (non-Dict)
    observations only.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.net = create_mlp(obs_dim + action_dim, 1, list(hidden_dims))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))
