"""PLAS's latent-space policy: state features -> a bounded vector in the VAE's latent space.

Cross-checked against ``Wenxuan-Zhou/PLAS/algos.py::Actor`` as used by the
``Latent`` policy class: two hidden layers (400, 300 by default), ReLU,
output ``max_latent_action * tanh(...)``. The output dimensionality is the
shared ``ConditionalVAE``'s ``latent_dim``, not the action space -- the
actor never touches raw actions directly; ``vae.decode(state, z=...)``
does that downstream.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from rl_garden.networks.actor_critic import BackboneType, _build_trunk
from rl_garden.networks.mlp import KernelInit


class LatentActor(nn.Module):
    """Deterministic actor producing a ``[-max_latent_action, max_latent_action]``-bounded latent."""

    def __init__(
        self,
        features_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (400, 300),
        *,
        max_latent_action: float = 2.0,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 32,
        dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
    ) -> None:
        super().__init__()
        self.max_latent_action = max_latent_action
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
        self.fc_latent = nn.Linear(trunk_dim, latent_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.trunk(features)
        return self.max_latent_action * torch.tanh(self.fc_latent(x))

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        return self(features)
