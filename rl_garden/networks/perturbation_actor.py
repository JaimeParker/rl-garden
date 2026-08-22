"""Perturbation network: BCQ's ``Actor`` -- (state, action) -> perturbed action.

Verified against a full clone of ``sfujim/BCQ/continuous_BCQ/BCQ.py::Actor``,
not just fetched raw files: two hidden layers (400, 300 by default), ReLU,
output ``phi * max_action * tanh(...)`` added to the input action and
clamped back into the action space. Reused verbatim (confirmed against the
cloned upstream ``Wenxuan-Zhou/PLAS/algos.py::ActorPerturbation``, whose
``l4-l6`` layers are this exact same shape) as the second stage of PLAS's
``LatentPerturbation`` ("-P") variant, so this class is not BCQ-specific
despite the name matching BCQ's own class.

Unlike BCQ's symmetric scalar ``max_action``, this uses per-dimension
``action_scale``/``action_bias`` buffers (the same asymmetric-action-space
convention already used by ``DeterministicTanhActor``/``ConditionalVAE``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.networks.actor_critic import BackboneType, _build_trunk
from rl_garden.networks.mlp import KernelInit


class PerturbationActor(nn.Module):
    """``phi``-bounded perturbation of a candidate action, given state features."""

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int] = (400, 300),
        *,
        phi: float = 0.05,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 32,
        dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
    ) -> None:
        super().__init__()
        self.phi = phi
        act_dim = int(np.prod(action_space.shape))
        self.trunk, trunk_dim = _build_trunk(
            features_dim + act_dim,
            hidden_dims,
            backbone_type=backbone_type,
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
        )
        self.fc_perturbation = nn.Linear(trunk_dim, act_dim)

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.trunk(torch.cat([features, action], dim=-1))
        perturbation = self.phi * self.action_scale * torch.tanh(self.fc_perturbation(x))
        low = self.action_bias - self.action_scale
        high = self.action_bias + self.action_scale
        return (perturbation + action).clamp(low, high)
