"""Goal-conditioned distance value ("Hilbert representation"), ported from
``HILP/hilp_gcrl/src/special_networks.py::GoalConditionedPhiValue``.

``phi`` is a 2-member ensemble of independent MLP trunks (LayerNorm+GELU
after every *hidden* layer, matching the reference's ``LayerNormRepresentation``
with ``activate_final=False`` -- the final ``skill_dim``-wide layer gets no
activation/norm, confirmed by reading the reference directly rather than
assumed). The value itself is not a learned head: it's the (negated)
Euclidean distance between ``phi(obs)`` and ``phi(goal)`` -- that
distance-shaped parameterization *is* the entire "Hilbert representation"
idea. Two independent ``create_mlp`` trunks rather than a generic
``n``-member ensemble class -- HILP never varies ``n=2``, and
``EnsembleQCritic``'s vmap-ensembling machinery is built for
action-conditioned ``n_critics``-many heads, a different shape.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from rl_garden.networks.mlp import Activation, KernelInit, create_mlp, resolve_activation


class GoalConditionedPhiValue(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        skill_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation_fn: Optional[Activation] = "gelu",
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        act = resolve_activation(activation_fn, default=nn.GELU)
        # output_dim=skill_dim (not net_arch=[*hidden_dims, skill_dim] with
        # output_dim=-1): the final skill_dim-wide layer must be a plain
        # nn.Linear with no activation/LayerNorm after it (activate_final=False
        # in the reference), matching create_mlp's own output_dim>0 shape
        # (hidden layers get norm+activation, the appended output layer does not).
        self.phi_0 = create_mlp(
            obs_dim, skill_dim, list(hidden_dims),
            activation_fn=act, use_layer_norm=True, kernel_init=kernel_init,
        )
        self.phi_1 = create_mlp(
            obs_dim, skill_dim, list(hidden_dims),
            activation_fn=act, use_layer_norm=True, kernel_init=kernel_init,
        )

    def phi(self, obs: torch.Tensor, member: int = 0) -> torch.Tensor:
        """``member=0`` matches the reference's ``get_phi`` (first ensemble
        member only)."""
        return (self.phi_0 if member == 0 else self.phi_1)(obs)

    def forward(
        self, obs: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(v1, v2)``, each ``-sqrt(max(||phi_i(obs)-phi_i(goal)||^2, 1e-6))``."""
        v1 = self._distance_value(self.phi_0, obs, goal)
        v2 = self._distance_value(self.phi_1, obs, goal)
        return v1, v2

    @staticmethod
    def _distance_value(
        phi_net: nn.Module, obs: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        squared_dist = (phi_net(obs) - phi_net(goal)).pow(2).sum(dim=-1).clamp(min=1e-6)
        return -squared_dist.sqrt()
