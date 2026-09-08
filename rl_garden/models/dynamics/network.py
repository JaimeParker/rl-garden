"""Ensemble forward-dynamics model for Uni-O4's OPE gating (`UniO4OPE`,
see `rl_garden/algorithms/unio4_ope.py`).

GPU-native rewrite of `3rd_party/Uni-O4/transition_model/models/
{dynamics_model.py, nets.py::EnsembleLinear}` -- the reference is torch
already for the network itself (no numpy here), so `EnsembleLinear`'s
batched-weight einsum forward, `Swish`, and `soft_clamp` are ported as-is.
The GPU-native rewrite work is in `trainer.py` (the reference's
`EnsembleDynamics.step()`/`train()` round-trip through `.cpu().numpy()`).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def soft_clamp(
    x: torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Clamp while keeping a gradient (softplus-based), matching upstream."""
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleLinear(nn.Module):
    """`num_ensemble` independent linear layers, batched into one
    `(num_ensemble, in_dim, out_dim)` weight tensor and applied via
    `einsum` -- verbatim port of upstream's `EnsembleLinear`."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_ensemble = num_ensemble
        self.weight_decay = weight_decay

        self.register_parameter(
            "weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim))
        )
        self.register_parameter(
            "bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim))
        )
        nn.init.trunc_normal_(self.weight, std=1 / (2 * input_dim**0.5))

        self.register_parameter(
            "saved_weight", nn.Parameter(self.weight.detach().clone())
        )
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = torch.einsum("ij,bjk->bik", x, self.weight)
        else:
            x = torch.einsum("bij,bjk->bik", x, self.weight)
        return x + self.bias

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]

    def get_decay_loss(self) -> torch.Tensor:
        return self.weight_decay * 0.5 * (self.weight**2).sum()


class EnsembleDynamicsModel(nn.Module):
    """`(obs, action) -> (mean, logvar)` of `(delta_obs, reward)`, an
    ensemble of `num_ensemble` members with elite-subset tracking. GPU-
    native port of upstream's `EnsembleDynamicsModel` (`transition_model/
    models/dynamics_model.py`)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int, ...]],
        *,
        num_ensemble: int = 7,
        num_elites: int = 5,
        weight_decays: Optional[Sequence[float]] = None,
        with_reward: bool = True,
    ) -> None:
        super().__init__()
        if num_elites > num_ensemble:
            raise ValueError(
                f"num_elites ({num_elites}) must be <= num_ensemble ({num_ensemble})."
            )
        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self.with_reward = with_reward

        self.activation = Swish()

        dims = [obs_dim + action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(dims))
        if len(weight_decays) != len(dims):
            raise ValueError(
                f"weight_decays must have len(hidden_dims)+1={len(dims)} entries, "
                f"got {len(weight_decays)}."
            )
        self.backbones = nn.ModuleList(
            [
                EnsembleLinear(in_dim, out_dim, num_ensemble, wd)
                for in_dim, out_dim, wd in zip(dims[:-1], dims[1:], weight_decays[:-1])
            ]
        )
        self.output_layer = EnsembleLinear(
            dims[-1], 2 * (obs_dim + int(with_reward)), num_ensemble, weight_decays[-1]
        )

        out_dim = obs_dim + int(with_reward)
        self.register_parameter(
            "max_logvar", nn.Parameter(torch.ones(out_dim) * 0.5)
        )
        self.register_parameter(
            "min_logvar", nn.Parameter(torch.ones(out_dim) * -10.0)
        )
        self.register_buffer(
            "elites", torch.arange(num_elites, dtype=torch.long), persistent=True
        )

    def forward(self, obs_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.output_layer.get_decay_loss()
        for layer in self.backbones:
            decay_loss = decay_loss + layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: Sequence[int]) -> None:
        if len(indexes) > self.num_ensemble or max(indexes) >= self.num_ensemble:
            raise ValueError(f"invalid elite indexes {indexes} for num_ensemble={self.num_ensemble}.")
        self.elites = torch.as_tensor(
            list(indexes), dtype=torch.long, device=self.elites.device
        )

    def random_elite_member(self, batch_size: int) -> torch.Tensor:
        """One randomly-chosen elite member index per row, matching
        upstream's `random_elite_idxs`."""
        idx = torch.randint(0, self.elites.numel(), (batch_size,), device=self.elites.device)
        return self.elites[idx]
