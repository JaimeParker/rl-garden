"""Random Network Distillation novelty bonus (Burda et al. 2018), ported for
ExPLORe-style exploration (``3rd_party/ExPLORe/rlpd/agents/rnd.py``'s
``StateActionFeature`` pair): a frozen random target network and a trained
predictor network over ``(obs, action)``; the squared prediction error is the
intrinsic reward bonus. Generic novelty-bonus module, independent of any
specific algorithm's reward-relabeling scheme.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from rl_garden.networks.mlp import Activation, KernelInit, create_mlp, resolve_activation


class RNDBonus(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        feature_dim: int = 256,
        net_arch: tuple[int, ...] = (256, 256),
        activation_fn: Optional[Activation] = None,
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        input_dim = obs_dim + action_dim
        resolved_activation = resolve_activation(activation_fn, default=nn.ReLU)
        self.predictor = create_mlp(
            input_dim,
            feature_dim,
            list(net_arch),
            activation_fn=resolved_activation,
            kernel_init=kernel_init,
        )
        self.target = create_mlp(
            input_dim,
            feature_dim,
            list(net_arch),
            activation_fn=resolved_activation,
            kernel_init=kernel_init,
        )
        for param in self.target.parameters():
            param.requires_grad_(False)

    def _squared_error(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        predicted = self.predictor(x)
        with torch.no_grad():
            target = self.target(x)
        return (predicted - target).pow(2).mean(dim=-1)

    def bonus(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._squared_error(obs, action)

    def loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._squared_error(obs, action).mean()
