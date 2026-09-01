"""Reward/mask relabeling head for ExPLORe-style optimistic reward labeling
(Li et al. 2023, ``3rd_party/ExPLORe/rlpd/agents/rm.py``).

Two independent MLPs over ``(obs, action)`` -- ``r_net`` regresses reward with
MSE, ``m_net`` predicts the episode mask (``1 - done``) with BCE-with-logits.
Trained only on transitions the caller supplies (ExPLORe trains it on online
data only); this module has no opinion on which data it sees or how its
predictions get used, keeping it reusable beyond ExPLORe.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rl_garden.networks.mlp import Activation, KernelInit, create_mlp, resolve_activation


class RewardMaskRelabeler(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        net_arch: tuple[int, ...] = (256, 256),
        activation_fn: Optional[Activation] = None,
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        input_dim = obs_dim + action_dim
        resolved_activation = resolve_activation(activation_fn, default=nn.ReLU)
        self.r_net = create_mlp(
            input_dim, 1, list(net_arch), activation_fn=resolved_activation, kernel_init=kernel_init
        )
        self.m_net = create_mlp(
            input_dim, 1, list(net_arch), activation_fn=resolved_activation, kernel_init=kernel_init
        )

    def predict_reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.r_net(torch.cat([obs, action], dim=-1)).squeeze(-1)

    def predict_mask(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.m_net(torch.cat([obs, action], dim=-1)).squeeze(-1))

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x = torch.cat([obs, action], dim=-1)
        reward_pred = self.r_net(x).squeeze(-1)
        mask_logit = self.m_net(x).squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, reward)
        mask_loss = F.binary_cross_entropy_with_logits(mask_logit, mask)
        total = reward_loss + mask_loss
        return total, {
            "relabeler/reward_loss": float(reward_loss.detach()),
            "relabeler/mask_loss": float(mask_loss.detach()),
        }
