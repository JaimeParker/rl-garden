"""GAIL discriminator network: a state-action MLP producing a single logit,
matching ``imitation``'s own default ``BasicRewardNet(use_state=True,
use_action=True, use_next_state=False, use_done=False)`` configuration
(``3rd_party/imitation/src/imitation/rewards/reward_nets.py``). A high logit
means "looks like an expert transition."
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.networks.mlp import KernelInit, create_mlp


class GAILDiscriminator(nn.Module):
    """``forward(obs, action) -> logit``, Box observations only."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        net_arch: Sequence[int] = (32, 32),
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(
                "GAILDiscriminator only supports Box observation spaces, got "
                f"{type(observation_space)}."
            )
        obs_dim = int(torch.tensor(observation_space.shape).prod().item())
        action_dim = int(torch.tensor(action_space.shape).prod().item())
        self.mlp = create_mlp(
            obs_dim + action_dim,
            1,
            net_arch,
            kernel_init=kernel_init,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([torch.flatten(obs, 1), torch.flatten(action, 1)], dim=1)
        return self.mlp(inputs).squeeze(-1)
