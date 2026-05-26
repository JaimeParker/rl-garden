"""Zero base-policy provider for residual RL debug runs."""
from __future__ import annotations

import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.policies.base_policies.base import BasePolicyOutput, BasePolicyProvider


class ZeroBasePolicy(BasePolicyProvider):
    """Base policy that always returns zero env-space actions."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(observation_space, action_space, device=device)

    def select_action(self, obs: Obs) -> BasePolicyOutput:
        obs = self._obs_to_device(obs)
        if isinstance(obs, dict):
            first = next(iter(obs.values()))
            batch_size = int(first.shape[0])
        else:
            batch_size = int(obs.shape[0])
        actions = torch.zeros(
            (batch_size,) + tuple(self.action_space.shape),
            dtype=torch.float32,
            device=self.device,
        )
        return BasePolicyOutput(actions=actions)
