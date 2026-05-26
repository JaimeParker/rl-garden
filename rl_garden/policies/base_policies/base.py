"""Base-policy provider interface for residual RL."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.types import Obs


@dataclass(frozen=True)
class BasePolicyOutput:
    """Action output consumed by residual RL.

    ``actions`` are raw env-space actions. Residual algorithms own conversion
    into their internal normalized action coordinates.
    """

    actions: torch.Tensor
    info: Optional[dict[str, Any]] = None


class BasePolicyProvider(nn.Module, ABC):
    """Abstract base class for policies that provide residual base actions."""

    observation_space: spaces.Space
    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if not isinstance(action_space, spaces.Box):
            raise TypeError(f"BasePolicyProvider requires a Box action space, got {action_space}.")
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)

    @abstractmethod
    def select_action(self, obs: Obs) -> BasePolicyOutput:
        """Return raw env-space actions for a batched observation."""

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        del env_ids

    def bind_env(self, env: Any) -> None:
        del env

    def to(self, *args, **kwargs):  # type: ignore[override]
        module = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            device = kwargs.get("device", args[0] if args else self.device)
            self.device = torch.device(device)
        return module

    def _obs_to_device(self, obs: Obs) -> Obs:
        if isinstance(obs, dict):
            return {
                key: value if value.device == self.device else value.to(self.device)
                for key, value in obs.items()
            }
        return obs if obs.device == self.device else obs.to(self.device)

    def _format_actions(self, actions: Any) -> torch.Tensor:
        tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        expected_shape = tuple(self.action_space.shape)
        if tuple(tensor.shape[1:]) != expected_shape:
            raise ValueError(
                "Base policy action shape mismatch: "
                f"expected batch shape (*, {expected_shape}), got {tuple(tensor.shape)}."
            )
        return tensor
