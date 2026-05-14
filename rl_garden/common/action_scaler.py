"""Action scaling helpers for residual RL.

Residual SAC follows the resfit convention: the algorithm, critic, and replay
buffer use normalized actions in ``[-1, 1]`` while the environment may consume a
different raw action range.
"""
from __future__ import annotations

import torch
from gymnasium import spaces


class ActionScaler:
    """Min-max scaler between env action space and normalized ``[-1, 1]``."""

    def __init__(
        self,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        if low.shape != high.shape:
            raise ValueError(
                f"action_low/action_high shape mismatch: {tuple(low.shape)} vs {tuple(high.shape)}"
            )
        if torch.any(high < low):
            raise ValueError("ActionScaler requires action_high >= action_low in every dimension.")
        self.low = low
        self.high = high
        self._range = torch.clamp(high - low, min=1e-8)

    @classmethod
    def from_action_space(
        cls,
        action_space: spaces.Box,
        device: torch.device | str = "cpu",
    ) -> "ActionScaler":
        if not isinstance(action_space, spaces.Box):
            raise TypeError(f"ActionScaler requires a Box action space, got {type(action_space)}")
        return cls(
            torch.as_tensor(action_space.low, dtype=torch.float32),
            torch.as_tensor(action_space.high, dtype=torch.float32),
            device=device,
        )

    def to(self, device: torch.device | str) -> "ActionScaler":
        return ActionScaler(self.low.to(device), self.high.to(device), device=device)

    def scale(self, action: torch.Tensor) -> torch.Tensor:
        device = action.device if isinstance(action, torch.Tensor) else self.device
        low = self.low.to(device)
        high = self.high.to(device)
        range_vals = self._range.to(device)
        action = torch.as_tensor(action, dtype=torch.float32, device=device)
        action = torch.clamp(action, low, high)
        return 2.0 * (action - low) / range_vals - 1.0

    def unscale(self, normalized_action: torch.Tensor) -> torch.Tensor:
        device = (
            normalized_action.device
            if isinstance(normalized_action, torch.Tensor)
            else self.device
        )
        low = self.low.to(device)
        high = self.high.to(device)
        normalized_action = torch.as_tensor(
            normalized_action, dtype=torch.float32, device=device
        )
        normalized_action = torch.clamp(normalized_action, -1.0, 1.0)
        return low + 0.5 * (normalized_action + 1.0) * (high - low)
