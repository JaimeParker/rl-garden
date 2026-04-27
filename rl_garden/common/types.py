"""Shared type aliases and data containers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import torch

TensorDict = Dict[str, torch.Tensor]
Obs = Union[torch.Tensor, TensorDict]
Schedule = Callable[[float], float]  # progress_remaining in [0, 1] -> lr


@dataclass
class ReplayBufferSample:
    """A single mini-batch drawn from a replay buffer.

    ``obs`` and ``next_obs`` are ``torch.Tensor`` for flat observations or
    ``Dict[str, torch.Tensor]`` for dict observations (e.g. RGBD + state).
    All fields live on the sample device.
    """

    obs: Obs
    next_obs: Obs
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


@dataclass
class MCReplayBufferSample(ReplayBufferSample):
    """Replay buffer sample with Monte Carlo returns for Cal-QL."""

    mc_returns: Optional[torch.Tensor] = None
