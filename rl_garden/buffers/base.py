"""Replay buffer interface.

All buffers in ``rl_garden`` are GPU-native: they store transitions as torch
tensors and never bounce through numpy. This keeps them compatible with
ManiSkill's GPU-parallel envs, which return cuda tensors directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from rl_garden.common.types import Obs, ReplayBufferSample


class BaseReplayBuffer(ABC):
    """Minimal interface every replay buffer implements.

    Layout: ``(per_env_buffer_size, num_envs, ...)`` — one slot per env per
    timestep, indexed circularly.
    """

    num_envs: int
    per_env_buffer_size: int
    pos: int
    full: bool
    storage_device: torch.device
    sample_device: torch.device

    @abstractmethod
    def add(
        self,
        obs: Obs,
        next_obs: Obs,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None: ...

    @abstractmethod
    def sample(self, batch_size: int) -> ReplayBufferSample: ...

    @property
    def size(self) -> int:
        """Number of transitions stored per env."""
        return self.per_env_buffer_size if self.full else self.pos

    def __len__(self) -> int:
        """Total transitions available across all envs."""
        return self.size * self.num_envs
