"""Replay buffer interface.

Replay buffers expose torch tensors throughout the training path. The default
storage is GPU-native; selected vision buffers can optionally use CPU memmaps
and move sampled batches directly to the policy device.
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

    def _advance(self) -> None:
        """Advance the circular write cursor, wrapping when the per-env buffer fills."""
        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0
