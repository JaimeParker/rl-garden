"""Replay buffers with residual-RL base action fields."""
from __future__ import annotations

from typing import Callable, Optional

import torch
from gymnasium import spaces

from rl_garden.buffers.dict_buffer import DictReplayBuffer
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.types import Obs, ResidualReplayBufferSample


class ResidualReplayBufferMixin:
    """Mixin adding normalized base-action storage to a replay buffer.

    Usage:
        class ResidualTensorReplayBuffer(ResidualReplayBufferMixin, TensorReplayBuffer):
            pass
    """

    def __init__(
        self,
        observation_space: spaces.Box | spaces.Dict,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
        observation_transform: Optional[Callable[[Obs], Obs]] = None,
    ) -> None:
        self.observation_transform = observation_transform
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            num_envs=num_envs,
            buffer_size=buffer_size,
            storage_device=storage_device,
            sample_device=sample_device,
        )
        shape = (self.per_env_buffer_size, num_envs) + tuple(action_space.shape)
        self.base_actions = torch.zeros(shape, device=self.storage_device)
        self.next_base_actions = torch.zeros(shape, device=self.storage_device)

    def add(
        self,
        obs: Obs,
        next_obs: Obs,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        *,
        base_actions: torch.Tensor,
        next_base_actions: torch.Tensor,
    ) -> None:
        if self.observation_transform is not None:
            obs = self.observation_transform(obs)
            next_obs = self.observation_transform(next_obs)
        if self.storage_device.type == "cpu":
            base_actions = base_actions.cpu()
            next_base_actions = next_base_actions.cpu()
        self.base_actions[self.pos] = base_actions
        self.next_base_actions[self.pos] = next_base_actions
        super().add(obs, next_obs, action, reward, done)

    def _index_batch(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> ResidualReplayBufferSample:
        sample = super()._index_batch(batch_inds, env_inds)
        return ResidualReplayBufferSample(
            obs=sample.obs,
            next_obs=sample.next_obs,
            actions=sample.actions,
            rewards=sample.rewards,
            dones=sample.dones,
            base_actions=self.base_actions[batch_inds, env_inds].to(self.sample_device),
            next_base_actions=self.next_base_actions[batch_inds, env_inds].to(
                self.sample_device
            ),
        )


class ResidualTensorReplayBuffer(ResidualReplayBufferMixin, TensorReplayBuffer):
    """Tensor replay buffer that also stores normalized base actions."""

    pass


class ResidualDictReplayBuffer(ResidualReplayBufferMixin, DictReplayBuffer):
    """Dict replay buffer that also stores normalized base actions."""

    pass
