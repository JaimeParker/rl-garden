"""Flat tensor replay buffer for non-dict (e.g. state-only) observations.

Structurally identical to ManiSkill's ``ReplayBuffer`` in
``examples/baselines/sac/sac.py``: a ``(per_env_buffer_size, num_envs, ...)``
torch tensor per field, GPU-resident by default.
"""
from __future__ import annotations

import torch
from gymnasium import spaces

from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.types import ReplayBufferSample


class TensorReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "TensorReplayBuffer expects a flat Box observation space; "
            "use DictReplayBuffer for Dict observations."
        )

        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.per_env_buffer_size = buffer_size // num_envs
        self.pos = 0
        self.full = False
        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)

        obs_shape = tuple(observation_space.shape)
        act_shape = tuple(action_space.shape)
        shape = (self.per_env_buffer_size, num_envs)

        self.obs = torch.zeros(shape + obs_shape, device=self.storage_device)
        self.next_obs = torch.zeros(shape + obs_shape, device=self.storage_device)
        self.actions = torch.zeros(shape + act_shape, device=self.storage_device)
        self.rewards = torch.zeros(shape, device=self.storage_device)
        self.dones = torch.zeros(shape, device=self.storage_device)

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        if self.storage_device.type == "cpu":
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        upper = self.per_env_buffer_size if self.full else self.pos
        batch_inds = torch.randint(0, upper, size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )
