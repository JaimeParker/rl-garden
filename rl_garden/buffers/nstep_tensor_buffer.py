"""N-step flat-tensor replay buffer for Box (state-only) observations.

Mirrors ``NStepDictReplayBuffer`` (nstep_buffer.py) with flat tensor storage
instead of DictArray.  No mmap support — always GPU/CPU tensor resident.
"""
from __future__ import annotations

import torch
from gymnasium import spaces

from rl_garden.buffers._nstep_sampling import NStepSamplingMixin
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.types import NStepReplayBufferSample


class NStepTensorReplayBuffer(NStepSamplingMixin, BaseReplayBuffer):
    """N-step replay buffer for flat Box observations (GPU-native).

    Storage layout: ``(per_env_buffer_size, num_envs, *obs_shape)``.
    Episode boundaries are tracked via per-transition episode-id and step-id
    tensors so n-step look-ahead never crosses episode boundaries.

    Parameters
    ----------
    nstep : int
        Number of steps for n-step return accumulation.
    gamma : float
        Per-step discount factor.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        nstep: int = 3,
        gamma: float = 0.99,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NStepTensorReplayBuffer requires a flat Box observation space."
        )
        if nstep < 1:
            raise ValueError(f"nstep must be >= 1, got {nstep}")

        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.per_env_buffer_size = buffer_size // num_envs
        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)
        self.nstep = nstep
        self.gamma = gamma
        self.pos = 0
        self.full = False

        obs_shape = tuple(observation_space.shape)
        act_shape = tuple(action_space.shape)
        shape = (self.per_env_buffer_size, num_envs)

        self.obs = torch.zeros(shape + obs_shape, device=self.storage_device)
        self.next_obs = torch.zeros(shape + obs_shape, device=self.storage_device)
        self.actions = torch.zeros(shape + act_shape, device=self.storage_device)
        self.rewards = torch.zeros(shape, device=self.storage_device)
        self.dones = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)
        self.episode_ends = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)

        self._ep_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_ep_id = torch.zeros(num_envs, dtype=torch.long, device=self.storage_device)
        self._step_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_step_id = torch.zeros(num_envs, dtype=torch.long, device=self.storage_device)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        episode_end: torch.Tensor,
    ) -> None:
        if self.storage_device.type == "cpu":
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()
            episode_end = episode_end.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        done_bool = done.to(self.storage_device).bool()
        episode_end_bool = episode_end.to(self.storage_device).bool()
        self.dones[self.pos] = done_bool
        self.episode_ends[self.pos] = episode_end_bool

        self._ep_id[self.pos] = self._current_ep_id
        self._step_id[self.pos] = self._current_step_id
        self._current_ep_id += episode_end_bool.long()
        self._current_step_id += 1

        self._advance()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _next_obs_at(self, inds, env_inds) -> torch.Tensor:
        return self.next_obs[inds, env_inds]

    def sample(self, batch_size: int) -> NStepReplayBufferSample:
        upper = self.size
        if upper < self.nstep:
            raise RuntimeError(
                f"Buffer has only {upper} transitions per env but nstep={self.nstep}. "
                "Wait for more data before sampling."
            )

        batch_inds, env_inds = self._sample_valid_indices(batch_size, upper)
        rewards, discounts, next_obs = self._compute_nstep_batch(batch_inds, env_inds)

        return NStepReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=next_obs.to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=rewards.to(self.sample_device),
            dones=(discounts == 0.0).to(self.sample_device),
            discounts=discounts.to(self.sample_device),
        )
