"""N-step flat-tensor replay buffer for Box (state-only) observations.

Mirrors ``NStepDictReplayBuffer`` (nstep_buffer.py) with flat tensor storage
instead of DictArray.  No mmap support — always GPU/CPU tensor resident.
"""
from __future__ import annotations

import torch
from gymnasium import spaces

from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.types import NStepReplayBufferSample


class NStepTensorReplayBuffer(BaseReplayBuffer):
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

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _valid_nstep_batch(
        self,
        batch_inds: torch.Tensor,
        env_inds: torch.Tensor,
    ) -> torch.Tensor:
        """Return validity mask for candidate n-step starting positions."""
        target_ep = self._ep_id[batch_inds, env_inds]
        base_step = self._step_id[batch_inds, env_inds]
        valid = (target_ep >= 0) & (base_step >= 0)
        active = valid.clone()

        for i in range(1, self.nstep):
            prev_idx = (batch_inds + i - 1) % self.per_env_buffer_size
            active &= ~self.episode_ends[prev_idx, env_inds]

            idx = (batch_inds + i) % self.per_env_buffer_size
            contiguous = (
                (self._ep_id[idx, env_inds] == target_ep)
                & (self._step_id[idx, env_inds] == base_step + i)
            )
            valid &= ~active | contiguous
            active &= contiguous

        return valid

    def _sample_valid_indices(
        self,
        batch_size: int,
        upper: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample uniformly from valid (time, env) pairs using batched rejection."""
        accepted_batch: list[torch.Tensor] = []
        accepted_env: list[torch.Tensor] = []
        remaining = batch_size
        attempted = 0
        max_attempts = max(1_000, batch_size * 100) * batch_size
        index_device = self.storage_device

        while remaining > 0:
            candidate_count = max(32, remaining * 2)
            env_inds = torch.randint(
                0, self.num_envs, (candidate_count,), device=index_device
            )
            batch_inds = torch.randint(
                0, upper, (candidate_count,), device=index_device
            )
            valid = self._valid_nstep_batch(batch_inds, env_inds)
            if valid.any():
                valid_batch = batch_inds[valid][:remaining]
                valid_env = env_inds[valid][:remaining]
                accepted_batch.append(valid_batch)
                accepted_env.append(valid_env)
                remaining -= valid_batch.numel()

            attempted += candidate_count
            if attempted >= max_attempts and remaining > 0:
                raise RuntimeError(
                    "Could not sample enough valid n-step windows. The buffer may "
                    "contain too few temporally contiguous transitions."
                )

        return torch.cat(accepted_batch), torch.cat(accepted_env)

    def _compute_nstep_batch(
        self,
        batch_inds: torch.Tensor,
        env_inds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute n-step returns for a batch of valid starting positions."""
        rewards = torch.zeros_like(self.rewards[batch_inds, env_inds])
        discounts = torch.ones_like(rewards)
        active = torch.ones_like(rewards, dtype=torch.bool)
        next_inds = (batch_inds + self.nstep - 1) % self.per_env_buffer_size

        for i in range(self.nstep):
            idx = (batch_inds + i) % self.per_env_buffer_size
            step_rewards = self.rewards[idx, env_inds]
            rewards += torch.where(active, discounts * step_rewards, 0.0)

            discounts = torch.where(active, discounts * self.gamma, discounts)
            terminal = active & self.dones[idx, env_inds]
            episode_end = active & self.episode_ends[idx, env_inds]
            stopped = terminal | episode_end
            discounts = torch.where(terminal, 0.0, discounts)
            next_inds = torch.where(stopped, idx, next_inds)
            active &= ~stopped

        return rewards, discounts, self.next_obs[next_inds, env_inds]

    def sample(self, batch_size: int) -> NStepReplayBufferSample:
        upper = self.per_env_buffer_size if self.full else self.pos
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
