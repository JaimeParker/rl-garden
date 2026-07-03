"""N-step sampling mixin shared by dict and tensor n-step replay buffers.

The host buffer must expose the standard ``BaseReplayBuffer`` ring-buffer
contract plus ``nstep``, ``gamma``, ``episode_ends``, ``_ep_id``, ``_step_id``,
and a ``_next_obs_at(inds, env_inds)`` hook for reading ``next_obs`` (dict tree
vs. flat tensor storage differ here).
"""
from __future__ import annotations

import torch


class NStepSamplingMixin:
    def _valid_nstep_batch(
        self,
        batch_inds: torch.Tensor,
        env_inds: torch.Tensor,
    ) -> torch.Tensor:
        """Return a validity mask for candidate n-step starting positions."""
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

    def _accumulate_nstep(
        self,
        batch_inds: torch.Tensor,
        env_inds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Accumulate n-step reward/discount; return (rewards, discounts, next_inds, active).

        ``active`` marks starting positions that never hit a terminal or
        episode-end within the n-step window — subclasses that reconstruct
        bootstrap observations beyond the window (e.g. lazy next_obs storage)
        need this in addition to the standard ``_next_obs_at`` lookup.
        """
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

        return rewards, discounts, next_inds, active

    def _compute_nstep_batch(
        self,
        batch_inds: torch.Tensor,
        env_inds: torch.Tensor,
    ):
        """Compute n-step returns for a batch of valid starting positions."""
        rewards, discounts, next_inds, _active = self._accumulate_nstep(
            batch_inds, env_inds
        )
        return rewards, discounts, self._next_obs_at(next_inds, env_inds)
