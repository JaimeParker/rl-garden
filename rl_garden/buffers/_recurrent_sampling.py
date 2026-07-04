"""Sampling/validity mixin for ``RecurrentReplayBuffer``.

Extends the vectorized contiguity-checking idiom already used by
``_nstep_sampling.py`` (rejection sampling via torch tensor ops, no Python
per-sample loops) to a much longer combined burn-in+learning+forward window,
proposed by priority (a ``SumTree``) rather than drawn IID.

The host buffer must expose: ``per_env_buffer_size``, ``num_envs``,
``storage_device``, ``gamma``, ``burn_in_len``/``learning_len``/``forward_len``,
``checkpoint_capacity``, ``_ep_id``, ``_step_id``, ``_ep_relative_step``,
``episode_ends``, ``dones``, ``rewards``, ``_ckpt_slot_to_pos``,
``_final_slot_ids``, ``_final_obs``, ``_priority_tree``.
"""
from __future__ import annotations

from typing import Optional

import torch


class RecurrentSamplingMixin:
    def _valid_window_batch(
        self, t0: torch.Tensor, env_inds: torch.Tensor
    ) -> torch.Tensor:
        """``t0`` candidates must reference already-collected, ring-buffer-
        contiguous data for the full ``burn_in_len+learning_len+forward_len``
        span. Episode boundaries WITHIN the window are fine (handled by
        ``episode_starts`` masking and n-step reward zeroing) -- this only
        guards against reading stale/uninitialized data past a wraparound."""
        target_ep = self._ep_id[t0, env_inds]
        base_step = self._step_id[t0, env_inds]
        valid = (target_ep >= 0) & (base_step >= 0)
        active = valid.clone()
        window_len = self.burn_in_len + self.learning_len + self.forward_len

        for i in range(1, window_len):
            prev_idx = (t0 + i - 1) % self.per_env_buffer_size
            active = active & ~self.episode_ends[prev_idx, env_inds]
            idx = (t0 + i) % self.per_env_buffer_size
            contiguous = (self._ep_id[idx, env_inds] == target_ep) & (
                self._step_id[idx, env_inds] == base_step + i
            )
            valid = valid & (~active | contiguous)
            active = active & contiguous

        return valid

    def _sample_valid_window_starts(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Priority-proposed, rejection-sampled ``(t0, env, leaf_index,
        is_weight)`` quadruple, all checkpoint-aligned and ring-buffer-safe."""
        accepted_t0: list[torch.Tensor] = []
        accepted_env: list[torch.Tensor] = []
        accepted_leaf: list[torch.Tensor] = []
        accepted_isw: list[torch.Tensor] = []
        remaining = batch_size
        attempted = 0
        max_attempts = max(1_000, batch_size * 100) * batch_size

        while remaining > 0:
            candidate_count = max(32, remaining * 2)
            leaf_indices, is_weights = self._priority_tree.sample(
                candidate_count, generator=generator
            )
            slot = leaf_indices % self.checkpoint_capacity
            env = leaf_indices // self.checkpoint_capacity
            t0 = self._ckpt_slot_to_pos[slot, env]
            has_ckpt = t0 >= 0
            safe_t0 = t0.clamp(min=0)
            candidate_valid = has_ckpt & self._valid_window_batch(safe_t0, env)

            if candidate_valid.any():
                sel = candidate_valid.nonzero(as_tuple=False).flatten()
                sel = sel[: remaining]
                accepted_t0.append(t0[sel])
                accepted_env.append(env[sel])
                accepted_leaf.append(leaf_indices[sel])
                accepted_isw.append(is_weights[sel])
                remaining -= sel.numel()

            attempted += candidate_count
            if attempted >= max_attempts and remaining > 0:
                raise RuntimeError(
                    "Could not sample enough valid recurrent windows. The buffer "
                    "may not yet contain enough checkpoint-aligned, temporally "
                    "contiguous data for burn_in_len+learning_len+forward_len."
                )

        return (
            torch.cat(accepted_t0),
            torch.cat(accepted_env),
            torch.cat(accepted_leaf),
            torch.cat(accepted_isw),
        )

    def _gather_window(
        self, t0: torch.Tensor, env_inds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-shot advanced-indexing gather grid: ``(window_len, batch)``."""
        window_len = self.burn_in_len + self.learning_len + self.forward_len
        offsets = torch.arange(window_len, device=self.storage_device).unsqueeze(1)
        idx_grid = (t0.unsqueeze(0) + offsets) % self.per_env_buffer_size
        env_grid = env_inds.unsqueeze(0).expand(window_len, -1)
        return idx_grid, env_grid

    def _patch_final_obs(self, window_obs, idx_grid: torch.Tensor, env_grid: torch.Tensor):
        """Gymnasium autoreset means the ring buffer's naturally-following ``obs``
        after an episode-end position is already the NEXT episode's reset
        observation, not the true final one -- patch it in from the compact
        final-obs side table wherever a boundary falls inside the window (needed
        for truncation bootstrapping; harmless but also applied at true
        terminations for consistency, matching ``LazyNextNStepDictReplayBuffer``'s
        existing precedent)."""
        slot_at_pos = self._final_slot_ids[idx_grid, env_grid]
        boundary = self.episode_ends[idx_grid, env_grid] & (slot_at_pos >= 0)
        boundary = boundary[:-1]  # last window position has no "next" to patch
        if not boundary.any():
            return window_obs

        time_idx, batch_idx = torch.nonzero(boundary, as_tuple=True)
        slots = slot_at_pos[time_idx, batch_idx]
        final_values = self._final_obs[slots]
        target_time = time_idx + 1

        def _patch(tree, final_tree):
            if isinstance(tree, dict):
                return {key: _patch(value, final_tree[key]) for key, value in tree.items()}
            tree = tree.clone()
            tree[target_time, batch_idx] = final_tree
            return tree

        return _patch(window_obs, final_values)

    def _accumulate_nstep_window(
        self, t0: torch.Tensor, env_inds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalizes ``_accumulate_nstep`` (``_nstep_sampling.py``) to
        ``learning_len`` overlapping n-step windows simultaneously -- every
        tensor op gains a leading ``learning_len`` axis. Returns
        ``(rewards, discounts)``, each ``(learning_len, batch)``."""
        batch = t0.shape[0]
        learning_offsets = torch.arange(
            self.learning_len, device=self.storage_device
        ).unsqueeze(1)
        learning_starts = self.burn_in_len + t0.unsqueeze(0) + learning_offsets
        env_grid = env_inds.unsqueeze(0).expand(self.learning_len, -1)

        rewards = torch.zeros(self.learning_len, batch, device=self.storage_device)
        discounts = torch.ones(self.learning_len, batch, device=self.storage_device)
        active = torch.ones(
            self.learning_len, batch, dtype=torch.bool, device=self.storage_device
        )

        for i in range(self.forward_len):
            idx = (learning_starts + i) % self.per_env_buffer_size
            step_rewards = self.rewards[idx, env_grid]
            rewards = rewards + torch.where(
                active, discounts * step_rewards, torch.zeros_like(rewards)
            )
            discounts = torch.where(active, discounts * self.gamma, discounts)
            terminal = active & self.dones[idx, env_grid]
            stopped = terminal | (active & self.episode_ends[idx, env_grid])
            discounts = torch.where(terminal, torch.zeros_like(discounts), discounts)
            active = active & ~stopped

        return rewards, discounts
