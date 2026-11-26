"""Sampling/validity mixin for ``RecurrentReplayBuffer``.

Extends the vectorized contiguity-checking idiom already used by
``nstep_sampling.py`` (rejection sampling via torch tensor ops, no Python
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
        """True where every row the window's losses read was stored in one run,
        without a cut (``cut_running_episodes``): the rows follow each other by
        ``_step_id``, and the episode id changes only after an ``episode_end``.

        An ``episode_end`` inside the window is fine: ``episode_starts`` resets
        the hidden state after it and the n-step returns stop at it. The rows
        after the first one at or past the last learning row are read by
        nothing, so they may be missing or belong to an older lap."""
        offsets = torch.arange(self.window_len, device=self.storage_device).unsqueeze(1)
        rows = (t0.unsqueeze(0) + offsets) % self.per_env_buffer_size
        ep_id, step_id = self._ep_id[rows, env_inds], self._step_id[rows, env_inds]
        ends = self.episode_ends[rows[:-1], env_inds]
        follows = (step_id[1:] == step_id[:-1] + 1) & ((ep_id[1:] == ep_id[:-1]) | ends)
        unread = torch.zeros_like(ends)
        last_learning = self.burn_in_len + self.learning_len - 1
        unread[last_learning:] = ends[last_learning:].cumsum(dim=0) > 0
        return (ep_id[0] >= 0) & (step_id[0] >= 0) & (follows | unread).all(dim=0)

    def _sample_valid_window_starts(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Priority-proposed, rejection-sampled ``(t0, env, leaf_index,
        is_weight)`` quadruple, all checkpoint-aligned and ring-buffer-safe.

        The windows follow the priority restricted to valid windows: every valid
        one of ``batch_size`` stratified candidates is kept, and the first valid
        ones of independent candidates, checked in the same pass, fill the rest."""
        accepted_t0: list[torch.Tensor] = []
        accepted_env: list[torch.Tensor] = []
        accepted_leaf: list[torch.Tensor] = []
        remaining = batch_size
        attempted = 0
        max_attempts = max(1_000, batch_size * 100) * batch_size
        stratified_count = batch_size

        while remaining > 0:
            candidate_count = stratified_count + max(32, 2 * remaining - stratified_count)
            leaf_indices, _ = self._priority_tree.sample(
                candidate_count, generator=generator, num_stratified=stratified_count
            )
            slot = leaf_indices % self.checkpoint_capacity
            env = leaf_indices // self.checkpoint_capacity
            t0 = self._ckpt_slot_to_pos[slot, env]
            has_ckpt = t0 >= 0
            safe_t0 = t0.clamp(min=0)
            candidate_valid = has_ckpt & self._valid_window_batch(safe_t0, env)
            independent_valid = candidate_valid[stratified_count:]
            still_needed = remaining - candidate_valid[:stratified_count].sum()
            candidate_valid[stratified_count:] = independent_valid & (
                independent_valid.cumsum(dim=0) <= still_needed
            )
            stratified_count = 0

            sel = candidate_valid.nonzero(as_tuple=False).flatten()
            accepted_t0.append(t0[sel])
            accepted_env.append(env[sel])
            accepted_leaf.append(leaf_indices[sel])
            remaining -= sel.numel()

            attempted += candidate_count
            if attempted >= max_attempts and remaining > 0:
                raise RuntimeError(
                    "Could not sample enough valid recurrent windows. A window needs "
                    f"burn_in_len+learning_len+forward_len={self.window_len} consecutive "
                    f"rows per env from one of the {self.checkpoint_capacity} newest "
                    "checkpoints (one at each episode start and every "
                    f"{self.stride} steps after it); {self.size} rows per env are stored. "
                    "Use a larger learning_starts or buffer_size per env, or shorter windows."
                )

        leaf_indices = torch.cat(accepted_leaf)
        return (
            torch.cat(accepted_t0),
            torch.cat(accepted_env),
            leaf_indices,
            self._priority_tree.importance_weights(leaf_indices),
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

    def _accumulate_nstep_window(
        self, t0: torch.Tensor, env_inds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generalizes ``_accumulate_nstep`` (``nstep_sampling.py``) to
        ``learning_len`` overlapping n-step windows simultaneously -- every
        tensor op gains a leading ``learning_len`` axis. Returns
        ``(rewards, discounts, truncated_at)``, each ``(learning_len, batch)``.
        ``truncated_at`` counts rows from the first learning row: the row whose
        truncation ended the return early, so the bootstrap observation is that
        episode's final one; -1 where the return bootstraps from the row
        ``forward_len`` later, or not at all (a termination)."""
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

        truncated_at = torch.full_like(learning_starts, -1)

        for i in range(self.forward_len):
            idx = (learning_starts + i) % self.per_env_buffer_size
            step_rewards = self.rewards[idx, env_grid]
            rewards = rewards + torch.where(
                active, discounts * step_rewards, torch.zeros_like(rewards)
            )
            discounts = torch.where(active, discounts * self.gamma, discounts)
            terminal = active & self.dones[idx, env_grid]
            ended = active & self.episode_ends[idx, env_grid]
            truncated_at = torch.where(ended & ~terminal, learning_offsets + i, truncated_at)
            stopped = terminal | ended
            discounts = torch.where(terminal, torch.zeros_like(discounts), discounts)
            active = active & ~stopped

        return rewards, discounts, truncated_at

    def _truncation_bootstraps(self, t0: torch.Tensor, env_inds: torch.Tensor, truncated_at: torch.Tensor):
        """``(final_obs, final_steps, final_columns, final_index)``: the final
        observations the returns of ``truncated_at`` bootstrap from, as ``K``
        compact rows, with the row (counted from the first learning row) and
        batch column whose episode each one ends, and, per learning row, which
        of them it uses (-1: none)."""
        needed = torch.zeros(
            (self.learning_len + self.forward_len, t0.shape[0]),
            dtype=torch.bool,
            device=self.storage_device,
        )
        learning_row, column = (truncated_at >= 0).nonzero(as_tuple=True)
        needed[truncated_at[learning_row, column], column] = True
        final_steps, final_columns = needed.nonzero(as_tuple=True)
        rows = (t0[final_columns] + self.burn_in_len + final_steps) % self.per_env_buffer_size
        final_obs = self._final_obs[self._final_slot_ids[rows, env_inds[final_columns]]]

        compact = torch.full_like(needed, -1, dtype=torch.long)
        compact[final_steps, final_columns] = torch.arange(
            final_steps.numel(), device=self.storage_device
        )
        final_index = torch.where(
            truncated_at >= 0, compact.gather(0, truncated_at.clamp(min=0)), -1
        )
        return final_obs, final_steps, final_columns, final_index
