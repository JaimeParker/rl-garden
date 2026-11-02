"""``StateSequenceReplayBuffer``: the strict single-episode
``SequenceReplayBuffer`` plus per-row observation-history sampling for
imagination start states (SuperMimic).

``sample()`` is inherited unchanged (world-model windows of ``horizon + 1``
rows). ``sample_history(batch_size, history_len)`` draws start rows uniformly
from the written rows and returns the ``history_len`` observation rows ending
at each start, walking back inside the start's episode only; rows before the
episode's first step are filled with that first row (first-frame
replication), and the action taken in the row before the start (zero at an
episode's first step).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl_garden.buffers.replay_buffer import _tree_to_device
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs


@dataclass
class HistorySample:
    #: Dict of ``(L, B, ...)``; row ``L - 1`` is the start row, earlier rows
    #: are its in-episode history.
    obs: Obs
    #: ``(B, A)``: the action of the row before the start, 0 at an episode's first step.
    prev_action: torch.Tensor
    #: ``(L, B)`` bool: the row is real history (``False`` rows are copies of
    #: the episode's earliest available row).
    obs_valid: torch.Tensor


class StateSequenceReplayBuffer(SequenceReplayBuffer):
    """Strict-mode (``cross_episode=False``) only."""

    def __init__(self, *args: Any, cross_episode: bool = False, **kwargs: Any) -> None:
        if cross_episode:
            raise ValueError(f"{type(self).__name__} supports cross_episode=False only.")
        super().__init__(*args, cross_episode=False, **kwargs)

    def _history_continuity(self, t0: torch.Tensor, env_inds: torch.Tensor, depth: int) -> torch.Tensor:
        """``(depth, B)`` bool: row ``i - 1`` says whether ring row ``t0 - i``
        holds step ``base_step - i`` of the start's episode, and so do all
        rows between it and the start."""
        target_ep = self._ep_id[t0, env_inds]
        base_step = self._step_id[t0, env_inds]
        rows = []
        still_valid = torch.ones_like(t0, dtype=torch.bool)
        for i in range(1, depth + 1):
            idx = (t0 - i) % self.per_env_buffer_size
            same_episode_row = (self._ep_id[idx, env_inds] == target_ep) & (
                self._step_id[idx, env_inds] == base_step - i
            )
            still_valid = still_valid & same_episode_row
            rows.append(still_valid)
        return torch.stack(rows, dim=0)

    def sample_history(self, batch_size: int, history_len: int) -> HistorySample:
        if history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {history_len}")
        if self.size == 0:
            raise RuntimeError("sample_history() called on an empty buffer.")
        t0 = torch.randint(0, self.size, (batch_size,), device=self.storage_device)
        env_inds = torch.randint(0, self.num_envs, (batch_size,), device=self.storage_device)

        continuity = self._history_continuity(t0, env_inds, max(history_len - 1, 1))
        # Steps back (0..L-1) for output rows L-1..0; past the earliest valid
        # row every offset is clamped to it.
        num_valid_back = continuity[: history_len - 1].sum(dim=0)
        steps_back = torch.arange(history_len - 1, -1, -1, device=self.storage_device).unsqueeze(1)
        clamped_back = torch.minimum(steps_back, num_valid_back.unsqueeze(0))
        idx_grid = (t0.unsqueeze(0) - clamped_back) % self.per_env_buffer_size
        env_grid = env_inds.unsqueeze(0).expand(history_len, -1)
        obs = index_obs(self.obs, (idx_grid, env_grid))
        obs_valid = steps_back <= num_valid_back.unsqueeze(0)

        prev_idx = (t0 - 1) % self.per_env_buffer_size
        prev_action = self.actions[prev_idx, env_inds]
        prev_action = torch.where(continuity[0].unsqueeze(-1), prev_action, torch.zeros_like(prev_action))

        return HistorySample(
            obs=_tree_to_device(obs, self.sample_device),
            prev_action=prev_action.to(self.sample_device),
            obs_valid=obs_valid.to(self.sample_device),
        )
