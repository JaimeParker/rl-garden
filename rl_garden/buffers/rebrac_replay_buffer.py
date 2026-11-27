"""Replay buffer with next-step actions, for ReBRAC's critic-side BC penalty.

See ``ReBRACReplayBufferSample``'s docstring (``rl_garden/common/types.py``)
for the exact semantics and the two documented, CORL-matching
simplifications (terminal-boundary spillover, buffer wraparound). The last
row before a cut (``cut_running_episodes``; ``add_flat_transitions`` cuts
each env's stretch of a dataset) has no stored next action and is not sampled.
"""
from __future__ import annotations

from typing import Any

import torch

from rl_garden.buffers.replay_buffer import ReplayBuffer, _tree_to_device
from rl_garden.common.types import ReBRACReplayBufferSample


class ReBRACReplayBuffer(ReplayBuffer):
    rows_form_trajectories = True
    _columns_absent_from_old_files = (
        *ReplayBuffer._columns_absent_from_old_files,
        "_next_action_unknown",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._next_action_unknown = torch.zeros(
            (self.per_env_buffer_size, self.num_envs),
            dtype=torch.bool,
            device=self.storage_device,
        )
        self._rows_with_next_action: torch.Tensor | None = None
        self._rows_version = 0
        self._perm_rows_version = -1

    def _rows_changed(self) -> None:
        self._rows_with_next_action = None
        self._rows_version += 1

    def _perm_is_stale(self) -> bool:
        return super()._perm_is_stale() or self._perm_rows_version != self._rows_version

    def _state_attrs(self) -> tuple[str, ...]:
        return (*super()._state_attrs(), "_next_action_unknown")

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        super().load_state_dict(state, strict=strict)
        self._rows_changed()
        self._perm_rows_version = self._rows_version  # a saved permutation matches its rows

    def add(self, *args: Any, **kwargs: Any) -> None:
        self._next_action_unknown[self.pos] = False
        self._rows_changed()
        super().add(*args, **kwargs)

    def cut_running_episodes(self) -> None:
        """The next row holds no action of the last row's episode, so the last
        row is not sampled (CORL drops a dataset's last row the same way)."""
        if len(self) == 0:
            return
        self._next_action_unknown[(self.pos - 1) % self.per_env_buffer_size] = True
        self._rows_changed()

    def _sampleable_rows(self) -> torch.Tensor:
        """``(K, 2)`` ``(row, env)`` pairs whose next action is stored."""
        if self._rows_with_next_action is None:
            self._rows_with_next_action = (
                ~self._next_action_unknown[: self.size]
            ).nonzero(as_tuple=False)
        return self._rows_with_next_action

    def sample(self, batch_size: int) -> ReBRACReplayBufferSample:
        rows = self._sampleable_rows()
        if rows.shape[0] == self.size * self.num_envs:
            return super().sample(batch_size)
        if rows.shape[0] == 0:
            raise ValueError(
                "ReBRAC replay buffer has no transition with a stored next action "
                "to sample from (every stored row ends a cut episode stretch)."
            )
        choice = torch.randint(0, rows.shape[0], (batch_size,), device=self.storage_device)
        return self._index_batch(rows[choice, 0], rows[choice, 1])

    def _reset_permutation(self) -> None:
        rows = self._sampleable_rows()
        self._perm_rows_version = self._rows_version
        if rows.shape[0] == self.size * self.num_envs:
            return super()._reset_permutation()
        order = torch.randperm(rows.shape[0], device=self.storage_device)
        self._perm = rows[order, 0] * self.num_envs + rows[order, 1]
        self._perm_cursor = 0
        self._perm_pos_at_build = self.pos
        self._perm_full_at_build = self.full

    @property
    def sampleable_size(self) -> int:
        return int(self._sampleable_rows().shape[0])

    @property
    def epoch_size(self) -> int:
        return self.sampleable_size

    def _index_batch(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> ReBRACReplayBufferSample:
        next_inds = (batch_inds + 1) % self.per_env_buffer_size
        next_actions = self.actions[next_inds, env_inds].to(self.sample_device)
        return ReBRACReplayBufferSample(
            obs={
                k: _tree_to_device(v, self.sample_device)
                for k, v in self.obs[batch_inds, env_inds].items()
            },
            next_obs={
                k: _tree_to_device(v, self.sample_device)
                for k, v in self.next_obs[batch_inds, env_inds].items()
            },
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
            next_actions=next_actions,
        )
