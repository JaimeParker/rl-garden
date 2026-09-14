"""Compact final/terminal-observation side table, shared by sequence-aware
replay buffers (``RecurrentReplayBuffer``, ``TransformerReplayBuffer``).

Gymnasium autoreset means the ring buffer's naturally-following ``obs`` after an
episode-end position is already the NEXT episode's reset obs, not the true final
one -- this side table stores the true final obs compactly (a small side array,
not one slot per buffer position) so ``RecurrentSamplingMixin._patch_final_obs``
can patch it back in wherever a boundary falls inside a sampled window. Mirrors
``LazyNextNStepReplayBuffer``'s exact mechanism (``nstep_buffer.py``),
generalized to Box observations too.

The host buffer must expose: ``observation_space`` (a ``spaces.Dict``, the
rl-garden observation contract -- every current user, ``SequenceReplayBuffer``
(both ``cross_episode`` modes, model-based-base plan 1.6) and
``LazyNextNStepReplayBuffer``, is Dict-only), ``storage_device``,
``buffer_size``.
"""
from __future__ import annotations

from typing import Optional

import torch

from rl_garden.buffers.replay_buffer import DictArray


def _copy_tree(src, dst, count: int) -> None:
    if isinstance(src, DictArray):
        for key, value in src.data.items():
            _copy_tree(value, dst.data[key], count)
    else:
        dst[:count].copy_(src[:count])


class FinalObsTableMixin:
    def _init_final_obs_table(
        self, shape: tuple[int, ...], *, capacity: Optional[int] = None
    ) -> None:
        self._final_slot_ids = torch.full(
            shape, -1, dtype=torch.long, device=self.storage_device
        )
        self._free_final_slots: list[int] = []
        self._next_final_slot = 0
        if capacity is not None:
            if capacity <= 0:
                raise ValueError("final_obs_capacity must be positive")
            final_obs_capacity = capacity
        else:
            final_obs_capacity = max(1024, self.buffer_size // 64)
        self._final_obs = DictArray(
            (final_obs_capacity,), self.observation_space, device=self.storage_device
        )

    def _grow_final_obs(self) -> None:
        current = self._final_obs.shape[0]
        new_capacity = current * 2
        grown = DictArray(
            (new_capacity,), self.observation_space, device=self.storage_device
        )
        _copy_tree(self._final_obs, grown, current)
        self._final_obs = grown

    def _allocate_final_slot(self) -> int:
        if self._free_final_slots:
            return self._free_final_slots.pop()
        if self._next_final_slot >= self._final_obs.shape[0]:
            self._grow_final_obs()
        slot = self._next_final_slot
        self._next_final_slot += 1
        return slot

    def _write_final_obs_slot(self, storage, slot: int, value, env: int) -> None:
        if isinstance(storage, DictArray):
            for key in storage.data:
                self._write_final_obs_slot(storage.data[key], slot, value[key], env)
        else:
            storage[slot] = value[env].to(self.storage_device)

    def _free_final_slot(self, pos: int) -> None:
        """Return any final-obs slot owned by the position about to be overwritten."""
        old_slots = self._final_slot_ids[pos]
        for slot in old_slots[old_slots >= 0].tolist():
            self._free_final_slots.append(int(slot))
        old_slots.fill_(-1)

    def _store_final_obs_for_episode_ends(
        self, pos: int, episode_end_bool: torch.Tensor, next_obs
    ) -> None:
        """Allocate and write a final-obs slot for every env whose episode just ended."""
        for env in episode_end_bool.nonzero(as_tuple=False).flatten().tolist():
            slot = self._allocate_final_slot()
            self._final_slot_ids[pos, env] = slot
            self._write_final_obs_slot(self._final_obs, slot, next_obs, env)
