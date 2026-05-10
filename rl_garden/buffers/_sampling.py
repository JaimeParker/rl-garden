"""Sampling-strategy mixins for replay buffers.

``WithoutReplaceSamplerMixin`` adds ``sample_without_repeat`` for offline
full-batch / epoch training. It maintains a stateful permutation cursor that
is reshuffled when exhausted, when the buffer grows, or when a circular wrap
is detected (``pos`` or ``full`` changes since the last build).
"""
from __future__ import annotations

import torch

from rl_garden.common.types import ReplayBufferSample


class WithoutReplaceSamplerMixin:
    """Adds ``sample_without_repeat(batch_size)`` to a replay buffer.

    The host buffer must implement ``_index_batch(batch_inds, env_inds) -> sample``
    and expose ``per_env_buffer_size``, ``num_envs``, ``pos``, ``full``,
    ``storage_device`` (the standard ``BaseReplayBuffer`` contract).

    State is held on class-level defaults so the mixin does not need to be wired
    into the host buffer's ``__init__`` chain.
    """

    _perm: torch.Tensor | None = None
    _perm_cursor: int = 0
    _perm_pos_at_build: int = -1
    _perm_full_at_build: bool = False

    # ------------------------------------------------------------------
    # Permutation management
    # ------------------------------------------------------------------

    def _reset_permutation(self) -> None:
        upper_t = self.per_env_buffer_size if self.full else self.pos
        total = upper_t * self.num_envs
        if total == 0:
            self._perm = None
            self._perm_cursor = 0
            return
        self._perm = torch.randperm(total, device=self.storage_device)
        self._perm_cursor = 0
        self._perm_pos_at_build = self.pos
        self._perm_full_at_build = self.full

    def _perm_is_stale(self) -> bool:
        if self._perm is None:
            return True
        return (
            self._perm_pos_at_build != self.pos
            or self._perm_full_at_build != self.full
        )

    # ------------------------------------------------------------------
    # Without-replacement sample
    # ------------------------------------------------------------------

    def sample_without_repeat(self, batch_size: int) -> ReplayBufferSample:
        """Sample ``batch_size`` distinct transitions; reshuffles when exhausted."""
        if (
            self._perm_is_stale()
            or self._perm_cursor + batch_size > self._perm.shape[0]
        ):
            self._reset_permutation()
            if self._perm is None:
                raise RuntimeError(
                    "Buffer is empty; cannot sample without replacement."
                )
            if batch_size > self._perm.shape[0]:
                raise ValueError(
                    f"batch_size ({batch_size}) exceeds available transitions "
                    f"({self._perm.shape[0]})."
                )

        flat = self._perm[self._perm_cursor : self._perm_cursor + batch_size]
        self._perm_cursor += batch_size
        batch_inds = torch.div(flat, self.num_envs, rounding_mode="floor")
        env_inds = flat % self.num_envs
        return self._index_batch(batch_inds, env_inds)

    @property
    def epoch_size(self) -> int:
        """Total transitions in one epoch (full pass through the buffer)."""
        upper_t = self.per_env_buffer_size if self.full else self.pos
        return upper_t * self.num_envs
