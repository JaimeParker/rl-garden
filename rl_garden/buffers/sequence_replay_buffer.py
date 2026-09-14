"""``SequenceReplayBuffer``: unified ``(T, N)`` windowed replay buffer,
sampling ``(window_len, B)`` windows, that both TD-MPC2's strict single-
episode windows and the R2D2-style recurrent/transformer buffers' tolerant
(boundary-masked) windows are built from -- model-based-base plan 1.6,
merging the former ``rl_garden.buffers.episode_slice_buffer`` module
(strict single-episode-window buffer + its sampling mixin, both deleted) and
the former ``_CheckpointedSequenceReplayBuffer`` scaffold (also deleted --
its storage/checkpoint/priority machinery lives here now) into one class,
selected at construction time by ``cross_episode``:

- ``cross_episode=False`` (**strict**, TD-MPC2): any candidate window whose
  ``[t0, t0 + horizon]`` span isn't fully contiguous within one episode is
  rejected outright (rejection sampling, see ``StrictWindowSamplingMixin``).
  No checkpoint grid, no priority tree.
- ``cross_episode=True`` (**tolerant**, ``RecurrentReplayBuffer``/
  ``TransformerReplayBuffer``, later Dreamer): an episode boundary *inside*
  the window is fine (masked downstream via ``episode_starts``/n-step reward
  zeroing, ``RecurrentSamplingMixin``); only a genuine ring-buffer-wraparound
  read is rejected. Windows are proposed by a priority sum-tree over a sparse
  ``stride``-periodic checkpoint grid.

``priority`` is not a constructor choice: it is fully determined by
``cross_episode`` (``self.priority = cross_episode``) -- every tolerant-mode
consumer today (``RecurrentReplayBuffer``/``TransformerReplayBuffer``) is
priority-tree-driven, and strict mode has no priority tree at all. A
non-priority *tolerant* mode (uniform sampling over cross-episode windows)
is unimplemented -- deferred to Part 2 (DreamerV3), which does not use PER
and will need to add that combination back (this class does not currently
expose a way to construct it).

``RecurrentReplayBuffer``/``TransformerReplayBuffer`` stay thin subclasses of
this class for the tolerant path (their public constructor args and
``sample()`` outputs are unchanged, see those modules); TD-MPC2's online
buffer is this class used **directly** with ``cross_episode=False`` (no
subclass -- ``StrictWindowSamplingMixin`` gives it everything ``add()``/
``sample()`` need, plus the tail-step fix below).

**Tail-step fix** (model-based-base plan 1.6, scratchpad
``tdmpc2-upstream-vs-port.md`` section 3): a strict-mode ring buffer relies
on gymnasium vector-env autoreset to write the *next* episode's reset
observation into the storage slot immediately after an episode's last
transition -- so, before this fix, any window needing that slot (i.e. one
ending exactly at an episode's true final transition) was unconditionally
rejected, systematically under-sampling the last ``horizon`` steps of every
episode relative to the interior (upstream's per-episode-block storage has
no such loss). This class fixes it the same way ``FinalObsTableMixin``
already fixes the identical aliasing problem for the tolerant path: every
``add()`` call stores the true post-action observation into a compact
final-obs side table keyed by episode-end position (regardless of
``cross_episode``), and strict-mode's extended ``_valid_window_batch``
additionally accepts (instead of rejecting) a window whose last position
lands exactly on a recorded episode end, patching that position's
observation in from the side table (``_patch_tail_final_obs``) rather than
reading the ring buffer's aliased next-episode slot. ``StrictWindowSamplingMixin``
itself is NOT changed to do this -- it stays the original, side-table-free
rejection logic, still used verbatim by ``MmapMultitaskEpisodeBuffer``
(offline, pre-loaded whole episodes; no online autoreset aliasing exists
there in the first place, so the fix does not apply and is out of scope for
that host, see that module's docstring).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from gymnasium import spaces

from rl_garden.buffers._final_obs_table import FinalObsTableMixin
from rl_garden.buffers._recurrent_sampling import RecurrentSamplingMixin
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.replay_buffer import DictArray, _tree_to_device
from rl_garden.buffers.sum_tree import SumTree
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs


class StrictWindowSamplingMixin:
    """Episode-strict windowed sampling: reject (not mask) any candidate
    whose ``[t0, t0 + horizon]`` span isn't fully contiguous within one
    episode. Moved unchanged from the former episode-slice sampling module
    (deleted, model-based-base plan 1.6) -- shared verbatim by ``SequenceReplayBuffer``
    (``cross_episode=False``, which layers an additional tail-step
    acceptance path on top via its own ``_valid_window_batch`` override, see
    module docstring) and ``MmapMultitaskEpisodeBuffer``.

    The host must expose: ``horizon`` (int), ``per_env_buffer_size`` (int),
    ``num_envs`` (int), ``storage_device`` (torch.device), ``size``
    (property, from ``BaseReplayBuffer``), ``_ep_id``/``_step_id``
    (``(per_env_buffer_size, num_envs)`` long tensors, -1 for unwritten
    slots).
    """

    def _valid_window_batch(
        self, t0: torch.Tensor, env_inds: torch.Tensor
    ) -> torch.Tensor:
        target_ep = self._ep_id[t0, env_inds]
        base_step = self._step_id[t0, env_inds]
        valid = (target_ep >= 0) & (base_step >= 0)
        for i in range(1, self.horizon + 1):
            idx = (t0 + i) % self.per_env_buffer_size
            contiguous = (self._ep_id[idx, env_inds] == target_ep) & (
                self._step_id[idx, env_inds] == base_step + i
            )
            valid = valid & contiguous
        return valid

    def _sample_valid_window_starts(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        accepted_t0: list[torch.Tensor] = []
        accepted_env: list[torch.Tensor] = []
        remaining = batch_size
        attempted = 0
        max_attempts = max(1_000, batch_size * 100) * batch_size
        upper = self.size

        while remaining > 0:
            candidate_count = max(32, remaining * 2)
            env_inds = torch.randint(
                0, self.num_envs, (candidate_count,), device=self.storage_device
            )
            t0 = torch.randint(0, upper, (candidate_count,), device=self.storage_device)
            valid = self._valid_window_batch(t0, env_inds)
            if valid.any():
                sel_t0 = t0[valid][:remaining]
                sel_env = env_inds[valid][:remaining]
                accepted_t0.append(sel_t0)
                accepted_env.append(sel_env)
                remaining -= sel_t0.numel()

            attempted += candidate_count
            if attempted >= max_attempts and remaining > 0:
                raise RuntimeError(
                    "Could not sample enough valid strict-window starts. The "
                    "buffer may not yet contain enough episodes at least "
                    f"horizon+1={self.horizon + 1} steps long."
                )

        return torch.cat(accepted_t0), torch.cat(accepted_env)

    def _gather_window(
        self, t0: torch.Tensor, env_inds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        window_len = self.horizon + 1
        offsets = torch.arange(window_len, device=self.storage_device).unsqueeze(1)
        idx_grid = (t0.unsqueeze(0) + offsets) % self.per_env_buffer_size
        env_grid = env_inds.unsqueeze(0).expand(window_len, -1)
        return idx_grid, env_grid


@dataclass
class SequenceReplayBufferSample:
    """``cross_episode=False`` (strict) sample -- mirrors the former
    strict-mode sample dataclass (deleted)."""

    obs: Obs                    # (horizon+1, B, *obs_shape)
    action: torch.Tensor        # (horizon, B, act_dim)
    reward: torch.Tensor        # (horizon, B)
    terminated: torch.Tensor    # (horizon, B), bool


class SequenceReplayBuffer(RecurrentSamplingMixin, FinalObsTableMixin, BaseReplayBuffer):
    """See module docstring. ``cross_episode`` selects the storage/sampling
    mode (``priority`` is derived from it, not a constructor argument);
    every other constructor argument is either strict-mode-only
    (``horizon``) or tolerant-mode-only (``burn_in_len``, ``learning_len``,
    ``forward_len``, ``stride``, ``gamma``, ``prio_exponent``,
    ``importance_sampling_exponent``, ``priority_eps``).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        *,
        cross_episode: bool = False,
        # cross_episode=False (strict) only:
        horizon: Optional[int] = None,
        # cross_episode=True (tolerant) only:
        burn_in_len: Optional[int] = None,
        learning_len: Optional[int] = None,
        forward_len: Optional[int] = None,
        stride: Optional[int] = None,
        gamma: float = 0.99,
        prio_exponent: float = 0.9,
        importance_sampling_exponent: float = 0.6,
        priority_eps: float = 1e-3,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                f"{type(self).__name__} requires a Dict observation space (the "
                f"rl-garden observation contract), got {type(observation_space)}."
            )
        self.cross_episode = cross_episode
        self.priority = cross_episode

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.per_env_buffer_size = buffer_size // num_envs
        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)
        self.pos = 0
        self.full = False

        shape = (self.per_env_buffer_size, num_envs)
        self.obs = DictArray(shape, observation_space, device=self.storage_device)
        self.actions = torch.zeros(
            shape + tuple(action_space.shape), dtype=torch.float32, device=self.storage_device
        )
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=self.storage_device)
        self.dones = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)
        self.episode_ends = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)

        # Episode-contiguity bookkeeping, shared by both modes.
        self._ep_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_ep_id = torch.zeros(num_envs, dtype=torch.long, device=self.storage_device)
        self._step_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_step_id = torch.zeros(num_envs, dtype=torch.long, device=self.storage_device)

        # Steps since episode start (resets to 0 the step after an episode
        # ends). Written by _add_common for both modes (strict mode never
        # reads it back -- only tolerant mode's checkpoint eligibility and
        # episode_starts computation do).
        self._ep_relative_step = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_ep_relative_step = torch.zeros(
            num_envs, dtype=torch.long, device=self.storage_device
        )

        # Final-obs side table (both modes -- strict mode's tail-step fix,
        # tolerant mode's pre-existing truncation-bootstrap patch, see module
        # docstring).
        self._init_final_obs_table(shape)

        if cross_episode:
            if horizon is not None:
                raise ValueError(
                    "horizon is a cross_episode=False (strict) parameter; "
                    "cross_episode=True uses burn_in_len/learning_len/forward_len."
                )
            if None in (burn_in_len, learning_len, forward_len, stride):
                raise ValueError(
                    "cross_episode=True requires burn_in_len/learning_len/"
                    "forward_len/stride."
                )
            if burn_in_len < 1:
                raise ValueError(f"burn_in_len must be >= 1, got {burn_in_len}")
            if learning_len < 1:
                raise ValueError(f"learning_len must be >= 1, got {learning_len}")
            if forward_len < 1:
                raise ValueError(f"forward_len must be >= 1, got {forward_len}")
            self.burn_in_len = burn_in_len
            self.learning_len = learning_len
            self.forward_len = forward_len
            self.stride = stride
            self.window_len = burn_in_len + learning_len + forward_len
            if self.per_env_buffer_size % self.stride != 0:
                raise ValueError(
                    f"per_env_buffer_size ({self.per_env_buffer_size}) must be "
                    f"divisible by the checkpoint stride ({self.stride})."
                )
            self.gamma = gamma

            # Compact checkpoint side-buffer (stride-x smaller than
            # per-timestep storage).
            self.checkpoint_capacity = self.per_env_buffer_size // self.stride
            self._current_ckpt_pos = torch.zeros(
                num_envs, dtype=torch.long, device=self.storage_device
            )
            self._ckpt_slot_to_pos = torch.full(
                (self.checkpoint_capacity, num_envs),
                -1,
                dtype=torch.long,
                device=self.storage_device,
            )
            self._init_checkpoint_extra_storage()

            capacity_total = self.checkpoint_capacity * num_envs
            self._priority_tree = SumTree(
                capacity=capacity_total,
                alpha=prio_exponent,
                beta=importance_sampling_exponent,
                device=self.storage_device,
                eps=priority_eps,
            )
        else:
            if any(p is not None for p in (burn_in_len, learning_len, forward_len, stride)):
                raise ValueError(
                    "burn_in_len/learning_len/forward_len/stride are "
                    "cross_episode=True parameters; cross_episode=False uses horizon."
                )
            if horizon is None:
                raise ValueError("cross_episode=False (strict) requires horizon.")
            if horizon < 1:
                raise ValueError(f"horizon must be >= 1, got {horizon}")
            if self.per_env_buffer_size <= horizon:
                raise ValueError(
                    "per_env_buffer_size "
                    f"({self.per_env_buffer_size}) must be > horizon ({horizon})."
                )
            self.horizon = horizon
            self.window_len = horizon + 1

    def _init_checkpoint_extra_storage(self) -> None:
        """No-op default. ``RecurrentReplayBuffer`` overrides to allocate RNN
        hidden-state checkpoint tensors."""
        return

    def _write_checkpoint_extra(
        self, slots: torch.Tensor, envs: torch.Tensor, hidden=None
    ) -> None:
        """No-op default. ``RecurrentReplayBuffer`` overrides to write hidden
        state for the given checkpoint slots."""
        return

    # ------------------------------------------------------------------
    # Strict-vs-tolerant dispatch for the three RNG-critical sampling
    # methods (see module docstring on why StrictWindowSamplingMixin is not
    # inherited directly -- its acceptance rule is extended here, not
    # replaced -- and RecurrentSamplingMixin.'s super() call).
    # ------------------------------------------------------------------

    def _valid_window_batch(self, t0: torch.Tensor, env_inds: torch.Tensor) -> torch.Tensor:
        if self.cross_episode:
            return super()._valid_window_batch(t0, env_inds)

        valid = StrictWindowSamplingMixin._valid_window_batch(self, t0, env_inds)

        # Tail-step fix (module docstring): a window ending exactly at a
        # recorded episode end is valid too -- the mixin's strict check
        # above rejects it because the ring buffer's slot right after an
        # episode end aliases the next episode's reset obs; that position
        # gets patched from the final-obs side table (_patch_tail_final_obs)
        # instead when this OR-in clause accepts the window. The clause
        # checks only the *second-to-last* position's contiguity (rather
        # than replaying the whole per-step loop): _step_id is a per-env
        # counter that increments by exactly 1 on every add() and never
        # resets, so step_id[p] === p (mod per_env_buffer_size) for
        # whichever lap last wrote p. Matching (ep_id, step_id) at both t0
        # and t0+horizon-1 (a span < per_env_buffer_size, enforced in
        # __init__) therefore pins every slot in between to that same lap
        # too -- there is no way for an intermediate slot to hold a
        # different lap's data while both endpoints agree.
        target_ep = self._ep_id[t0, env_inds]
        base_step = self._step_id[t0, env_inds]
        prev_idx = (t0 + self.horizon - 1) % self.per_env_buffer_size
        interior_contiguous = (self._ep_id[prev_idx, env_inds] == target_ep) & (
            self._step_id[prev_idx, env_inds] == base_step + self.horizon - 1
        )
        has_final_obs = self.episode_ends[prev_idx, env_inds] & (
            self._final_slot_ids[prev_idx, env_inds] >= 0
        )
        return valid | (interior_contiguous & has_final_obs)

    def _sample_valid_window_starts(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ):
        if self.cross_episode:
            return super()._sample_valid_window_starts(batch_size, generator=generator)
        return StrictWindowSamplingMixin._sample_valid_window_starts(self, batch_size)

    def _gather_window(self, t0: torch.Tensor, env_inds: torch.Tensor):
        if self.cross_episode:
            return super()._gather_window(t0, env_inds)
        return StrictWindowSamplingMixin._gather_window(self, t0, env_inds)

    def _patch_tail_final_obs(self, window_obs, idx_grid: torch.Tensor, env_grid: torch.Tensor):
        """Strict-mode counterpart of ``RecurrentSamplingMixin._patch_final_obs``,
        restricted to the window's last row -- the only position strict
        mode's rejection rule allows to coincide with an episode boundary
        (see module docstring)."""
        last_row = idx_grid.shape[0] - 1
        prev_idx, last_env = idx_grid[last_row - 1], env_grid[last_row]
        boundary = self.episode_ends[prev_idx, last_env] & (
            self._final_slot_ids[prev_idx, last_env] >= 0
        )
        if not boundary.any():
            return window_obs

        batch_idx = boundary.nonzero(as_tuple=False).flatten()
        slots = self._final_slot_ids[prev_idx, last_env][batch_idx]
        final_values = self._final_obs[slots]

        def _patch(tree, final_tree):
            if isinstance(tree, dict):
                return {key: _patch(value, final_tree[key]) for key, value in tree.items()}
            tree = tree.clone()
            tree[last_row, batch_idx] = final_tree
            return tree

        return _patch(window_obs, final_values)

    # ------------------------------------------------------------------
    # Storage (_add_common is the shared write body for both modes' add();
    # strict mode's own add() below is the sole entry point when this class
    # is used directly).
    # ------------------------------------------------------------------

    def _add_common(
        self,
        obs,
        next_obs,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        episode_end: torch.Tensor,
    ) -> torch.Tensor:
        done_bool = done.to(self.storage_device).bool()
        episode_end_bool = episode_end.to(self.storage_device).bool().reshape(self.num_envs)

        assert isinstance(obs, dict)
        self.obs[self.pos] = {k: v.to(self.storage_device) for k, v in obs.items()}
        self.actions[self.pos] = action.to(self.storage_device)
        self.rewards[self.pos] = reward.reshape(self.num_envs).to(self.storage_device)
        self.dones[self.pos] = done_bool.reshape(self.num_envs)
        self.episode_ends[self.pos] = episode_end_bool

        self._free_final_slot(self.pos)
        self._store_final_obs_for_episode_ends(self.pos, episode_end_bool, next_obs)

        self._ep_id[self.pos] = self._current_ep_id
        self._step_id[self.pos] = self._current_step_id
        self._ep_relative_step[self.pos] = self._current_ep_relative_step

        return episode_end_bool

    def _write_checkpoint_slots(
        self, episode_end_bool: torch.Tensor, hidden=None
    ) -> None:
        is_checkpoint = self._current_ep_relative_step % self.stride == 0
        if not is_checkpoint.any():
            return
        envs = is_checkpoint.nonzero(as_tuple=False).flatten()
        slots = self._current_ckpt_pos[envs] % self.checkpoint_capacity
        self._write_checkpoint_extra(slots, envs, hidden=hidden)
        self._ckpt_slot_to_pos[slots, envs] = self.pos
        leaf_indices = envs * self.checkpoint_capacity + slots
        self._priority_tree.set_uninitialized(leaf_indices)
        self._current_ckpt_pos[envs] += 1

    def add(
        self,
        obs: Obs,
        next_obs: Obs,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        episode_end: torch.Tensor,
    ) -> None:
        """Strict-mode (``cross_episode=False``) storage -- the only mode
        this class implements ``add()`` for directly; tolerant-mode
        subclasses (``RecurrentReplayBuffer``/``TransformerReplayBuffer``)
        override ``add()`` themselves (their checkpoint payload, e.g. RNN
        hidden state, differs per subclass) using ``_add_common``/
        ``_write_checkpoint_slots`` above."""
        if self.cross_episode:
            raise NotImplementedError(
                f"{type(self).__name__}(cross_episode=True) has no direct add() -- "
                "subclass it and override add() (see RecurrentReplayBuffer/"
                "TransformerReplayBuffer)."
            )
        episode_end_bool = self._add_common(obs, next_obs, action, reward, done, episode_end)

        self._current_ep_id = self._current_ep_id + episode_end_bool.long()
        self._current_step_id = self._current_step_id + 1

        self._advance()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_common(self, batch_size: int, generator: Optional[torch.Generator] = None):
        """Tolerant-mode sample body, unchanged from the former
        ``_CheckpointedSequenceReplayBuffer._sample_common`` -- used by
        ``RecurrentReplayBuffer``/``TransformerReplayBuffer``'s own
        ``sample()``."""
        t0, env_inds, leaf_indices, is_weights = self._sample_valid_window_starts(
            batch_size, generator=generator
        )
        idx_grid, env_grid = self._gather_window(t0, env_inds)

        window_obs = index_obs(self.obs, (idx_grid, env_grid))
        window_obs = self._patch_final_obs(window_obs, idx_grid, env_grid)

        learn_slice = slice(self.burn_in_len, self.burn_in_len + self.learning_len)
        actions = self.actions[idx_grid[learn_slice], env_grid[learn_slice]]

        ep_rel = self._ep_relative_step[idx_grid, env_grid]
        episode_starts = (ep_rel == 0).float()

        rewards, discounts = self._accumulate_nstep_window(t0, env_inds)

        return (
            env_inds,
            leaf_indices,
            is_weights,
            window_obs,
            actions,
            episode_starts,
            rewards,
            discounts,
        )

    def sample(self, batch_size: int) -> SequenceReplayBufferSample:
        """Strict-mode (``cross_episode=False``) sampling -- the only mode
        this class implements ``sample()`` for directly."""
        if self.cross_episode:
            raise NotImplementedError(
                f"{type(self).__name__}(cross_episode=True) has no direct sample() "
                "-- subclass it and override sample() (see RecurrentReplayBuffer/"
                "TransformerReplayBuffer, which call self._sample_common())."
            )
        t0, env_inds = self._sample_valid_window_starts(batch_size)
        idx_grid, env_grid = self._gather_window(t0, env_inds)

        window_obs = index_obs(self.obs, (idx_grid, env_grid))
        window_obs = self._patch_tail_final_obs(window_obs, idx_grid, env_grid)

        action = self.actions[idx_grid[:-1], env_grid[:-1]]
        reward = self.rewards[idx_grid[:-1], env_grid[:-1]]
        terminated = self.dones[idx_grid[:-1], env_grid[:-1]]

        obs = _tree_to_device(window_obs, self.sample_device)

        return SequenceReplayBufferSample(
            obs=obs,
            action=action.to(self.sample_device),
            reward=reward.to(self.sample_device),
            terminated=terminated.to(self.sample_device),
        )

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        self._priority_tree.update(
            indices.to(self.storage_device), td_errors.to(self.storage_device)
        )
