"""Monte Carlo return computation for replay buffers.

Extends ``ReplayBuffer`` with on-the-fly MC return computation for
Cal-QL. Tracks episode boundaries and computes discounted returns when
sampling batches.

Key features:
- Mixin pattern: works with any host replay buffer sharing ``BaseReplayBuffer``'s
  ``add``/``_index_batch`` shape (currently only ``ReplayBuffer``, the
  strict Dict-observation contract)
- Episode boundary tracking via done flags
- Vectorized GPU-native MC return table (built lazily, invalidated on add())
- ~100× faster than per-sample loop on large buffers
- Optional sparse-reward MC handling: failed episodes use infinite-horizon
  approximation ``r_neg / (1 - γ)`` (mirrors the WSRL/Cal-QL reference's
  ``calc_return_to_go`` for antmaze/adroit/kitchen-style environments).
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from gymnasium import spaces

from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.common.types import MCReplayBufferSample, Obs, TensorDict


class MCReplayBufferMixin:
    """Mixin to add MC return computation to replay buffers.

    This mixin extends the base replay buffer with:
    - Episode boundary tracking via done flags
    - Lazy vectorized MC return table (cached, invalidated on add())
    - Efficient GPU-native implementation
    - Optional sparse-reward MC handling

    Usage:
        class MCReplayBuffer(MCReplayBufferMixin, ReplayBuffer):
            pass
    """

    rows_form_trajectories = True

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        gamma: float = 0.99,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
        sparse_reward_mc: bool = False,
        sparse_negative_reward: float = 0.0,
        success_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            num_envs=num_envs,
            buffer_size=buffer_size,
            storage_device=storage_device,
            sample_device=sample_device,
        )
        self.gamma = gamma
        self.sparse_reward_mc = sparse_reward_mc
        self.sparse_negative_reward = sparse_negative_reward
        self.success_threshold = success_threshold
        # Cached MC return table; invalidated whenever add() is called.
        # Shape: (per_env_buffer_size, num_envs).
        self._mc_table: torch.Tensor | None = None
        # Episode-boundary mask (termination | truncation), independent of the
        # Bellman `done` used for TD bootstrapping. MC recursion must stop here
        # even when TD bootstraps through a timeout (`done=False`).
        self._episode_end = torch.zeros(
            (self.per_env_buffer_size, num_envs),
            device=self.storage_device,
            dtype=torch.bool,
        )
        # True on the last row of a trajectory that cut_running_episodes() cut
        # off: the MC recursion stops there, but the trajectory never becomes a
        # complete one.
        self._episode_cut = torch.zeros_like(self._episode_end)
        # Cached validity mask (True where a transition's trajectory has a
        # known episode_end at or after it among currently stored data);
        # invalidated whenever add() is called.
        self._valid_table: torch.Tensor | None = None
        # Derived from _valid_table; cached separately since `.nonzero()`
        # and `.sum().item()` are themselves CUDA-synchronizing ops. Reset
        # together with _valid_table everywhere it's invalidated, so a
        # gradient step's repeated sample()/sampleable_size calls between
        # add()s pay for at most one scan+sync, not one per call.
        self._valid_indices_cache: torch.Tensor | None = None
        self._sampleable_size_cache: int | None = None
        # Per-position flag: True where this row's MC value was supplied by
        # something other than _build_mc_table()'s own recursion (offline
        # loader precompute, or a restored checkpoint) rather than derived
        # from data currently in this buffer. Persists across add()'s cache
        # invalidation -- unlike _mc_table/_valid_table, which are rebuild
        # caches, this is authoritative state that must survive until the
        # position itself is overwritten by a new add().
        self._externally_valid = torch.zeros(
            (self.per_env_buffer_size, num_envs),
            device=self.storage_device,
            dtype=torch.bool,
        )
        # The preserved MC value for _externally_valid rows (unread wherever
        # _externally_valid is False).
        self._external_mc = torch.zeros(
            (self.per_env_buffer_size, num_envs),
            device=self.storage_device,
            dtype=torch.float32,
        )
        # Per-step success flag (only allocated when sparse_reward_mc is True).
        # Stored as the same dtype as rewards for cheap arithmetic.
        if sparse_reward_mc:
            self._step_success = torch.zeros(
                (self.per_env_buffer_size, num_envs),
                device=self.storage_device,
                dtype=torch.float32,
            )
        else:
            self._step_success = None

    # ------------------------------------------------------------------
    # Checkpoint state
    # ------------------------------------------------------------------

    def _state_attrs(self) -> tuple[str, ...]:
        success = () if self._step_success is None else ("_step_success",)
        return (
            *super()._state_attrs(),
            "_episode_end",
            "_episode_cut",
            "_externally_valid",
            "_external_mc",
            *success,
        )

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "mc_table": self._mc_table}

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        super().load_state_dict(self._with_keys_absent_from_old_files(state), strict=strict)
        self._mc_table = state.get("mc_table")
        self._valid_table = None
        self._valid_indices_cache = None
        self._sampleable_size_cache = None

    def _with_keys_absent_from_old_files(self, state: dict[str, Any]) -> dict[str, Any]:
        state = dict(state)
        if state.get("episode_end") is None:
            # Matches add()'s own fallback for rows added without an episode_end.
            state["episode_end"] = state["dones"].clone().bool()
        if state.get("episode_cut") is None:
            state["episode_cut"] = torch.zeros_like(state["episode_end"])
        if state.get("externally_valid") is None or state.get("external_mc") is None:
            mc_table = state.get("mc_table")
            # An old file's whole mc_table is trusted rather than rebuilt from
            # `dones`, which can be all-False (bootstrap_at_done="always") and
            # would make every restored row look like an incomplete trajectory.
            if mc_table is not None:
                state["externally_valid"] = torch.ones_like(state["dones"], dtype=torch.bool)
                state["external_mc"] = mc_table
            else:
                state["externally_valid"] = torch.zeros_like(state["dones"], dtype=torch.bool)
                state["external_mc"] = torch.zeros_like(state["rewards"])
        if self._step_success is not None and state.get("step_success") is None:
            state["step_success"] = self._step_success
        return state

    def cut_running_episodes(self) -> None:
        """A cut trajectory has no Monte Carlo return: its rows stay unsampleable
        until overwritten. ``_externally_valid`` rows keep their own return."""
        super().cut_running_episodes()
        if len(self) == 0:
            return
        last = (self.pos - 1) % self.per_env_buffer_size
        self._episode_cut[last] = ~self._episode_end[last]
        self._mc_table = None
        self._valid_table = None
        self._valid_indices_cache = None
        self._sampleable_size_cache = None

    # ------------------------------------------------------------------
    # Cache invalidation: any add() invalidates the MC table.
    # ------------------------------------------------------------------

    def add(
        self,
        *args,
        success: Optional[torch.Tensor] = None,
        episode_end: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Add a transition. Optional ``success`` tensor records per-step success.

        ``success`` should be a (num_envs,) tensor (bool or float) marking which
        envs achieved success at this step. If not provided and
        ``sparse_reward_mc`` is enabled, success is inferred from
        ``reward >= success_threshold``.

        ``episode_end`` should be a (num_envs,) tensor marking
        ``termination | truncation`` -- the MC recursion boundary, which can
        differ from the Bellman ``done`` passed positionally to this buffer
        family's ``add()`` (e.g. Cal-QL's online rollout passes
        ``terminations | truncations`` here while ``done`` reflects
        ``bootstrap_at_done``). Consumed here (never forwarded to the wrapped
        buffer, whose ``add()`` has a fixed signature). Defaults to the
        positional ``done`` value when not given, preserving the pre-existing
        "done marks the episode boundary" behavior for callers that don't
        distinguish the two.
        """
        self._mc_table = None
        self._valid_table = None
        self._valid_indices_cache = None
        self._sampleable_size_cache = None
        # New online data overwrites this position; it is never externally
        # valid, regardless of what previously occupied this slot.
        self._externally_valid[self.pos] = False
        self._external_mc[self.pos] = 0.0
        self._episode_cut[self.pos] = False
        if episode_end is None:
            episode_end = args[4] if len(args) >= 5 else kwargs.get("done")
        self._episode_end[self.pos] = (
            episode_end.to(self.storage_device).bool()
            if episode_end is not None
            else False
        )
        if self.sparse_reward_mc and self._step_success is not None:
            if success is None:
                # Best-effort fallback: infer from current reward signal
                reward = (
                    args[3] if len(args) >= 4 else kwargs.get("reward")
                )
                if reward is None:
                    raise ValueError(
                        "sparse_reward_mc=True requires either explicit success= "
                        "or a positional reward arg in add()."
                    )
                success_tensor = (
                    reward.to(self.storage_device) >= self.success_threshold
                ).to(self._step_success.dtype)
            else:
                success_tensor = success.to(self.storage_device).to(
                    self._step_success.dtype
                )
            self._step_success[self.pos] = success_tensor
        return super().add(*args, **kwargs)

    # ------------------------------------------------------------------
    # Vectorized MC return table.
    # ------------------------------------------------------------------

    def _chronological_order(self) -> torch.Tensor:
        T = self.per_env_buffer_size
        if self.full:
            return torch.cat(
                [
                    torch.arange(self.pos, T, device=self.storage_device),
                    torch.arange(0, self.pos, device=self.storage_device),
                ]
            )
        return torch.arange(0, self.pos, device=self.storage_device)

    @staticmethod
    def _hillis_steele_affine(A: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Solve ``y_k = A_k * y_{k-1} + R_k`` (``y_{-1} = 0``) for every k in
        one pass via a Hillis-Steele parallel scan: ``log2(len)`` sequential
        vectorized doubling steps instead of one Python iteration per k.

        Never takes a reciprocal or an explicit large power -- the ``A``
        products can only shrink toward zero (safe underflow), never blow up
        -- so this is numerically safe regardless of how far apart resets
        (``A_k == 0``) are, unlike a cumulative-product-ratio formulation
        (which divides by a partial product that can hit exact zero at a
        reset, producing ``0 * inf = nan``).
        """
        P, Q = A.clone(), R.clone()
        shift = 1
        while shift < P.shape[0]:
            P_shifted = torch.ones_like(P)
            Q_shifted = torch.zeros_like(Q)
            P_shifted[shift:] = P[:-shift]
            Q_shifted[shift:] = Q[:-shift]
            Q = P * Q_shifted + Q
            P = P * P_shifted
            shift *= 2
        return Q

    def _build_mc_table(self) -> torch.Tensor:
        """Build full (T, N) MC return table via a vectorized parallel scan.

        Standard recurrence:
            G_T = r_T
            G_t = r_t + γ * G_{t+1} * (1 - episode_end_t)

        The recursion stops at ``episode_end`` (termination | truncation), not
        the Bellman ``done`` used for TD bootstrapping -- the official Cal-QL
        semantics bootstrap TD through a timeout but MC return-to-go must still
        stop at every trajectory boundary (see the JAX reference's
        ``TrajSampler``, which computes MC per finished-episode array
        regardless of the ``terminals`` flag used for TD).

        With ``sparse_reward_mc`` enabled, transitions belonging to a *failed*
        episode (no step in [t, episode_end] is marked success) are assigned
        the infinite-horizon value ``r_neg / (1 - γ)`` instead.

        Rows marked ``_externally_valid`` (offline-loader-precomputed, or
        checkpoint-restored) keep their preserved ``_external_mc`` value
        instead of being recomputed -- this buffer usually cannot reconstruct
        them correctly on its own (e.g. a dataset trajectory that the loader
        continues in the next column). A transition into or out of an externally-valid run is
        treated as a hard recursion barrier, exactly like ``episode_end``: the
        online recursion must not flow across that seam, since neither side
        is guaranteed to be the other's real trajectory continuation.

        Implemented via ``_hillis_steele_affine`` rather than a Python loop
        over every stored timestep -- the loop form cost 10-15s per rebuild
        at realistic buffer sizes (~1M transitions), all Python/CUDA-launch
        overhead, and this table is rebuilt on the first ``sample()`` after
        every rollout batch. The recursion is a linear map with per-step
        reset (``boundary_t`` zeros the carry), a textbook parallel-scan
        shape once reindexed into forward-recurrence form.
        """
        rewards = self.rewards  # (T, N)
        episode_end = self._episode_end  # (T, N), bool
        ext_valid = self._externally_valid  # (T, N), bool
        external_mc = self._external_mc  # (T, N)
        order = self._chronological_order()

        r_o = rewards[order]
        ee_o = episode_end[order]
        ext_o = ext_valid[order]
        emc_o = external_mc[order]

        # Crossing into or out of an externally-valid run is also a hard
        # recursion barrier (see docstring): boundary_o[i] additionally
        # includes the more-recent neighbor's ext flag.
        succ_ext_o = torch.zeros_like(ext_o)
        succ_ext_o[:-1] = ext_o[1:]
        boundary_o = ee_o | ext_o | succ_ext_o | self._episode_cut[order]
        succ_o = (
            self._step_success[order] > 0.5
            if self.sparse_reward_mc and self._step_success is not None
            else None
        )
        computed_o = self._returns_in_order(r_o, ~boundary_o, succ_o)

        mc_o = torch.where(ext_o, emc_o, computed_o)
        mc = torch.zeros_like(rewards)
        mc[order] = mc_o
        return mc

    def stream_mc_returns(
        self,
        rewards: torch.Tensor,
        episode_ends: torch.Tensor,
        successes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(returns, ended)`` of the rows of one stream of trajectories in
        order (e.g. a flat dataset), computed as this buffer computes its own
        rows; ``ended`` is False on the rows of a trajectory the stream does not
        end, whose return is unknown."""
        ends = episode_ends.bool().unsqueeze(1)
        succ = None
        if self.sparse_reward_mc and self._step_success is not None:
            if successes is None:
                successes = rewards >= self.success_threshold
            succ = successes.unsqueeze(1).to(self._step_success.dtype) > 0.5
        returns = self._returns_in_order(rewards.unsqueeze(1), ~ends, succ)[:, 0]
        ended = torch.flip(torch.cummax(torch.flip(ends[:, 0], dims=(0,)), dim=0).values, dims=(0,))
        return returns, ended

    def _returns_in_order(
        self, r_o: torch.Tensor, keep_o: torch.Tensor, succ_o: torch.Tensor | None
    ) -> torch.Tensor:
        """MC returns of rows in chronological order, the recursion carried from
        a row to the one before it where ``keep_o``; ``succ_o``: per-row success
        for ``sparse_reward_mc``."""
        rewards = r_o
        if succ_o is not None:
            inf_horizon_value = self.sparse_negative_reward / (1.0 - self.gamma)

            # acc_succ's own recursion never depends on the reward side (only
            # the reverse coupling exists below) -- compute it first via its
            # own boolean scan (AND/OR in place of multiply/add).
            Rb = torch.flip(succ_o, dims=(0,))
            Ab = torch.flip(keep_o, dims=(0,))
            Pb, Qb = Ab.clone(), Rb.clone()
            shift = 1
            while shift < Pb.shape[0]:
                Pb_shifted = torch.ones_like(Pb)
                Qb_shifted = torch.zeros_like(Qb)
                Pb_shifted[shift:] = Pb[:-shift]
                Qb_shifted[shift:] = Qb[:-shift]
                Qb = (Pb & Qb_shifted) | Qb
                Pb = Pb & Pb_shifted
                shift *= 2
            acc_succ_o = torch.flip(Qb, dims=(0,))

            # The reward carry actually propagated between steps is the
            # *final* per-position value (mc_t, after the acc_succ-gated
            # inf_horizon override), not a raw discounted-reward
            # accumulator -- whenever acc_succ is False at a position,
            # everything downstream of it is the constant inf_horizon_value
            # regardless of the reward-only trajectory. Fold that into the
            # affine scan's own coefficients/base terms (using the
            # already-known acc_succ) so it collapses back into one plain
            # affine scan:
            a_o = torch.where(acc_succ_o, self.gamma * keep_o.to(rewards.dtype), torch.zeros_like(r_o))
            r_prime_o = torch.where(acc_succ_o, r_o, torch.full_like(r_o, inf_horizon_value))
        else:
            a_o = self.gamma * keep_o.to(rewards.dtype)
            r_prime_o = r_o

        # Reverse to forward-recurrence form and scan.
        A = torch.flip(a_o, dims=(0,))
        R = torch.flip(r_prime_o, dims=(0,))
        return torch.flip(self._hillis_steele_affine(A, R), dims=(0,))

    def _build_validity_mask(self) -> torch.Tensor:
        """(T, N) mask: True where a transition's trajectory has reached a
        known ``episode_end`` among currently stored data (i.e. it is not
        part of a still-open trailing trajectory or of one cut off by
        ``cut_running_episodes``), or where the row is
        ``_externally_valid`` (always individually valid on its own merit,
        regardless of what surrounds it).

        Vectorized as a ``cummax`` on a flipped view (GPU-native,
        single-kernel-launch primitives) instead of a Python loop over every
        stored timestep -- the loop form cost seconds per rebuild at realistic buffer sizes,
        entirely from Python/launch overhead, not from the actual work.
        """
        order = self._chronological_order()
        # Reversed in time, the latest boundary seen decides: odd codes are ends,
        # even ones cuts.
        ee_r = torch.flip(self._episode_end[order], dims=(0,))
        boundary_r = ee_r | torch.flip(self._episode_cut[order], dims=(0,))
        age = torch.arange(1, ee_r.shape[0] + 1, device=self.storage_device).unsqueeze(1)
        code_r = torch.where(boundary_r, 2 * age + ee_r, 0)
        seen_end_o = torch.flip(torch.cummax(code_r, dim=0).values % 2 == 1, dims=(0,))
        valid_o = seen_end_o | self._externally_valid[order]
        valid = torch.zeros_like(self._episode_end)
        valid[order] = valid_o
        return valid

    def _compute_validity(self) -> torch.Tensor:
        if self._valid_table is None:
            self._valid_table = self._build_validity_mask()
        return self._valid_table

    def _valid_indices(self) -> torch.Tensor:
        """(K, 2) tensor of (t, env) index pairs eligible for sampling.

        Cached alongside ``_valid_table``: ``.nonzero()`` is itself a
        CUDA-synchronizing, data-dependent-shape op, so recomputing it on
        every call (e.g. once per gradient step under "auto" mixing) would
        reintroduce the same sync-in-the-hot-path problem this cache exists
        to avoid.
        """
        if self._valid_indices_cache is None:
            self._valid_indices_cache = self._compute_validity().nonzero(as_tuple=False)
        return self._valid_indices_cache

    @property
    def sampleable_size(self) -> int:
        """Total transitions eligible for sampling (excludes incomplete
        trailing trajectories, unlike the raw ``len(buffer)``).

        Cached alongside ``_valid_table`` for the same reason as
        ``_valid_indices()`` -- ``.item()`` forces a host sync.
        """
        if self._sampleable_size_cache is None:
            self._sampleable_size_cache = int(self._compute_validity().sum().item())
        return self._sampleable_size_cache

    def _compute_mc_returns(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> torch.Tensor:
        """Compute Monte Carlo returns for sampled transitions via cached table."""
        if self._mc_table is None:
            self._mc_table = self._build_mc_table()
        return self._mc_table[batch_inds, env_inds]

    # ------------------------------------------------------------------
    # Sample with MC returns attached.
    # ------------------------------------------------------------------

    def _index_batch(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> MCReplayBufferSample:
        """Override host buffer's ``_index_batch`` to attach MC returns."""
        base_sample = super()._index_batch(batch_inds, env_inds)
        # Index the MC table on storage_device for cache locality, then move.
        storage_batch_inds = batch_inds.to(self.storage_device)
        storage_env_inds = env_inds.to(self.storage_device)
        mc_returns = self._compute_mc_returns(storage_batch_inds, storage_env_inds)
        return MCReplayBufferSample(
            obs=base_sample.obs,
            next_obs=base_sample.next_obs,
            actions=base_sample.actions,
            rewards=base_sample.rewards,
            dones=base_sample.dones,
            mc_returns=mc_returns.to(self.sample_device),
        )

    def sample(self, batch_size: int) -> MCReplayBufferSample:
        """Sample batch with MC returns, restricted to complete trajectories."""
        valid_idx = self._valid_indices()  # (K, 2): columns (t, env)
        num_valid = valid_idx.shape[0]
        if num_valid == 0:
            raise ValueError(
                "MC replay buffer has no complete-trajectory transitions to "
                "sample from yet (every stored episode is still open)."
            )
        choice = torch.randint(0, num_valid, size=(batch_size,), device=self.storage_device)
        batch_inds = valid_idx[choice, 0]
        env_inds = valid_idx[choice, 1]
        return self._index_batch(batch_inds, env_inds)

    def _reset_permutation(self) -> None:
        """Override ``WithoutReplaceSamplerMixin``: permute only valid indices."""
        valid_idx = self._valid_indices()  # (K, 2): columns (t, env)
        num_valid = valid_idx.shape[0]
        if num_valid == 0:
            self._perm = None
            self._perm_cursor = 0
            return
        order = torch.randperm(num_valid, device=self.storage_device)
        self._perm = valid_idx[order, 0] * self.num_envs + valid_idx[order, 1]
        self._perm_cursor = 0
        self._perm_pos_at_build = self.pos
        self._perm_full_at_build = self.full


class MCReplayBuffer(MCReplayBufferMixin, ReplayBuffer):
    """ReplayBuffer with Monte Carlo return computation.

    Usage:
        buffer = MCReplayBuffer(
            observation_space=env.observation_space,
            action_space=env.action_space,
            num_envs=16,
            buffer_size=1_000_000,
            gamma=0.99,
        )
        sample = buffer.sample(256)
        # sample.mc_returns contains MC returns for Cal-QL
    """

    pass
