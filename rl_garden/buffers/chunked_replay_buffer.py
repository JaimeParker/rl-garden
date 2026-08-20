"""Action-chunked replay buffer for Box (state-only) observations.

Built for Q-chunking (QC, ``3rd_party/qc``) -- samples an H-step ("chunk")
window per training example: the full ``(horizon_length, action_dim)`` raw
action window (needed for both agents' chunked critic/actor inputs), plus a
collapsed H-step discounted return using the exact same accumulate-and-
early-stop-at-terminal recurrence as ``NStepTensorReplayBuffer``
(``nstep := horizon_length``) -- reused via ``NStepSamplingMixin`` almost
unmodified, since QC's own reward recurrence (``rewards[i] = rewards[i-1] +
r[i]*gamma**i``, ``3rd_party/qc/utils/datasets.py:120-129``) is identical to
rl-garden's existing per-step accumulation once accumulation stops at the
first true terminal.

One deliberate, documented deviation from ``3rd_party/qc/utils/datasets.py``:
QC's own ``sample_sequence`` always reads a full ``horizon_length`` window
regardless of an in-window terminal (silently summing reward from whatever
data follows the terminal in the ring buffer -- the next episode's data),
and instead discards the ENTIRE sample from the critic loss via
``valid[..., -1] = 0`` whenever the terminal falls strictly before the final
window position. This buffer instead stops reward/discount accumulation
exactly at the first true terminal (matching ``NStepTensorReplayBuffer``'s
existing, already-battle-tested convention) -- mathematically identical to
QC's target for every sample QC itself would keep (window doesn't cross a
terminal, or terminates exactly at the last position), and additionally
produces a correct, unbiased partial-length target for windows QC would
have discarded outright. Strictly more sample-efficient, never less
correct. The per-position ``valid`` mask QC needs for its actor's chunked
BC-flow loss (a genuinely different use -- masking WITHIN a kept window, not
discarding whole windows) is still faithfully reproduced.
"""
from __future__ import annotations

from typing import Optional

import torch
from gymnasium import spaces

from rl_garden.buffers._nstep_sampling import NStepSamplingMixin
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.types import ChunkedReplayBufferSample


class ChunkedTensorReplayBuffer(NStepSamplingMixin, BaseReplayBuffer):
    """H-step action-chunked replay buffer for flat Box observations.

    Storage layout: ``(per_env_buffer_size, num_envs, *shape)``, identical to
    ``NStepTensorReplayBuffer``. Internally reuses ``self.nstep`` (the field
    name ``NStepSamplingMixin`` expects) to mean ``horizon_length`` --
    window-validity and rejection-sampling logic
    (``_valid_nstep_batch``/``_sample_valid_indices``) is shared verbatim
    with the n-step buffers; only ``_accumulate_chunk``/``sample`` are new.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        horizon_length: int,
        gamma: float = 0.99,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "ChunkedTensorReplayBuffer requires a flat Box observation space."
        )
        if horizon_length < 1:
            raise ValueError(f"horizon_length must be >= 1, got {horizon_length}")

        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.per_env_buffer_size = buffer_size // num_envs
        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)
        self.horizon_length = horizon_length
        self.nstep = horizon_length  # NStepSamplingMixin's window-length field
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
        episode_end: Optional[torch.Tensor] = None,
    ) -> None:
        if self.storage_device.type == "cpu":
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()
            if episode_end is not None:
                episode_end = episode_end.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        done_bool = done.to(self.storage_device).bool()
        # Offline H5 loading (``_add_flat_transitions`` in
        # ``_dataset_common.py``) only detects episode-boundary support via
        # an MC-buffer-specific ``hasattr(buffer, "_episode_end")`` probe
        # (unrelated to this buffer's ``episode_ends`` field) and never
        # passes ``episode_end`` -- fall back to ``done``, matching
        # ``_load_traj_transitions``'s own ``episode_end = dones`` default
        # when no explicit field exists (``h5_dataset.py``).
        episode_end_bool = (
            done_bool if episode_end is None else episode_end.to(self.storage_device).bool()
        )
        self.dones[self.pos] = done_bool
        self.episode_ends[self.pos] = episode_end_bool

        self._ep_id[self.pos] = self._current_ep_id
        self._step_id[self.pos] = self._current_step_id
        self._current_ep_id += episode_end_bool.long()
        self._current_step_id += 1

        self._advance()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _accumulate_chunk(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extends ``NStepSamplingMixin._accumulate_nstep``'s single-window
        accumulation loop (window length ``self.nstep == horizon_length``)
        to also gather the raw action window and a per-position validity
        mask. ``rewards``/``discounts``/``next_inds`` are exactly what
        ``_accumulate_nstep`` would produce; ``action_chunk``/``valid`` are
        new. Returns ``(rewards, discounts, next_inds, action_chunk, valid)``
        with ``action_chunk``/``valid`` shaped ``(horizon_length, batch,
        ...)`` (transposed to batch-first by the caller).
        """
        batch = batch_inds.shape[0]
        rewards = torch.zeros(batch, device=self.storage_device)
        discounts = torch.ones(batch, device=self.storage_device)
        active = torch.ones(batch, dtype=torch.bool, device=self.storage_device)
        next_inds = (batch_inds + self.nstep - 1) % self.per_env_buffer_size

        action_chunk = torch.zeros(
            (self.nstep, batch) + self.actions.shape[2:], device=self.storage_device
        )
        valid = torch.zeros((self.nstep, batch), dtype=torch.bool, device=self.storage_device)

        for i in range(self.nstep):
            idx = (batch_inds + i) % self.per_env_buffer_size
            valid[i] = active
            action_chunk[i] = self.actions[idx, env_inds]

            step_rewards = self.rewards[idx, env_inds]
            rewards = rewards + torch.where(
                active, discounts * step_rewards, torch.zeros_like(rewards)
            )

            discounts = torch.where(active, discounts * self.gamma, discounts)
            terminal = active & self.dones[idx, env_inds]
            episode_end = active & self.episode_ends[idx, env_inds]
            stopped = terminal | episode_end
            discounts = torch.where(terminal, torch.zeros_like(discounts), discounts)
            next_inds = torch.where(stopped, idx, next_inds)
            active = active & ~stopped

        return rewards, discounts, next_inds, action_chunk, valid

    def sample(self, batch_size: int) -> ChunkedReplayBufferSample:
        upper = self.size
        if upper < self.horizon_length:
            raise RuntimeError(
                f"Buffer has only {upper} transitions per env but "
                f"horizon_length={self.horizon_length}. Wait for more data "
                "before sampling."
            )

        batch_inds, env_inds = self._sample_valid_indices(batch_size, upper)
        rewards, discounts, next_inds, action_chunk, valid = self._accumulate_chunk(
            batch_inds, env_inds
        )

        return ChunkedReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[next_inds, env_inds].to(self.sample_device),
            actions=action_chunk.transpose(0, 1).to(self.sample_device),
            rewards=rewards.to(self.sample_device),
            dones=(discounts == 0.0).to(self.sample_device),
            discounts=discounts.to(self.sample_device),
            valid=valid.transpose(0, 1).to(self.sample_device),
        )
