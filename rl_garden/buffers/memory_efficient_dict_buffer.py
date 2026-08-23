"""Memory-efficient Dict replay buffer: stores one frame per timestep for
designated image keys instead of ``DictReplayBuffer``'s full duplicated
``(T, H, W, C)`` stack in both ``obs`` and ``next_obs`` (HIL-SERL's
``MemoryEfficientReplayBuffer``/``pack_obs_and_next_obs``,
``3rd_party/hil-serl/serl_launcher/serl_launcher/data/
memory_efficient_replay_buffer.py``). The temporally-stacked window is
reconstructed at sample time via a torch-native windowed gather (no NumPy
``sliding_window_view``, staying GPU-resident per this project's
no-NumPy-in-hot-paths convention).

Only meaningful paired with ``rl_garden.envs.wrappers.frame_stack
.ImageFrameStackWrapper`` upstream of it in the env wrapper stack -- ``add()``
enforces this explicitly (asserts the incoming image obs already carries a
leading ``frame_stack`` dim) rather than silently mis-shaping/mis-populating
storage if fed unstacked obs.

Design resolution: **edge-replicate at episode boundaries, do not
reject-sample them.** ``ImageFrameStackWrapper`` already edge-replicates on
``reset()`` (all ``frame_stack`` copies = the first frame), so an
early-episode transition is fully samplable in ``DictReplayBuffer`` today --
matching that is this buffer's correctness bar. The gather is clamped by a
per-position ``steps_available`` bound (how many steps have occurred since
*this episode* started, capped at ``frame_stack - 1``) -- and because
transitions are written by one strictly monotonic ring-buffer cursor, the
``steps_available`` positions immediately preceding any stored position are
always that same episode's own immediately-preceding writes, never a
different (possibly since-overwritten) episode's. A per-key ``_ep_id``
equality check on every gathered slot still guards this explicitly (cheap,
and the one real backstop against a degenerate ``per_env_buffer_size <
frame_stack`` misconfiguration, which this class does not otherwise
validate against) -- verified by test to hold across episode boundaries and
ring-buffer wraparound alike whenever ``per_env_buffer_size >= frame_stack``.
``next_obs``'s window is reconstructed as ``obs``'s window shifted by one
step (drop the oldest frame, append the recorded ``next_obs`` frame as the
new newest) rather than computed independently -- this mirrors
``ImageFrameStackWrapper.step()``'s own ``cat(frames[:, 1:], new_obs)``
exactly, and needs no separate reach/clamp computation.

Episode-boundary bookkeeping (``_ep_id``/``_step_id``, own to this class --
NOT the same semantics as ``NStepDictReplayBuffer``'s same-named attributes,
whose ``_step_id`` is a monotonic global counter rather than a
resets-per-episode one) is derived from ``done`` alone, since that is the
only boundary signal the real-world transition schema actually carries
(``ActorLoop.push_transition`` never forwards ``truncated`` -- see its
module docstring) -- consistent with, not a new gap introduced by, that
existing constraint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.buffers.dict_buffer import DictReplayBuffer, _tree_to_device
from rl_garden.buffers.mmap_storage import MmapMode
from rl_garden.common.types import GraspPenaltyReplayBufferSample, ReplayBufferSample


class MemoryEfficientDictReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        image_keys: Sequence[str],
        frame_stack: int,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
        mmap_dir: Optional[str | Path] = None,
        mmap_mode: MmapMode = "create",
        store_grasp_penalty: bool = False,
    ) -> None:
        if mmap_dir is not None:
            # Non-goal this round: this class's own _ep_id/_step_id bookkeeping
            # isn't mmap-backed, so a crash/reload would desync it from the
            # (persisted) obs/next_obs/rewards storage. Regular tensor storage
            # only, consistent with DictReplayBuffer's non-mmap default path.
            raise NotImplementedError(
                "MemoryEfficientDictReplayBuffer does not support mmap_dir yet."
            )
        self.image_keys = tuple(image_keys)
        if not self.image_keys:
            raise ValueError("MemoryEfficientDictReplayBuffer requires image_keys.")
        if frame_stack < 2:
            raise ValueError("frame_stack must be at least 2")
        self.frame_stack = int(frame_stack)

        unstacked_spaces = dict(observation_space.spaces)
        for key in self.image_keys:
            stacked_space = unstacked_spaces[key]
            if stacked_space.shape[0] != self.frame_stack:
                raise ValueError(
                    f"observation_space['{key}'].shape[0] ({stacked_space.shape[0]}) "
                    f"must equal frame_stack ({self.frame_stack})."
                )
            unstacked_spaces[key] = spaces.Box(
                low=0, high=255, shape=stacked_space.shape[1:], dtype=stacked_space.dtype
            )

        super().__init__(
            observation_space=spaces.Dict(unstacked_spaces),
            action_space=action_space,
            num_envs=num_envs,
            buffer_size=buffer_size,
            storage_device=storage_device,
            sample_device=sample_device,
            mmap_dir=None,
            mmap_mode=mmap_mode,
            store_grasp_penalty=store_grasp_penalty,
        )

        shape = (self.per_env_buffer_size, num_envs)
        self._ep_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_ep_id = torch.zeros(num_envs, dtype=torch.long, device=self.storage_device)
        self._step_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_step_id = torch.zeros(num_envs, dtype=torch.long, device=self.storage_device)

    def add(
        self,
        obs,
        next_obs,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        grasp_penalty: Optional[torch.Tensor] = None,
    ) -> None:
        for key in self.image_keys:
            shape = tuple(obs[key].shape)
            if len(shape) < 2 or shape[1] != self.frame_stack:
                raise ValueError(
                    f"MemoryEfficientDictReplayBuffer requires obs['{key}'] to arrive "
                    f"pre-stacked with a leading (N, {self.frame_stack}, ...) shape "
                    f"from ImageFrameStackWrapper -- got shape {shape}."
                )

        pos = self.pos
        done_bool = done.to(self.storage_device).bool()

        newest_obs = {k: (v[:, -1] if k in self.image_keys else v) for k, v in obs.items()}
        newest_next_obs = {
            k: (v[:, -1] if k in self.image_keys else v) for k, v in next_obs.items()
        }
        super().add(newest_obs, newest_next_obs, action, reward, done, grasp_penalty=grasp_penalty)

        self._ep_id[pos] = self._current_ep_id
        self._step_id[pos] = self._current_step_id
        self._current_ep_id = self._current_ep_id + done_bool.long()
        self._current_step_id = torch.where(
            done_bool, torch.zeros_like(self._current_step_id), self._current_step_id + 1
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _valid_batch(self, batch_inds: torch.Tensor, env_inds: torch.Tensor) -> torch.Tensor:
        """A window at (batch_inds, env_inds) is valid unless ring-buffer
        wraparound has overwritten a slot its gather would reach. next_obs's
        window is obs's window shifted by one (see ``_index_batch``), so it
        never reaches farther back than obs's own window -- one check
        covers both."""
        target_ep = self._ep_id[batch_inds, env_inds]
        step_id = self._step_id[batch_inds, env_inds]
        steps_available = step_id.clamp(max=self.frame_stack - 1)
        valid = target_ep >= 0
        for back in range(1, self.frame_stack):
            idx = (batch_inds - back) % self.per_env_buffer_size
            same_ep = self._ep_id[idx, env_inds] == target_ep
            needed = back <= steps_available
            valid = valid & (~needed | same_ep)
        return valid

    def _sample_valid_indices(
        self, batch_size: int, upper: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        accepted_batch: list[torch.Tensor] = []
        accepted_env: list[torch.Tensor] = []
        remaining = batch_size
        attempted = 0
        max_attempts = max(1_000, batch_size * 100) * batch_size
        device = self.storage_device

        while remaining > 0:
            candidate_count = max(32, remaining * 2)
            env_inds = torch.randint(0, self.num_envs, (candidate_count,), device=device)
            batch_inds = torch.randint(0, upper, (candidate_count,), device=device)
            valid = self._valid_batch(batch_inds, env_inds)
            if valid.any():
                accepted_batch.append(batch_inds[valid][:remaining])
                accepted_env.append(env_inds[valid][:remaining])
                remaining -= accepted_batch[-1].numel()
            attempted += candidate_count
            if attempted >= max_attempts and remaining > 0:
                raise RuntimeError(
                    "Could not sample enough valid windows -- the buffer may be "
                    "dominated by ring-buffer-overwritten episodes shorter than "
                    "frame_stack."
                )
        return torch.cat(accepted_batch), torch.cat(accepted_env)

    def _gather_stack(
        self,
        key: str,
        batch_inds: torch.Tensor,
        env_inds: torch.Tensor,
        steps_available: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """``num_frames`` frames ending at (inclusive of) ``batch_inds``,
        oldest first, each clamped (edge-replicated) to ``steps_available``."""
        back = torch.arange(num_frames - 1, -1, -1, device=batch_inds.device)
        clamped_back = torch.minimum(back.unsqueeze(0), steps_available.unsqueeze(1))
        idx = (batch_inds.unsqueeze(1) - clamped_back) % self.per_env_buffer_size
        storage = self.obs.data[key]
        return storage[idx, env_inds.unsqueeze(1)]

    def _index_batch(self, batch_inds: torch.Tensor, env_inds: torch.Tensor):
        step_id = self._step_id[batch_inds, env_inds]
        steps_available = step_id.clamp(max=self.frame_stack - 1)

        obs_raw = self.obs[batch_inds, env_inds]
        next_obs_raw = self.next_obs[batch_inds, env_inds]
        obs_sample = {
            k: _tree_to_device(v, self.sample_device)
            for k, v in obs_raw.items()
            if k not in self.image_keys
        }
        next_obs_sample = {
            k: _tree_to_device(v, self.sample_device)
            for k, v in next_obs_raw.items()
            if k not in self.image_keys
        }

        for key in self.image_keys:
            # next_obs's window is obs's window shifted by one step: drop
            # the oldest frame, append the recorded next_obs frame as the
            # new newest -- exactly matching ImageFrameStackWrapper's own
            # step() (cat(frames[:, 1:], new_obs)), so this reconstruction
            # needs no separate clamp/reach computation for next_obs at all.
            obs_window = self._gather_stack(
                key, batch_inds, env_inds, steps_available, self.frame_stack
            )
            next_newest = self.next_obs.data[key][batch_inds, env_inds].unsqueeze(1)
            next_window = torch.cat([obs_window[:, 1:], next_newest], dim=1)
            obs_sample[key] = obs_window.to(self.sample_device)
            next_obs_sample[key] = next_window.to(self.sample_device)

        base = dict(
            obs=obs_sample,
            next_obs=next_obs_sample,
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )
        if self.store_grasp_penalty:
            return GraspPenaltyReplayBufferSample(
                **base,
                grasp_penalty=self.grasp_penalty[batch_inds, env_inds].to(self.sample_device),
            )
        return ReplayBufferSample(**base)

    def sample(self, batch_size: int):
        upper = self.size
        batch_inds, env_inds = self._sample_valid_indices(batch_size, upper)
        return self._index_batch(batch_inds, env_inds)
