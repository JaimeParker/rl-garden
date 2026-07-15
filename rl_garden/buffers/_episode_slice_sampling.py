"""Episode-strict windowed sampling mixin, shared by ``EpisodeSliceBuffer``
(dense GPU/CPU storage) and the TD-MPC2 multitask buffer (mmap storage).

Extracted unchanged from ``EpisodeSliceBuffer`` so both hosts share the exact
same window-validity/rejection-sampling/gather logic -- see
``rl_garden/buffers/episode_slice_buffer.py``'s module docstring for why this
differs from ``RecurrentSamplingMixin`` (rejects any boundary-crossing window
outright instead of tolerating/masking it).

The host must expose: ``horizon`` (int), ``per_env_buffer_size`` (int),
``num_envs`` (int), ``storage_device`` (torch.device), ``size`` (property,
from ``BaseReplayBuffer``), ``_ep_id``/``_step_id``
(``(per_env_buffer_size, num_envs)`` long tensors, -1 for unwritten slots).
"""
from __future__ import annotations

import torch


class EpisodeSliceSamplingMixin:
    def _valid_window_batch(
        self, t0: torch.Tensor, env_inds: torch.Tensor
    ) -> torch.Tensor:
        """Reject (not mask) any candidate whose ``[t0, t0+horizon]`` span
        isn't fully contiguous within one episode -- see module docstring for
        why this differs from ``RecurrentSamplingMixin``'s tolerant version."""
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
                    "Could not sample enough valid episode-slice windows. The "
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
