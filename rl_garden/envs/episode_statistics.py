"""Per-env episode return/length accumulators for torch vector envs.

For backends whose envs auto-reset inside ``step()`` and report no episode
statistics themselves. Mirrors gymnasium's ``RecordEpisodeStatistics`` /
ManiSkill's ``record_metrics``: the returned values are full-width and valid
only on the rows that finished this step (the ``final_info`` masking
convention). Everything stays on the env device; no host sync.
"""
from __future__ import annotations

import torch


class EpisodeStatistics:
    """Running ``(N,)`` episode return (float32) and length (int64) per env."""

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        self._return = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._length = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        """Zero every env (a full reset), or only the rows in ``env_idx``, an
        index tensor (the ``options["env_idx"]`` partial-reset convention)."""
        if env_idx is None:
            self._return.zero_()
            self._length.zero_()
        else:
            self._return[env_idx] = 0.0
            self._length[env_idx] = 0

    def step(self, reward: torch.Tensor, done: torch.Tensor) -> dict[str, torch.Tensor]:
        """Add one step, return ``{"return", "episode_len"}`` copies (valid on
        ``done`` rows), then restart the ``done`` rows from zero."""
        self._return += reward
        self._length += 1
        episode = {"return": self._return.clone(), "episode_len": self._length.clone()}
        self._return.masked_fill_(done, 0.0)
        self._length.masked_fill_(done, 0)
        return episode
