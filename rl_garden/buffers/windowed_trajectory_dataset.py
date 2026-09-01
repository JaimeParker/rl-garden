"""Fixed-length ``(obs_window, action_window)`` sampler over an offline H5
trajectory dataset, both windows sharing the same start index -- unlike
``chunked_dataset.py::load_h5_dataset_as_chunks``, which pairs an asymmetric
obs-*history* window against a *future* action-chunk window (built for
diffusion-BC conditioning). OPAL needs ``obs[i:i+H]`` aligned with
``action[i:i+H]``, so this is a new, generically-named sibling rather than
an extension of that loader.

A stateful, resample-every-call sampler (mirrors ``HindsightGoalDataset``'s
shape) ported from SUPE's ``ChunkDataset`` windowing
(``SUPE/supe/data/chunk_dataset.py:58-93,163-174``): valid start
indices -- those where the ``chunk_size``-length window doesn't cross a
trajectory boundary -- are precomputed once, then sampled fresh via
``torch.randint`` every ``.sample()`` call (rl-garden's established
substitute for upstream's ``np.random.choice(..., replace=False)``, the same
tradeoff already accepted in ``HindsightGoalDataset``). ``.all_windows()``
additionally exposes a deterministic, exactly-once-per-window full pass
(mirrors ``ChunkDataset.create``'s one-time relabeling pass, used by SUPE's
skill-relabeled offline buffer), separate from ``.sample()``'s random draws.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from rl_garden.buffers._dataset_common import _concat
from rl_garden.buffers._h5_common import _read_node, _require_h5py
from rl_garden.buffers.h5_dataset import _load_traj_transitions
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs


def _terminated_only(traj: dict[str, Any], length: int, device: torch.device) -> Optional[torch.Tensor]:
    """True-termination-only signal, distinct from truncation -- returns
    ``None`` if the trajectory doesn't separately record it (caller falls
    back to the combined done signal). Mirrors upstream's ``masks =
    1 - terminals`` (``SUPE/supe/data/d4rl_datasets.py:44``), used for
    reward-zeroing (``ChunkDataset.create``'s ``masks`` field) as distinct
    from its ``dones`` output field (``traj_end OR terminals``,
    ``d4rl_datasets.py:40``, used only for the aggregated episode-boundary
    flag) -- see ``WindowedTrajectorySample``'s ``terminated_window`` vs.
    ``done_window`` docstring.
    """
    for key in ("terminated", "terminations"):
        if key in traj:
            return torch.as_tensor(traj[key][:length], device=device).bool().float()
    return None


@dataclass
class WindowedTrajectorySample:
    obs_window: Obs  # (B, chunk_size, *obs_shape)
    action_window: torch.Tensor  # (B, chunk_size, action_dim)
    next_obs: Obs  # (B, *obs_shape) -- the window's next_obs after its last step
    reward_window: torch.Tensor  # (B, chunk_size)
    done_window: torch.Tensor  # (B, chunk_size) -- combined boundary (terminated OR
    # truncated), matching upstream's aggregated `dones` output field.
    terminated_window: torch.Tensor  # (B, chunk_size) -- true termination only
    # (excludes truncation), matching upstream's `masks` field used for
    # reward-zeroing; falls back to done_window when the source H5 doesn't
    # separately record termination vs. truncation.


class WindowedTrajectoryDataset:
    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int,
        device: torch.device | str = "cpu",
        num_traj: Optional[int] = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        self.chunk_size = chunk_size
        self.device = torch.device(device)

        h5py = _require_h5py()
        path = Path(path)
        obs_parts: list[Obs] = []
        next_obs_parts: list[Obs] = []
        action_parts: list[torch.Tensor] = []
        reward_parts: list[torch.Tensor] = []
        done_parts: list[torch.Tensor] = []
        terminated_parts: list[torch.Tensor] = []
        valid_start_parts: list[torch.Tensor] = []
        running_total = 0

        with h5py.File(path, "r") as f:
            keys = sorted(
                (key for key in f if key.startswith("traj_")),
                key=lambda key: int(key.split("_")[-1]),
            )
            if num_traj is not None:
                keys = keys[:num_traj]
            for key in keys:
                traj = _read_node(f[key])
                if not isinstance(traj, dict):
                    continue
                obs, next_obs, actions, rewards, dones, _episode_ends = (
                    _load_traj_transitions(traj, self.device)
                )
                length = actions.shape[0]
                num_windows = length - chunk_size + 1
                if num_windows <= 0:
                    continue
                terminated = _terminated_only(traj, length, self.device)
                if terminated is None:
                    terminated = dones
                obs_parts.append(obs)
                next_obs_parts.append(next_obs)
                action_parts.append(actions)
                reward_parts.append(rewards)
                done_parts.append(dones)
                terminated_parts.append(terminated)
                valid_start_parts.append(
                    running_total + torch.arange(num_windows, device=self.device)
                )
                running_total += length

        if not action_parts:
            raise ValueError(
                f"No trajectory in {path} is long enough for chunk_size={chunk_size}."
            )

        self._obs = _concat(obs_parts)
        self._next_obs = _concat(next_obs_parts)
        self._actions = torch.cat(action_parts, dim=0)
        self._rewards = torch.cat(reward_parts, dim=0)
        self._dones = torch.cat(done_parts, dim=0)
        self._terminated = torch.cat(terminated_parts, dim=0)
        self.valid_starts = torch.cat(valid_start_parts, dim=0)

    def _gather(self, starts: torch.Tensor) -> WindowedTrajectorySample:
        window_idx = starts[:, None] + torch.arange(self.chunk_size, device=self.device)[None, :]
        last_idx = starts + (self.chunk_size - 1)
        return WindowedTrajectorySample(
            obs_window=index_obs(self._obs, window_idx),
            action_window=self._actions[window_idx],
            next_obs=index_obs(self._next_obs, last_idx),
            reward_window=self._rewards[window_idx],
            done_window=self._dones[window_idx],
            terminated_window=self._terminated[window_idx],
        )

    def sample(self, batch_size: int) -> WindowedTrajectorySample:
        idx = torch.randint(0, self.valid_starts.shape[0], (batch_size,), device=self.device)
        return self._gather(self.valid_starts[idx])

    def all_windows(self, batch_size: int) -> Iterator[WindowedTrajectorySample]:
        """Yield every valid window exactly once, in order, in batches of
        ``batch_size`` (the last batch may be smaller) -- a deterministic
        full-dataset pass, unlike ``.sample()``'s per-call random draws."""
        n = self.valid_starts.shape[0]
        for start in range(0, n, batch_size):
            yield self._gather(self.valid_starts[start : start + batch_size])
