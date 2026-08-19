"""Load H5 trajectory files as ``(obs_history, action_chunk)`` windows for
diffusion BC pretraining.

Mirrors ``3rd_party/dppo/agent/dataset/sequence.py::StitchedSequenceDataset``'s
windowing exactly (verified against source): within each trajectory, valid
window starts are ``[0, traj_length - horizon_steps]`` (no padding at the
trajectory end -- windows that would run past it are simply dropped); the
``cond_steps``-length observation history is padded by *repeating the
trajectory's own first observation* for any step before the trajectory
start, not zero-padded, with the most recent observation last. Reuses
``h5_dataset.py``'s per-trajectory parsing (``_load_traj_transitions``)
directly rather than re-implementing H5 traversal -- only the windowing
step (replacing ``load_h5_dataset_to_replay_buffer``'s flattening) is new.
"""
from __future__ import annotations

from pathlib import Path

import torch

from rl_garden.buffers._dataset_common import _concat
from rl_garden.buffers._h5_common import _read_node, _require_h5py
from rl_garden.buffers.h5_dataset import _load_traj_transitions
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs


def load_h5_dataset_as_chunks(
    path: str | Path,
    *,
    horizon_steps: int,
    cond_steps: int,
    device: torch.device | str = "cpu",
    num_traj: int | None = None,
) -> tuple[Obs, torch.Tensor]:
    """Returns ``(obs_history, action_chunks)``:
    ``obs_history`` has shape ``(N, cond_steps, *obs_shape)`` (Box or Dict),
    ``action_chunks`` has shape ``(N, horizon_steps, action_dim)``.
    """
    if horizon_steps < 1:
        raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}.")
    if cond_steps < 1:
        raise ValueError(f"cond_steps must be >= 1, got {cond_steps}.")
    h5py = _require_h5py()
    path = Path(path)
    device = torch.device(device)

    obs_hist_parts: list[Obs] = []
    action_chunk_parts: list[torch.Tensor] = []
    offsets = torch.arange(cond_steps - 1, -1, -1, device=device)
    chunk_offsets = torch.arange(horizon_steps, device=device)

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
            obs, _next_obs, actions, _rewards, _dones, _episode_ends = _load_traj_transitions(
                traj, device
            )
            length = actions.shape[0]
            num_windows = length - horizon_steps + 1
            if num_windows <= 0:
                continue
            starts = torch.arange(num_windows, device=device)
            obs_idx = (starts[:, None] - offsets[None, :]).clamp(min=0)
            action_idx = starts[:, None] + chunk_offsets[None, :]
            obs_hist_parts.append(index_obs(obs, obs_idx))
            action_chunk_parts.append(actions[action_idx])

    if not action_chunk_parts:
        raise ValueError(
            f"No trajectory in {path} is long enough for horizon_steps={horizon_steps}."
        )
    return _concat(obs_hist_parts), torch.cat(action_chunk_parts, dim=0)
