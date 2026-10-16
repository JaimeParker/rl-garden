"""Converter: official NCRL offline dataset (HF ``zhaoyi11/ncrl``) -> rl-garden's
image sequence-dataset format (read by ``rl_garden.buffers.sequence_dataset``).

Official layout: ``<root>/<benchmark>/<embodiment-or-task>/**/*.npz`` (for
example ``ncrl_datasets/dmcontrol/walker/*.npz``, or
``ncrl_datasets/metaworld/mw-assembly/0.1/default/*.npz``). One episode per
file, named ``{ts}-{uuid}-{eps_idx}-{eps_len}.npz``
(``3rd_party/ncrl/src/replay_buffer.py``). Keys:

    observation  (T+1, 3, 64, 64) uint8   row 0 = reset frame
    action       (T+1, A)         float32 row 0 = zero placeholder;
                                          action[t] led to observation[t]
    reward       (T+1, 1)         float32 reward[t] arrived with observation[t]
    discount     (T+1, 1)         float32 (not converted)

``eps_len`` in the file name is ``T``.

Conversion (per episode). The output uses rl-garden's convention, where
``action[t]`` is executed at ``obs[t]`` and ``reward[t]`` is its result.
The official arrays follow the Dreamer convention (``action[t]`` caused
``observation[t]``), so the action/reward columns shift up by one and the
final frame is dropped:

    obs[t]    = observation[t].transpose(1, 2, 0)   (CHW -> HWC), t < T
    action[t] = action[t + 1], zero-padded to padded_action_dim
    reward[t] = reward[t + 1]

Output, one directory per immediate subdirectory of ``--src_dir``
(``<dst_dir>/<embodiment>/``): ``obs.npy`` (uint8 ``(N_steps, 64, 64, 3)``,
written incrementally through ``np.lib.format.open_memmap`` so a 60 GB part
never sits in memory), ``episodes.pt`` (action/reward/episode_id/is_first)
and ``manifest.json``. ``padded_action_dim`` defaults to the largest action
dim under ``--src_dir`` (12 for dmcontrol), and ``--padded_action_dim``
overrides it.

Usage:
    python tools/conversion/convert_ncrl_dataset.py \\
        --src_dir /data/ncrl/raw/ncrl_datasets/dmcontrol --dst_dir /data/ncrl/dmcontrol_rlgarden
    python tools/conversion/convert_ncrl_dataset.py ... --embodiments walker --padded_action_dim 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_garden.buffers.sequence_dataset import (  # noqa: E402
    EPISODES_FILE,
    MANIFEST_FILE,
    OBS_FILE,
    SEQUENCE_DATASET_FORMAT,
)

OBS_KEY = "rgb_default"
SHA256_SAMPLE_SIZE = 3


def _name_fields(path: Path) -> list[str]:
    fields = path.stem.split("-")
    if len(fields) != 4:
        raise ValueError(f"{path.name}: expected '{{ts}}-{{uuid}}-{{eps_idx}}-{{eps_len}}.npz'.")
    return fields


def episode_length_from_name(path: Path) -> int:
    """``eps_len`` (``T``) from ``{ts}-{uuid}-{eps_idx}-{eps_len}.npz``."""
    return int(_name_fields(path)[3])


def _episode_order(path: Path) -> tuple[str, int]:
    """Sort key: containing directory, then ``eps_idx`` (official order)."""
    return str(path.parent), int(_name_fields(path)[2])


def embodiment_files(src_dir: Path) -> dict[str, list[Path]]:
    """``{embodiment: npz files}`` for every immediate subdirectory of
    ``src_dir`` containing npz files (recursively), ordered by directory then
    ``eps_idx``. If ``src_dir`` holds npz files directly, it is one
    embodiment named after itself."""
    if any(src_dir.glob("*.npz")):
        return {src_dir.name: sorted(src_dir.glob("*.npz"), key=_episode_order)}
    groups = {}
    for sub in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        files = sorted(sub.rglob("*.npz"), key=_episode_order)
        if files:
            groups[sub.name] = files
    if not groups:
        raise FileNotFoundError(f"No .npz files under {src_dir}")
    return groups


def read_action_dim(path: Path) -> int:
    with np.load(path) as data:
        return int(data["action"].shape[-1])


def convert_episode(
    data: dict[str, np.ndarray], padded_action_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One official episode -> ``(obs (T,H,W,3) uint8, action (T,A_pad) f32,
    reward (T,) f32)`` in rl-garden's alignment (module docstring)."""
    observation, action, reward = data["observation"], data["action"], data["reward"]
    rows = observation.shape[0]
    if rows < 2 or action.shape[0] != rows or reward.shape[0] != rows:
        raise ValueError(
            f"inconsistent episode rows: observation {observation.shape}, action {action.shape}, "
            f"reward {reward.shape}"
        )
    if observation.dtype != np.uint8 or observation.ndim != 4 or observation.shape[1] != 3:
        raise ValueError(f"expected uint8 (T+1, 3, H, W) observation, got {observation.dtype} {observation.shape}")
    if action.shape[-1] > padded_action_dim:
        raise ValueError(f"action dim {action.shape[-1]} exceeds padded_action_dim {padded_action_dim}")
    length = rows - 1
    obs = observation[:length].transpose(0, 2, 3, 1)
    padded = np.zeros((length, padded_action_dim), dtype=np.float32)
    padded[:, : action.shape[-1]] = action[1:]
    return obs, padded, reward.reshape(rows, -1)[1:, 0].astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_embodiment(
    embodiment: str, files: Sequence[Path], dst_dir: Path, padded_action_dim: int
) -> dict:
    lengths = [episode_length_from_name(f) for f in files]
    num_steps = sum(lengths)
    with np.load(files[0]) as first:
        _, channels, height, width = first["observation"].shape
    action_dim = read_action_dim(files[0])

    dst_dir.mkdir(parents=True, exist_ok=True)
    if (dst_dir / OBS_FILE).exists():
        raise FileExistsError(f"{dst_dir / OBS_FILE} already exists; remove it or pick another --dst_dir.")
    obs_out = np.lib.format.open_memmap(
        dst_dir / OBS_FILE, mode="w+", dtype=np.uint8, shape=(num_steps, height, width, channels)
    )
    action_out = torch.zeros(num_steps, padded_action_dim)
    reward_out = torch.zeros(num_steps)
    episode_id = torch.empty(num_steps, dtype=torch.int64)
    is_first = torch.zeros(num_steps, dtype=torch.bool)

    offset = 0
    for episode, (path, length) in enumerate(zip(files, lengths)):
        with np.load(path) as data:
            obs, action, reward = convert_episode(
                {k: data[k] for k in ("observation", "action", "reward")}, padded_action_dim
            )
            if data["action"].shape[-1] != action_dim:
                raise ValueError(f"{path}: action dim {data['action'].shape[-1]} != {action_dim} (first file)")
        if obs.shape[0] != length:
            raise ValueError(f"{path}: {obs.shape[0]} transitions but the file name says {length}")
        obs_out[offset : offset + length] = obs
        action_out[offset : offset + length] = torch.from_numpy(action)
        reward_out[offset : offset + length] = torch.from_numpy(reward)
        episode_id[offset : offset + length] = episode
        is_first[offset] = True
        offset += length
    obs_out.flush()
    del obs_out

    torch.save(
        {"action": action_out, "reward": reward_out, "episode_id": episode_id, "is_first": is_first},
        dst_dir / EPISODES_FILE,
    )
    manifest = {
        "format": SEQUENCE_DATASET_FORMAT,
        "embodiment": embodiment,
        "obs_key": OBS_KEY,
        "action_dim": action_dim,
        "padded_action_dim": padded_action_dim,
        "num_episodes": len(files),
        "episode_lengths": lengths,
        "image_shape": [height, width, channels],
        "source_files_sha256_sample": {f.name: _sha256(f) for f in files[:SHA256_SAMPLE_SIZE]},
    }
    (dst_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def convert_dataset(
    src_dir: Path,
    dst_dir: Path,
    *,
    embodiments: Optional[Sequence[str]] = None,
    padded_action_dim: Optional[int] = None,
) -> list[dict]:
    groups = embodiment_files(src_dir)
    max_action_dim = max(read_action_dim(files[0]) for files in groups.values())
    if padded_action_dim is None:
        padded_action_dim = max_action_dim
    elif padded_action_dim < max_action_dim:
        raise ValueError(f"--padded_action_dim {padded_action_dim} < largest action dim {max_action_dim}")
    if embodiments is not None:
        missing = sorted(set(embodiments) - set(groups))
        if missing:
            raise ValueError(f"embodiments {missing} not under {src_dir}; found {sorted(groups)}")
        groups = {name: groups[name] for name in embodiments}

    manifests = []
    for name, files in groups.items():
        start = time.time()
        manifest = convert_embodiment(name, files, dst_dir / name, padded_action_dim)
        manifests.append(manifest)
        print(
            f"[convert] {name}: {manifest['num_episodes']} episodes, "
            f"{sum(manifest['episode_lengths'])} steps, action_dim={manifest['action_dim']} "
            f"-> padded {padded_action_dim} ({time.time() - start:.0f}s)",
            flush=True,
        )
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src_dir", required=True, help="Benchmark root, e.g. .../ncrl_datasets/dmcontrol.")
    parser.add_argument("--dst_dir", required=True, help="Output root; one subdirectory per embodiment.")
    parser.add_argument("--embodiments", nargs="+", default=None, help="Convert only these subdirectories.")
    parser.add_argument(
        "--padded_action_dim", type=int, default=None, help="Default: largest action dim under --src_dir."
    )
    args = parser.parse_args()
    convert_dataset(
        Path(args.src_dir),
        Path(args.dst_dir),
        embodiments=args.embodiments,
        padded_action_dim=args.padded_action_dim,
    )


if __name__ == "__main__":
    main()
