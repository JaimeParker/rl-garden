"""Synthetic NCRL-format episodes for the NCRL tests (the real HF files use
the same keys/shapes, ``3rd_party/ncrl/src/replay_buffer.py``).

Every value is recognisable, so tests can check the converter's one-step
shift. In official row ``t`` of episode ``ep`` of embodiment ``emb``:

- ``observation`` pixels = ``obs_value(emb, ep, t)``
- ``action`` = ``ep * 100 + t`` in every real dim (row 0 = zeros)
- ``reward`` = ``t + 0.5`` (row 0 = 0)
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np


def obs_value(emb: int, ep: int, t: int) -> int:
    return emb * 128 + ep * 20 + t


def write_episode(directory: Path, emb: int, ep: int, length: int, action_dim: int, image_size: int) -> Path:
    rows = length + 1
    observation = np.empty((rows, 3, image_size, image_size), dtype=np.uint8)
    action = np.zeros((rows, action_dim), dtype=np.float32)
    reward = np.zeros((rows, 1), dtype=np.float32)
    for t in range(rows):
        observation[t] = obs_value(emb, ep, t)
        if t > 0:
            action[t] = ep * 100 + t
            reward[t] = t + 0.5
    discount = np.ones((rows, 1), dtype=np.float32)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"20230922T110643-{uuid.uuid4().hex}-{ep}-{length}.npz"
    np.savez_compressed(path, observation=observation, action=action, reward=reward, discount=discount)
    return path


def write_benchmark(
    root: Path,
    embodiments: dict[str, tuple[int, tuple[int, ...]]],
    image_size: int = 16,
) -> Path:
    """``embodiments``: ``{name: (action_dim, episode_lengths)}``, written in
    insertion order (``emb`` index = position). Returns ``root``."""
    for emb, (name, (action_dim, lengths)) in enumerate(embodiments.items()):
        for ep, length in enumerate(lengths):
            write_episode(root / name, emb, ep, length, action_dim, image_size)
    return root
