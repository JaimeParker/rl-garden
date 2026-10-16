"""Reader for rl-garden's on-disk image sequence-dataset format (written by
``tools/conversion/convert_ncrl_dataset.py``) and its two sampling paths.

On-disk layout. ``dataset_dir`` is either one part directory or a root whose
immediate subdirectories are part directories (one per embodiment). Each
part directory holds:

- ``obs.npy``: ``uint8 (N_steps, H, W, 3)``, a plain ``.npy`` opened with
  ``np.load(mmap_mode="r")``, so a 60+ GB part is never read into memory.
- ``episodes.pt``: ``{"action": f32 (N_steps, A_pad), "reward": f32
  (N_steps,), "episode_id": int64 (N_steps,), "is_first": bool (N_steps,)}``,
  loaded with ``torch.load(mmap=True)``.
- ``manifest.json``: ``format``, ``embodiment``, ``obs_key``, ``action_dim``,
  ``padded_action_dim``, ``num_episodes``, ``episode_lengths``,
  ``image_shape``, ``source_files_sha256_sample``.

Episodes are stored back to back. Row alignment follows the rest of
rl-garden: ``action[t]`` is executed at ``obs[t]`` and ``reward[t]`` is its
result. The final observation of an episode is not stored.

The two sampling paths:

- :func:`load_sequence_dataset` copies the dataset into a uniform-mode
  ``SequenceReplayBuffer`` (``cross_episode=True, priority=False``). This is
  for data that fits in memory, such as NCRL's retrieved prior buffer.
  Windows are uniform over steps and may cross episode boundaries. The
  per-step ``is_first`` flag masks the boundary.
- :class:`MmapSequenceDataset` samples straight from the memory-mapped files
  and returns a ``SequenceBatch``. It draws an episode uniformly, then a
  window start uniformly inside it, and never crosses an episode boundary
  (the official NCRL sampler). Every window starts from a
  zero carry with ``is_first[0] = True`` (no latent replay; the official NCRL code does the same).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.world_models.batch import SequenceBatch

SEQUENCE_DATASET_FORMAT = "rl_garden_sequence_v1"
OBS_FILE = "obs.npy"
EPISODES_FILE = "episodes.pt"
MANIFEST_FILE = "manifest.json"


def sequence_dataset_part_dirs(dataset_dir: str | Path) -> list[Path]:
    """The part directories under ``dataset_dir``: itself if it holds a
    ``manifest.json``, otherwise every immediate subdirectory that does
    (sorted by name)."""
    root = Path(dataset_dir)
    if (root / MANIFEST_FILE).is_file():
        return [root]
    parts = sorted(p for p in root.iterdir() if (p / MANIFEST_FILE).is_file()) if root.is_dir() else []
    if not parts:
        raise FileNotFoundError(
            f"No {MANIFEST_FILE} in {root} or its subdirectories; run "
            "tools/conversion/convert_ncrl_dataset.py first."
        )
    return parts


def read_part_manifest(part_dir: str | Path) -> dict:
    manifest = json.loads((Path(part_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("format") != SEQUENCE_DATASET_FORMAT:
        raise ValueError(
            f"{part_dir}: unsupported sequence-dataset format {manifest.get('format')!r}, "
            f"expected {SEQUENCE_DATASET_FORMAT!r}."
        )
    return manifest


@dataclass(frozen=True)
class SequenceDatasetSpec:
    """Shape facts read from the manifests only (no tensor data touched)."""

    obs_key: str
    image_shape: tuple[int, int, int]
    padded_action_dim: int
    embodiments: tuple[str, ...]
    action_dims: tuple[int, ...]
    num_episodes: int
    num_steps: int
    max_episode_length: int
    min_episode_length: int


def infer_sequence_dataset_spec(dataset_dir: str | Path) -> SequenceDatasetSpec:
    manifests = [read_part_manifest(p) for p in sequence_dataset_part_dirs(dataset_dir)]
    for field in ("obs_key", "image_shape", "padded_action_dim"):
        values = {json.dumps(m[field]) for m in manifests}
        if len(values) > 1:
            raise ValueError(f"Parts of {dataset_dir} disagree on {field}: {sorted(values)}.")
    lengths = [length for m in manifests for length in m["episode_lengths"]]
    return SequenceDatasetSpec(
        obs_key=manifests[0]["obs_key"],
        image_shape=tuple(manifests[0]["image_shape"]),
        padded_action_dim=int(manifests[0]["padded_action_dim"]),
        embodiments=tuple(m["embodiment"] for m in manifests),
        action_dims=tuple(int(m["action_dim"]) for m in manifests),
        num_episodes=sum(int(m["num_episodes"]) for m in manifests),
        num_steps=sum(lengths),
        max_episode_length=max(lengths),
        min_episode_length=min(lengths),
    )


@dataclass
class _Part:
    manifest: dict
    obs: np.ndarray  # read-only memmap, (N_steps, H, W, 3) uint8
    action: torch.Tensor  # (N_steps, A_pad)
    reward: torch.Tensor  # (N_steps,)
    episode_starts: np.ndarray  # (num_episodes,)
    episode_lengths: np.ndarray  # (num_episodes,)


def _open_part(part_dir: Path) -> _Part:
    manifest = read_part_manifest(part_dir)
    obs = np.load(part_dir / OBS_FILE, mmap_mode="r")
    episodes = torch.load(part_dir / EPISODES_FILE, mmap=True, weights_only=True)
    lengths = np.asarray(manifest["episode_lengths"], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    num_steps = int(lengths.sum())
    if obs.shape[0] != num_steps or episodes["action"].shape[0] != num_steps:
        raise ValueError(
            f"{part_dir}: manifest has {num_steps} steps but obs/action have "
            f"{obs.shape[0]}/{episodes['action'].shape[0]} rows."
        )
    return _Part(manifest, obs, episodes["action"], episodes["reward"], starts, lengths)


def load_sequence_dataset(
    dataset_dir: str | Path,
    buffer: SequenceReplayBuffer,
    *,
    embodiments: Optional[tuple[str, ...]] = None,
) -> int:
    """Copies every episode of ``dataset_dir`` (or only the parts named in
    ``embodiments``) into ``buffer``, a uniform-mode ``SequenceReplayBuffer``
    with ``num_envs=1`` whose observation space has the dataset's
    ``obs_key``. Each step is written with ``add()``: ``episode_end`` on the
    last step of an episode, ``is_first`` on its first, ``done=False`` (the
    converted datasets carry no terminal flag) and a zero ``carry`` for every
    ``carry_spec`` key. Actions narrower than the buffer's action space are
    zero-padded. Size the buffer with ``infer_sequence_dataset_spec(...)
    .num_steps`` to hold everything. Returns the number of steps written."""
    if not (buffer.cross_episode and not buffer.priority):
        raise ValueError("load_sequence_dataset needs a uniform-mode buffer (cross_episode=True, priority=False).")
    if buffer.num_envs != 1:
        raise ValueError(f"load_sequence_dataset writes one stream; got buffer.num_envs={buffer.num_envs}.")
    buffer_action_dim = int(buffer.action_space.shape[0])
    zero_carry = {
        key: torch.zeros((1, *shape), dtype=torch.float32) for key, shape in buffer._carry_spec.items()
    } or None

    written = 0
    for part_dir in sequence_dataset_part_dirs(dataset_dir):
        part = _open_part(part_dir)
        if embodiments is not None and part.manifest["embodiment"] not in embodiments:
            continue
        obs_key = part.manifest["obs_key"]
        if obs_key not in buffer.observation_space.spaces:
            raise KeyError(f"Buffer observation space has no {obs_key!r} key.")
        part_action_dim = part.action.shape[1]
        if part_action_dim > buffer_action_dim:
            raise ValueError(
                f"{part_dir}: padded action dim {part_action_dim} exceeds the buffer's {buffer_action_dim}."
            )
        for start, length in zip(part.episode_starts.tolist(), part.episode_lengths.tolist()):
            obs = torch.from_numpy(np.array(part.obs[start : start + length]))
            action = torch.zeros(length, buffer_action_dim)
            action[:, :part_action_dim] = part.action[start : start + length]
            reward = part.reward[start : start + length].float()
            for t in range(length):
                last = t == length - 1
                buffer.add(
                    {obs_key: obs[t : t + 1]},
                    {obs_key: obs[t : t + 1] if last else obs[t + 1 : t + 2]},
                    action[t : t + 1],
                    reward[t : t + 1],
                    torch.zeros(1, dtype=torch.bool),
                    torch.tensor([last]),
                    is_first=torch.tensor([t == 0]),
                    carry=zero_carry,
                )
            written += length
    return written


class MmapSequenceDataset:
    """Samples ``(horizon + 1, B)`` windows straight from the memory-mapped
    dataset files. ``sample(B)`` has the same return type as
    ``SequenceReplayBuffer.sample`` in uniform mode (a ``SequenceBatch``), so
    an offline model-based algorithm can use either.

    Window layout (loss rows ``1..horizon`` of ``obs``, the convention
    ``RSSM.model_loss`` reads): pick an episode uniformly among those with at
    least ``horizon`` steps, then a first loss step ``s`` uniformly in
    ``[0, length - horizon]``. ``obs[1:]`` are episode steps ``s ..
    s + horizon - 1``; ``action[k]``/``reward[k]`` are the transition that led
    into loss row ``k`` (episode step ``s + k - 1``). Row 0 of ``obs`` and
    ``action[0]``/``reward[0]`` are never used, because every window starts
    with ``is_first[0] = True`` and a zero ``carry``. When ``s`` is the
    episode's first step they hold copies/zeros. Windows never cross an
    episode boundary, so ``is_first[1:]`` is all False.

    ``carry_spec`` maps each carry key to its per-sample shape, for example
    ``{"deter": (deter,), "stoch": (stoch * classes,)}`` for an RSSM.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        horizon: int,
        *,
        carry_spec: Optional[dict[str, tuple[int, ...]]] = None,
        sample_device: torch.device | str = "cuda",
        embodiments: Optional[tuple[str, ...]] = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = horizon
        self.sample_device = torch.device(sample_device)
        self.carry_spec = dict(carry_spec or {})
        self.parts = [
            part
            for part in (_open_part(p) for p in sequence_dataset_part_dirs(dataset_dir))
            if embodiments is None or part.manifest["embodiment"] in embodiments
        ]
        if not self.parts:
            raise ValueError(f"No dataset parts selected from {dataset_dir} (embodiments={embodiments}).")
        obs_keys = {part.manifest["obs_key"] for part in self.parts}
        action_dims = {part.action.shape[1] for part in self.parts}
        if len(obs_keys) != 1 or len(action_dims) != 1:
            raise ValueError(f"Parts disagree on obs_key/padded action dim: {obs_keys}/{action_dims}.")
        self.obs_key = obs_keys.pop()
        self.action_dim = action_dims.pop()

        part_index, starts, lengths = [], [], []
        for index, part in enumerate(self.parts):
            eligible = part.episode_lengths >= horizon
            part_index.append(np.full(int(eligible.sum()), index, dtype=np.int64))
            starts.append(part.episode_starts[eligible])
            lengths.append(part.episode_lengths[eligible])
        self._episode_part = np.concatenate(part_index)
        self._episode_start = np.concatenate(starts)
        self._episode_length = np.concatenate(lengths)
        if self._episode_part.size == 0:
            raise ValueError(f"No episode in {dataset_dir} has at least horizon={horizon} steps.")

    @property
    def num_episodes(self) -> int:
        return int(self._episode_part.size)

    @property
    def num_steps(self) -> int:
        return int(sum(part.episode_lengths.sum() for part in self.parts))

    def sample(self, batch_size: int) -> SequenceBatch:
        horizon = self.horizon
        episodes = torch.randint(0, self.num_episodes, (batch_size,)).numpy()
        max_offset = self._episode_length[episodes] - horizon
        offsets = (torch.rand(batch_size).numpy() * (max_offset + 1)).astype(np.int64)
        offsets = np.minimum(offsets, max_offset)

        obs = np.empty((horizon + 1, batch_size, *self.parts[0].obs.shape[1:]), dtype=np.uint8)
        action = torch.zeros(horizon, batch_size, self.action_dim)
        reward = torch.zeros(horizon, batch_size)
        for b, (episode, offset) in enumerate(zip(episodes.tolist(), offsets.tolist())):
            part = self.parts[self._episode_part[episode]]
            first = int(self._episode_start[episode]) + offset
            obs[1:, b] = part.obs[first : first + horizon]
            if offset == 0:
                obs[0, b] = obs[1, b]
                action[1:, b] = part.action[first : first + horizon - 1]
                reward[1:, b] = part.reward[first : first + horizon - 1]
            else:
                obs[0, b] = part.obs[first - 1]
                action[:, b] = part.action[first - 1 : first + horizon - 1]
                reward[:, b] = part.reward[first - 1 : first + horizon - 1]

        device = self.sample_device
        is_first = torch.zeros(horizon, batch_size, dtype=torch.bool, device=device)
        is_first[0] = True
        return SequenceBatch(
            obs={self.obs_key: torch.from_numpy(obs).to(device)},
            action=action.to(device),
            reward=reward.to(device),
            is_first=is_first,
            terminated=torch.zeros(horizon, batch_size, dtype=torch.bool, device=device),
            carry={
                key: torch.zeros((batch_size, *shape), device=device) for key, shape in self.carry_spec.items()
            },
        )
