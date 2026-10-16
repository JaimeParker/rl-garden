"""``RehearsalSequenceBuffer``: NCRL experience rehearsal.

It composes two uniform-mode ``SequenceReplayBuffer``s
(``cross_episode=True, priority=False``):

- ``online``: filled by the rollout loop (``add()`` goes here only).
- ``prior``: retrieved offline episodes, written once with
  :meth:`RehearsalSequenceBuffer.load_prior_episodes` and fixed afterwards.

``sample(B)`` draws ``n_on = round(online_ratio * B)`` windows from ``online``
and ``B - n_on`` from ``prior`` and concatenates them along the batch dim,
online columns first (official ``dreamer.py:90``). Prior columns always get
``is_first[0] = True`` (official chunks start from the zero state); online
columns keep the DreamerV3 carry write-back. ``reward_valid`` marks the
online columns, so the reward head only trains on them (as in the official NCRL code, which detaches the reward-head input). With
no prior (``None`` or too short to hold one window) or ``online_ratio == 1``
the batch is purely online and ``prior`` is never touched.

Episodes passed to ``load_prior_episodes`` use the rl-garden sequence-dataset
alignment (``rl_garden.buffers.sequence_dataset``): ``action[t]`` is executed
at ``obs[t]`` and ``reward[t]`` is its result. :func:`read_dataset_episodes`
reads such episodes from a converted dataset directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

import numpy as np
import torch

from rl_garden.buffers.sequence_dataset import (
    EPISODES_FILE,
    OBS_FILE,
    read_part_manifest,
    sequence_dataset_part_dirs,
)
from rl_garden.buffers.sequence_replay_buffer import DreamerSequenceBatch, SequenceReplayBuffer


@dataclass
class SequenceEpisode:
    """One offline episode in rl-garden alignment. ``obs``: uint8 ``(L, H, W,
    C)``; ``action``: ``(L, A)``; ``reward``: ``(L,)``. ``key`` identifies it
    as ``(embodiment, episode_id)``."""

    key: tuple[str, int]
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.obs.shape[0])


def read_dataset_episodes(dataset_dir: str | Path, keys: Iterable[tuple[str, int]]) -> list[SequenceEpisode]:
    """Reads the episodes named by ``(embodiment, episode_id)`` from a converted
    sequence dataset (``tools/conversion/convert_ncrl_dataset.py`` output).
    ``episode_id`` is the per-embodiment index stored in ``episodes.pt``.
    Returned in the order of ``keys``."""
    keys = [(str(emb), int(ep)) for emb, ep in keys]
    wanted: dict[str, set[int]] = {}
    for emb, ep in keys:
        wanted.setdefault(emb, set()).add(ep)

    found: dict[tuple[str, int], SequenceEpisode] = {}
    for part_dir in sequence_dataset_part_dirs(dataset_dir):
        manifest = read_part_manifest(part_dir)
        embodiment = manifest["embodiment"]
        if embodiment not in wanted:
            continue
        obs = np.load(part_dir / OBS_FILE, mmap_mode="r")
        episodes = torch.load(part_dir / EPISODES_FILE, mmap=True, weights_only=True)
        lengths = np.asarray(manifest["episode_lengths"], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        for ep in sorted(wanted[embodiment]):
            if not 0 <= ep < lengths.size:
                raise KeyError(f"{part_dir}: episode_id {ep} out of range [0, {lengths.size}).")
            start, length = int(starts[ep]), int(lengths[ep])
            found[(embodiment, ep)] = SequenceEpisode(
                key=(embodiment, ep),
                obs=torch.from_numpy(np.array(obs[start : start + length])),
                action=episodes["action"][start : start + length].clone().float(),
                reward=episodes["reward"][start : start + length].clone().float(),
            )
    missing = [key for key in keys if key not in found]
    if missing:
        raise KeyError(f"Episodes {missing[:5]} (of {len(missing)} missing) not found under {dataset_dir}.")
    return [found[key] for key in keys]


class RehearsalIndices(NamedTuple):
    """``DreamerSequenceBatch.indices`` of a rehearsal batch: the first
    ``num_online`` batch columns came from ``online`` (its own
    ``(idx_grid, env_grid)``), the rest from ``prior`` (``None`` when the
    batch has no prior columns)."""

    num_online: int
    online: tuple[torch.Tensor, torch.Tensor]
    prior: Optional[tuple[torch.Tensor, torch.Tensor]]


class RehearsalSequenceBuffer:
    def __init__(
        self,
        online: SequenceReplayBuffer,
        prior: Optional[SequenceReplayBuffer] = None,
        online_ratio: float = 0.75,
    ) -> None:
        if not 0.0 < online_ratio <= 1.0:
            raise ValueError(f"online_ratio must be in (0, 1], got {online_ratio}.")
        for name, buffer in (("online", online), ("prior", prior)):
            if buffer is not None and not (buffer.cross_episode and not buffer.priority):
                raise ValueError(f"{name} must be a uniform-mode SequenceReplayBuffer (cross_episode=True, priority=False).")
        self.online = online
        self.prior = prior
        self.online_ratio = online_ratio

    # ------------------------------------------------------------------
    # OffPolicyAlgorithm-facing surface (forwarded to the online buffer)
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self.online.num_envs

    @property
    def buffer_size(self) -> int:
        return self.online.buffer_size

    @property
    def per_env_buffer_size(self) -> int:
        return self.online.per_env_buffer_size

    @property
    def pos(self) -> int:
        return self.online.pos

    @property
    def full(self) -> bool:
        return self.online.full

    @property
    def size(self) -> int:
        return self.online.size

    @property
    def horizon(self) -> int:
        return self.online.horizon

    @property
    def storage_device(self) -> torch.device:
        return self.online.storage_device

    @property
    def sample_device(self) -> torch.device:
        return self.online.sample_device

    def add(self, *args, **kwargs) -> None:
        self.online.add(*args, **kwargs)

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------

    @property
    def prior_size(self) -> int:
        return 0 if self.prior is None else self.prior.size * self.prior.num_envs

    def _prior_sampleable(self) -> bool:
        return self.prior is not None and self.prior.size > self.prior.horizon

    def load_prior_episodes(self, episodes: Iterable[SequenceEpisode]) -> int:
        """Writes every episode into ``prior`` step by step (``is_first`` on
        its first step, ``episode_end`` on its last, ``done=False``, zero
        carry). Actions narrower than the prior's action space are
        zero-padded. Returns the number of steps written."""
        if self.prior is None:
            raise RuntimeError("load_prior_episodes needs a prior buffer; set .prior first.")
        prior = self.prior
        if prior.num_envs != 1:
            raise ValueError(f"The prior buffer must have num_envs=1, got {prior.num_envs}.")
        (obs_key,) = prior.observation_space.spaces.keys()
        action_dim = int(prior.action_space.shape[0])
        zero_carry = {key: torch.zeros((1, *shape)) for key, shape in prior._carry_spec.items()} or None
        written = 0
        for episode in episodes:
            length = episode.length
            if episode.action.shape[1] > action_dim:
                raise ValueError(
                    f"Episode {episode.key}: action dim {episode.action.shape[1]} exceeds the prior's {action_dim}."
                )
            obs = episode.obs.to(prior.storage_device)
            action = torch.zeros(length, action_dim, device=prior.storage_device)
            action[:, : episode.action.shape[1]] = episode.action.to(prior.storage_device)
            reward = episode.reward.to(prior.storage_device)
            no_done = torch.zeros(1, dtype=torch.bool)
            for t in range(length):
                last = t == length - 1
                next_t = t if last else t + 1
                prior.add(
                    {obs_key: obs[t : t + 1]},
                    {obs_key: obs[next_t : next_t + 1]},
                    action[t : t + 1],
                    reward[t : t + 1],
                    no_done,
                    torch.tensor([last]),
                    is_first=torch.tensor([t == 0]),
                    carry=zero_carry,
                )
            written += length
        return written

    # ------------------------------------------------------------------
    # Sampling / carry write-back
    # ------------------------------------------------------------------

    def split_sizes(self, batch_size: int) -> tuple[int, int]:
        """``(n_online, n_prior)`` for one ``sample(batch_size)`` call."""
        if self.online_ratio >= 1.0 or not self._prior_sampleable():
            return batch_size, 0
        # At least one online column, so the reward loss never averages over none.
        num_online = max(1, int(round(self.online_ratio * batch_size)))
        return num_online, batch_size - num_online

    def sample(self, batch_size: int) -> DreamerSequenceBatch:
        num_online, num_prior = self.split_sizes(batch_size)
        online_batch = self.online.sample(num_online)
        device = online_batch.action.device
        if num_prior == 0:
            online_batch.reward_valid = torch.ones(num_online, dtype=torch.bool, device=device)
            online_batch.indices = RehearsalIndices(num_online, online_batch.indices, None)
            return online_batch

        prior_batch = self.prior.sample(num_prior)
        both = (online_batch, prior_batch)
        is_first = torch.cat([b.is_first for b in both], dim=1)
        # Prior columns start cold (zero carry, zeroed first action), as every
        # official NCRL chunk does; online columns keep the carry write-back.
        is_first[0, num_online:] = True
        return DreamerSequenceBatch(
            obs={key: torch.cat([b.obs[key] for b in both], dim=1) for key in online_batch.obs},
            action=torch.cat([b.action for b in both], dim=1),
            reward=torch.cat([b.reward for b in both], dim=1),
            is_first=is_first,
            terminated=torch.cat([b.terminated for b in both], dim=1),
            is_last=torch.cat([b.is_last for b in both], dim=1),
            carry={key: torch.cat([b.carry[key] for b in both], dim=0) for key in online_batch.carry},
            reward_valid=torch.cat(
                [
                    torch.ones(num_online, dtype=torch.bool, device=device),
                    torch.zeros(num_prior, dtype=torch.bool, device=device),
                ]
            ),
            indices=RehearsalIndices(num_online, online_batch.indices, prior_batch.indices),
        )

    def write_back_carry(self, indices: RehearsalIndices, carry: dict[str, torch.Tensor]) -> None:
        """``carry`` tensors are ``(horizon, B, ...)`` (a posterior); columns
        ``[:num_online]`` go back to ``online``, the rest to ``prior``."""
        num_online = indices.num_online
        self.online.write_back_carry(indices.online, {k: v[:, :num_online] for k, v in carry.items()})
        if indices.prior is not None:
            self.prior.write_back_carry(indices.prior, {k: v[:, num_online:] for k, v in carry.items()})
