"""``RehearsalSequenceBuffer``: batch-dim concatenation (online columns
first) and ``reward_valid``; prior columns start cold (``is_first[0]``); carry write-back routed to the right buffer and
rows; pure-online fallback without a usable prior; ``online_ratio=1`` never
touches the prior; ``load_prior_episodes`` layout."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers.rehearsal_sequence_buffer import (
    RehearsalSequenceBuffer,
    SequenceEpisode,
)
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer

HORIZON = 4
OBS_SPACE = spaces.Dict({"state": spaces.Box(-1e6, 1e6, (2,), dtype=np.float32)})
ACTION_SPACE = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
CARRY_SPEC = {"deter": (5,), "stoch": (2,)}


def _buffer(num_envs: int, size: int) -> SequenceReplayBuffer:
    return SequenceReplayBuffer(
        OBS_SPACE,
        ACTION_SPACE,
        num_envs,
        size,
        cross_episode=True,
        priority=False,
        horizon=HORIZON,
        carry_spec=CARRY_SPEC,
        storage_device="cpu",
        sample_device="cpu",
    )


def _fill_online(buffer: SequenceReplayBuffer, steps: int) -> None:
    """Online observations are positive: ``1000 + 10 * t + env``."""
    for t in range(steps):
        obs = {"state": torch.stack([torch.full((2,), 1000.0 + 10 * t + e) for e in range(buffer.num_envs)])}
        buffer.add(
            obs,
            obs,
            torch.zeros(buffer.num_envs, 3),
            torch.ones(buffer.num_envs),
            torch.zeros(buffer.num_envs, dtype=torch.bool),
            torch.zeros(buffer.num_envs, dtype=torch.bool),
            is_first=torch.full((buffer.num_envs,), t == 0),
            carry={k: torch.zeros(buffer.num_envs, *s) for k, s in CARRY_SPEC.items()},
        )


def _prior_episodes() -> list[SequenceEpisode]:
    """Prior observations are negative: ``-(100 * ep + t + 1)``."""
    episodes = []
    for ep, length in enumerate((7, 9)):
        t = torch.arange(length, dtype=torch.float32)
        episodes.append(
            SequenceEpisode(
                key=("walker", ep),
                obs=(-(100.0 * ep + t + 1)).unsqueeze(1).repeat(1, 2),
                action=torch.full((length, 2), float(ep + 1)),  # narrower than the action space: padded
                reward=torch.zeros(length),
            )
        )
    return episodes


def _rehearsal(online_ratio: float = 0.75, with_prior: bool = True) -> RehearsalSequenceBuffer:
    online = _buffer(num_envs=2, size=40)
    _fill_online(online, 15)
    buffer = RehearsalSequenceBuffer(online, prior=_buffer(1, 16) if with_prior else None, online_ratio=online_ratio)
    if with_prior:
        assert buffer.load_prior_episodes(_prior_episodes()) == 16
    return buffer


def test_load_prior_episodes_layout():
    buffer = _rehearsal()
    prior = buffer.prior
    assert prior.size == 16 and buffer.prior_size == 16
    assert prior._is_first[:, 0].nonzero().flatten().tolist() == [0, 7]
    assert prior.episode_ends[:, 0].nonzero().flatten().tolist() == [6, 15]
    assert torch.equal(prior.actions[0, 0], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(prior.actions[7, 0], torch.tensor([2.0, 2.0, 0.0]))
    assert not prior.dones.any()


def test_sample_concatenates_online_first_with_reward_valid():
    buffer = _rehearsal()
    batch = buffer.sample(8)
    assert buffer.split_sizes(8) == (6, 2)
    obs = batch.obs["state"][..., 0]  # (HORIZON + 1, 8)
    assert obs.shape == (HORIZON + 1, 8)
    assert (obs[:, :6] > 0).all() and (obs[:, 6:] < 0).all()
    assert batch.reward_valid.tolist() == [True] * 6 + [False] * 2
    assert batch.action.shape == (HORIZON, 8, 3)
    for field in ("reward", "is_first", "terminated", "is_last"):
        assert getattr(batch, field).shape == (HORIZON, 8), field
    assert batch.carry["deter"].shape == (8, 5)
    assert batch.indices.num_online == 6
    assert batch.indices.online[0].shape == (HORIZON, 6) and batch.indices.prior[0].shape == (HORIZON, 2)


def test_prior_columns_start_cold_and_online_columns_are_untouched():
    buffer = _rehearsal()
    online_first = []
    original_sample = buffer.online.sample

    def recording_sample(n):
        batch = original_sample(n)
        online_first.append(batch.is_first.clone())
        return batch

    buffer.online.sample = recording_sample
    for _ in range(20):
        batch = buffer.sample(8)
        assert batch.is_first[0, 6:].all()
        assert torch.equal(batch.is_first[:, :6], online_first[-1])
    # Online windows rarely start on an episode boundary, so is_first[0] is not forced there.
    assert not torch.cat([first[0] for first in online_first]).all()


def test_write_back_carry_routes_columns_to_their_buffer():
    buffer = _rehearsal()
    batch = buffer.sample(8)
    column = torch.arange(8, dtype=torch.float32)
    carry = {
        "deter": column.reshape(1, 8, 1).expand(HORIZON, 8, 5).clone(),
        "stoch": column.reshape(1, 8, 1).expand(HORIZON, 8, 2).clone(),
    }
    buffer.write_back_carry(batch.indices, carry)
    online_idx, online_env = batch.indices.online
    prior_idx, prior_env = batch.indices.prior
    online_written = buffer.online._carry["deter"][online_idx, online_env][..., 0]
    prior_written = buffer.prior._carry["deter"][prior_idx, prior_env][..., 0]
    # Uniform windows can overlap, so a slot may hold any of the columns that wrote it.
    assert set(online_written.unique().tolist()) <= set(range(6))
    assert set(prior_written.unique().tolist()) <= {6.0, 7.0}
    assert (buffer.online._carry["deter"][..., 0] < 6).all()


def test_without_prior_batch_is_purely_online():
    buffer = _rehearsal(with_prior=False)
    batch = buffer.sample(8)
    assert batch.reward_valid.tolist() == [True] * 8
    assert (batch.obs["state"] > 0).all()
    assert batch.indices.prior is None
    buffer.write_back_carry(batch.indices, {k: torch.ones(HORIZON, 8, *s) for k, s in CARRY_SPEC.items()})


def test_prior_shorter_than_a_window_is_ignored():
    online = _buffer(num_envs=2, size=40)
    _fill_online(online, 15)
    buffer = RehearsalSequenceBuffer(online, prior=_buffer(1, 16), online_ratio=0.5)
    buffer.load_prior_episodes(_prior_episodes()[:1])  # 7 steps
    buffer.prior.pos = 3  # pretend only 3 rows (< HORIZON + 1) were written
    assert buffer.split_sizes(8) == (8, 0)


def test_online_ratio_one_never_touches_prior(monkeypatch):
    buffer = _rehearsal(online_ratio=1.0)

    def _forbidden(*args, **kwargs):
        raise AssertionError("prior was sampled")

    monkeypatch.setattr(buffer.prior, "sample", _forbidden)
    monkeypatch.setattr(buffer.prior, "write_back_carry", _forbidden)
    batch = buffer.sample(8)
    assert batch.reward_valid.all()
    buffer.write_back_carry(batch.indices, {k: torch.ones(HORIZON, 8, *s) for k, s in CARRY_SPEC.items()})


def test_add_and_surface_forward_to_online():
    buffer = _rehearsal()
    pos_before = buffer.pos
    _fill_online(buffer, 1)  # add() through the composite buffer
    assert buffer.pos == pos_before + 1 == buffer.online.pos
    assert buffer.num_envs == 2 and buffer.size == buffer.online.size and buffer.horizon == HORIZON


def test_invalid_online_ratio_rejected():
    with pytest.raises(ValueError, match="online_ratio"):
        RehearsalSequenceBuffer(_buffer(1, 16), online_ratio=0.0)
