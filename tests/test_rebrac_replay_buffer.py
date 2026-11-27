from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers import ReBRACReplayBuffer
from rl_garden.buffers.dataset_common import add_flat_transitions

OBS_SPACE = spaces.Dict(
    {"state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)}
)
ACT_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _obs() -> dict:
    return {"state": torch.randn(1, 4)}


def _make_buffer(buffer_size: int = 10, num_envs: int = 1) -> ReBRACReplayBuffer:
    return ReBRACReplayBuffer(
        OBS_SPACE,
        ACT_SPACE,
        num_envs=num_envs,
        buffer_size=buffer_size,
        storage_device="cpu",
        sample_device="cpu",
    )


def test_next_actions_matches_the_following_stored_transition():
    buf = _make_buffer()
    stored_actions = []
    for i in range(6):
        a = torch.full((1, 2), float(i))
        stored_actions.append(a)
        buf.add(_obs(), _obs(), a, torch.randn(1), torch.zeros(1))

    sample = buf._index_batch(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 0, 0, 0]))
    for i in range(4):
        assert torch.equal(sample.next_actions[i], stored_actions[i + 1][0])
        assert torch.equal(sample.actions[i], stored_actions[i][0])


def test_next_actions_can_spill_into_next_episode_at_a_terminal():
    """Documented, CORL-matching simplification: no terminal guard on the
    index-shift, so next_actions at a true terminal reads whatever was
    stored next (possibly the start of the following episode)."""
    buf = _make_buffer()
    a0 = torch.full((1, 2), 0.0)
    a1 = torch.full((1, 2), 1.0)  # first action of the *next* episode
    buf.add(_obs(), _obs(), a0, torch.randn(1), torch.ones(1))  # terminal
    buf.add(_obs(), _obs(), a1, torch.randn(1), torch.zeros(1))

    sample = buf._index_batch(torch.tensor([0]), torch.tensor([0]))
    assert torch.equal(sample.next_actions[0], a1[0])


def test_sample_returns_rebrac_sample_with_next_actions():
    buf = _make_buffer()
    for _ in range(8):
        buf.add(_obs(), _obs(), torch.randn(1, 2), torch.randn(1), torch.zeros(1))
    sample = buf.sample(4)
    assert sample.next_actions.shape == (4, 2)
    assert sample.actions.shape == (4, 2)


def _load_flat_rows(buf: ReBRACReplayBuffer, num_rows: int) -> None:
    obs = {"state": torch.randn(num_rows, 4)}
    add_flat_transitions(
        buf, obs, obs, torch.randn(num_rows, 2), torch.randn(num_rows), torch.zeros(num_rows)
    )


def test_sampleable_size_and_epoch_size_exclude_the_cut_rows():
    buf = _make_buffer(num_envs=2)
    _load_flat_rows(buf, 10)  # one stretch of 5 rows per env, the last row of each is cut
    assert len(buf) == 10
    assert buf.sampleable_size == buf.epoch_size == 8


def test_sample_with_every_row_cut_raises_a_named_error():
    buf = _make_buffer(num_envs=2)
    _load_flat_rows(buf, 2)  # one row per env, both cut
    assert buf.sampleable_size == buf.epoch_size == 0
    with pytest.raises(ValueError, match="no transition with a stored next action"):
        buf.sample(1)
