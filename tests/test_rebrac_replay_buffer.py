from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers import ReBRACReplayBuffer

OBS_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


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
        buf.add(torch.randn(1, 4), torch.randn(1, 4), a, torch.randn(1), torch.zeros(1))

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
    buf.add(torch.randn(1, 4), torch.randn(1, 4), a0, torch.randn(1), torch.ones(1))  # terminal
    buf.add(torch.randn(1, 4), torch.randn(1, 4), a1, torch.randn(1), torch.zeros(1))

    sample = buf._index_batch(torch.tensor([0]), torch.tensor([0]))
    assert torch.equal(sample.next_actions[0], a1[0])


def test_sample_returns_rebrac_sample_with_next_actions():
    buf = _make_buffer()
    for _ in range(8):
        buf.add(
            torch.randn(1, 4), torch.randn(1, 4), torch.randn(1, 2), torch.randn(1), torch.zeros(1)
        )
    sample = buf.sample(4)
    assert sample.next_actions.shape == (4, 2)
    assert sample.actions.shape == (4, 2)
