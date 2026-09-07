from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers import SarsaMCTensorReplayBuffer

OBS_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _make_buffer(buffer_size: int = 10, num_envs: int = 1) -> SarsaMCTensorReplayBuffer:
    return SarsaMCTensorReplayBuffer(
        OBS_SPACE,
        ACT_SPACE,
        num_envs=num_envs,
        buffer_size=buffer_size,
        gamma=0.99,
        storage_device="cpu",
        sample_device="cpu",
    )


def test_next_actions_matches_following_transition_within_an_episode():
    buf = _make_buffer()
    stored_actions = []
    for i in range(4):
        a = torch.full((1, 2), float(i))
        stored_actions.append(a)
        # Mid-episode: done=False, episode_end=False.
        buf.add(
            torch.randn(1, 4), torch.randn(1, 4), a, torch.randn(1),
            torch.zeros(1), episode_end=torch.zeros(1),
        )

    sample = buf._index_batch(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]))
    for i in range(3):
        assert torch.equal(sample.next_actions[i], stored_actions[i + 1][0])
        assert bool(sample.next_action_valid[i])


def test_next_action_invalid_at_timeout_truncation_unlike_rebrac():
    """The exact gap ReBRAC's own next-action shift doesn't guard against:
    a `timeouts`-only boundary (done=False, episode_end=True) still spills
    the index-shift into the next episode's first action, but this buffer
    must flag that row as invalid via next_action_valid."""
    buf = _make_buffer()
    a0 = torch.full((1, 2), 0.0)
    a1 = torch.full((1, 2), 1.0)  # first action of the *next* episode
    # Locomotion-style timeout: done=False (TD bootstraps through it), but
    # episode_end=True (MC/SARSA must stop here).
    buf.add(
        torch.randn(1, 4), torch.randn(1, 4), a0, torch.randn(1),
        torch.zeros(1), episode_end=torch.ones(1),
    )
    buf.add(
        torch.randn(1, 4), torch.randn(1, 4), a1, torch.randn(1),
        torch.zeros(1), episode_end=torch.zeros(1),
    )

    sample = buf._index_batch(torch.tensor([0]), torch.tensor([0]))
    # The raw shift still lands on the next episode's action (documented
    # mechanism), but it must be marked invalid.
    assert torch.equal(sample.next_actions[0], a1[0])
    assert not bool(sample.next_action_valid[0])


def test_next_action_invalid_at_true_terminal():
    buf = _make_buffer()
    a0 = torch.full((1, 2), 0.0)
    a1 = torch.full((1, 2), 1.0)
    buf.add(
        torch.randn(1, 4), torch.randn(1, 4), a0, torch.randn(1),
        torch.ones(1), episode_end=torch.ones(1),
    )
    buf.add(
        torch.randn(1, 4), torch.randn(1, 4), a1, torch.randn(1),
        torch.zeros(1), episode_end=torch.zeros(1),
    )

    sample = buf._index_batch(torch.tensor([0]), torch.tensor([0]))
    assert not bool(sample.next_action_valid[0])


def test_sample_returns_sarsa_mc_sample_with_expected_shapes():
    buf = _make_buffer()
    for i in range(8):
        done = torch.ones(1) if i == 7 else torch.zeros(1)
        buf.add(
            torch.randn(1, 4), torch.randn(1, 4), torch.randn(1, 2), torch.randn(1),
            done, episode_end=done,
        )
    sample = buf.sample(4)
    assert sample.next_actions.shape == (4, 2)
    assert sample.next_action_valid.shape == (4,)
    assert sample.mc_returns.shape == (4,)
    assert sample.actions.shape == (4, 2)
