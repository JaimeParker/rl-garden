"""``StateSequenceReplayBuffer.sample_history``: in-episode observation
history with first-frame replication, and an unchanged ``sample()``."""
from __future__ import annotations

import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.buffers.state_sequence_replay_buffer import StateSequenceReplayBuffer

# state = [episode id, step in episode, global write counter]
_OBS_SPACE = spaces.Dict({"state": spaces.Box(-1e6, 1e6, (3,), dtype="float32")})
_ACTION_SPACE = spaces.Box(-1e6, 1e6, (1,), dtype="float32")


def _buffer(cls=StateSequenceReplayBuffer, *, num_envs: int = 1, buffer_size: int = 16, horizon: int = 2):
    return cls(
        _OBS_SPACE,
        _ACTION_SPACE,
        num_envs,
        buffer_size,
        cross_episode=False,
        horizon=horizon,
        storage_device="cpu",
        sample_device="cpu",
    )


class _Writer:
    """Writes episodes into a buffer; the action of a row is ``counter + 100``."""

    def __init__(self, buffer, num_envs: int = 1) -> None:
        self.buffer = buffer
        self.num_envs = num_envs
        self.counter = 0

    def episode(self, episode_id: int, length: int, *, ends: bool = True) -> None:
        for step in range(length):
            last = ends and step == length - 1

            def obs(k: int) -> dict[str, torch.Tensor]:
                row = torch.tensor([episode_id, k, self.counter], dtype=torch.float32)
                return {"state": row.expand(self.num_envs, 3).clone()}

            self.buffer.add(
                obs(step),
                obs(step + 1),
                torch.full((self.num_envs, 1), self.counter + 100.0),
                torch.full((self.num_envs,), float(self.counter)),
                torch.full((self.num_envs,), last),
                torch.full((self.num_envs,), last),
            )
            self.counter += 1


def _check_history(sample, history_len: int, earliest_step) -> set[int]:
    """Checks every sampled column against the expected in-episode history;
    ``earliest_step(start)`` is the earliest step still stored for that
    start's episode. Returns the start steps seen."""
    seen = set()
    states = sample.obs["state"]
    for b in range(states.shape[1]):
        episode, step, counter = (int(v) for v in states[-1, b])
        seen.add(step)
        first = earliest_step(episode, step)
        for back in range(1, history_len):
            row = history_len - 1 - back
            expected_step = max(step - back, first)
            assert int(states[row, b, 0]) == episode
            assert int(states[row, b, 1]) == expected_step
            assert int(states[row, b, 2]) == counter - (step - expected_step)
            assert bool(sample.obs_valid[row, b]) == (step - back >= first)
        assert bool(sample.obs_valid[-1, b])
        expected_prev = counter - 1 + 100.0 if step - 1 >= first else 0.0
        assert float(sample.prev_action[b, 0]) == expected_prev
    return seen


def test_history_stays_in_episode_and_replicates_first_frame():
    buffer = _buffer()
    writer = _Writer(buffer)
    writer.episode(0, 3)
    writer.episode(1, 6)
    torch.manual_seed(0)
    sample = buffer.sample_history(512, 4)
    assert sample.obs["state"].shape == (4, 512, 3)
    assert sample.prev_action.shape == (512, 1)
    assert sample.obs_valid.shape == (4, 512)

    seen = _check_history(sample, 4, lambda episode, step: 0)
    assert seen == set(range(6))
    # Start at the episode's step 2: the two oldest rows are the first row.
    at_step_two = (sample.obs["state"][-1, :, 1] == 2).nonzero(as_tuple=True)[0]
    assert at_step_two.numel() > 0
    for b in at_step_two.tolist():
        assert int(sample.obs["state"][0, b, 1]) == 0 and int(sample.obs["state"][1, b, 1]) == 0
        assert sample.obs_valid[:, b].tolist() == [False, True, True, True]


def test_history_does_not_read_overwritten_rows():
    buffer = _buffer(buffer_size=8)
    writer = _Writer(buffer)
    writer.episode(0, 10, ends=False)  # rows 0, 1 now hold steps 8, 9
    torch.manual_seed(0)
    sample = buffer.sample_history(512, 4)
    seen = _check_history(sample, 4, lambda episode, step: 2)
    assert seen == set(range(2, 10))


def test_history_len_one_and_rejects_cross_episode():
    buffer = _buffer()
    _Writer(buffer).episode(0, 4)
    sample = buffer.sample_history(32, 1)
    assert sample.obs["state"].shape == (1, 32, 3)
    assert bool(sample.obs_valid.all())
    with pytest.raises(ValueError, match="cross_episode=False"):
        StateSequenceReplayBuffer(
            _OBS_SPACE, _ACTION_SPACE, 1, 16, cross_episode=True, horizon=2, storage_device="cpu", sample_device="cpu"
        )


def test_window_sample_matches_sequence_replay_buffer():
    base = _buffer(SequenceReplayBuffer, num_envs=2, buffer_size=32)
    derived = _buffer(num_envs=2, buffer_size=32)
    for buffer in (base, derived):
        writer = _Writer(buffer, num_envs=2)
        writer.episode(0, 5)
        writer.episode(1, 7)
    torch.manual_seed(3)
    expected = base.sample(16)
    torch.manual_seed(3)
    actual = derived.sample(16)
    assert torch.equal(actual.obs["state"], expected.obs["state"])
    assert torch.equal(actual.action, expected.action)
    assert torch.equal(actual.reward, expected.reward)
    assert torch.equal(actual.terminated, expected.terminated)
