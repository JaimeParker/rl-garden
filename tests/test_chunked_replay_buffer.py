from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers.chunked_replay_buffer import ChunkedReplayBuffer
from rl_garden.buffers.h5_dataset import load_h5_dataset_to_replay_buffer

_DICT_OBS_SPACE = spaces.Dict(
    {
        "rgb_cam": spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
        "state": spaces.Box(-np.inf, np.inf, (1,), np.float32),
    }
)
_STATE_OBS_SPACE = spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (1,), np.float32)})


def _make_buffer(horizon_length: int = 3, gamma: float = 0.9, buffer_size: int = 10):
    return ChunkedReplayBuffer(
        observation_space=_DICT_OBS_SPACE,
        action_space=spaces.Box(-1.0, 1.0, (1,), np.float32),
        num_envs=1,
        buffer_size=buffer_size,
        horizon_length=horizon_length,
        gamma=gamma,
        storage_device="cpu",
        sample_device="cpu",
    )


def _make_state_buffer(horizon_length: int = 3, gamma: float = 0.9, buffer_size: int = 10):
    return ChunkedReplayBuffer(
        observation_space=_STATE_OBS_SPACE,
        action_space=spaces.Box(-1.0, 1.0, (1,), np.float32),
        num_envs=1,
        buffer_size=buffer_size,
        horizon_length=horizon_length,
        gamma=gamma,
        storage_device="cpu",
        sample_device="cpu",
    )


def _dict_obs(state_val: float) -> dict[str, torch.Tensor]:
    return {
        "rgb_cam": torch.randint(0, 256, (1, 8, 8, 3), dtype=torch.uint8),
        "state": torch.tensor([[state_val]], dtype=torch.float32),
    }


def _add_dict(rb, state_val, action_val, reward, done=False, episode_end=None):
    rb.add(
        _dict_obs(state_val),
        _dict_obs(state_val + 1.0),
        torch.tensor([[action_val]], dtype=torch.float32),
        torch.tensor([reward], dtype=torch.float32),
        torch.tensor([done]),
        None if episode_end is None else torch.tensor([episode_end]),
    )


def _add_state(rb, obs_val, action_val, reward, done=False, episode_end=None):
    rb.add(
        {"state": torch.tensor([[obs_val]], dtype=torch.float32)},
        {"state": torch.tensor([[obs_val + 1.0]], dtype=torch.float32)},
        torch.tensor([[action_val]], dtype=torch.float32),
        torch.tensor([reward], dtype=torch.float32),
        torch.tensor([done]),
        None if episode_end is None else torch.tensor([episode_end]),
    )


def test_chunk_accumulates_discounted_reward_no_terminal():
    rb = _make_state_buffer(horizon_length=3, gamma=0.9)
    _add_state(rb, 0.0, 10.0, 1.0)
    _add_state(rb, 1.0, 20.0, 2.0)
    _add_state(rb, 2.0, 30.0, 3.0)
    _add_state(rb, 3.0, 40.0, 4.0)

    batch_inds = torch.tensor([0])
    env_inds = torch.tensor([0])
    rewards, discounts, next_inds, action_chunk, valid = rb._accumulate_chunk(
        batch_inds, env_inds
    )

    expected_reward = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0
    assert torch.allclose(rewards, torch.tensor([expected_reward]))
    assert torch.allclose(discounts, torch.tensor([0.9**3]))
    assert next_inds.item() == 2
    assert torch.equal(action_chunk[:, 0, 0], torch.tensor([10.0, 20.0, 30.0]))
    assert torch.equal(valid[:, 0], torch.tensor([True, True, True]))


def test_chunk_stops_reward_accumulation_at_true_terminal():
    rb = _make_state_buffer(horizon_length=3, gamma=0.9)
    _add_state(rb, 0.0, 10.0, 1.0)
    _add_state(rb, 1.0, 20.0, 2.0, done=True)
    _add_state(rb, 2.0, 30.0, 100.0)

    rewards, discounts, next_inds, action_chunk, valid = rb._accumulate_chunk(
        torch.tensor([0]), torch.tensor([0])
    )

    assert torch.allclose(rewards, torch.tensor([1.0 + 0.9 * 2.0]))
    assert torch.allclose(discounts, torch.tensor([0.0]))
    assert next_inds.item() == 1
    # action_chunk still gathers the full raw window (terminal-agnostic),
    # matching QC's own unconditional action-window gather.
    assert torch.equal(action_chunk[:, 0, 0], torch.tensor([10.0, 20.0, 30.0]))
    # valid[i] = "was position i reached before any true terminal" -- the
    # terminal position itself is still valid, positions after are not.
    assert torch.equal(valid[:, 0], torch.tensor([True, True, False]))


def test_chunk_truncation_keeps_bootstrap_discount_but_stops_reward():
    rb = _make_state_buffer(horizon_length=3, gamma=0.9)
    _add_state(rb, 0.0, 10.0, 1.0)
    _add_state(rb, 1.0, 20.0, 2.0, done=False, episode_end=True)
    _add_state(rb, 2.0, 30.0, 100.0)

    rewards, discounts, next_inds, action_chunk, valid = rb._accumulate_chunk(
        torch.tensor([0]), torch.tensor([0])
    )

    assert torch.allclose(rewards, torch.tensor([1.0 + 0.9 * 2.0]))
    assert torch.allclose(discounts, torch.tensor([0.9**2]))
    assert next_inds.item() == 1
    assert torch.equal(valid[:, 0], torch.tensor([True, True, False]))


def test_early_terminating_window_is_still_sampleable_not_discarded():
    # Deliberate deviation from QC's own `valid[...,-1]`-discards-whole-window
    # convention (see chunked_replay_buffer.py's module docstring): a
    # window that terminates strictly before the last position is still a
    # valid, unbiased partial-length sample here, not rejected/masked to zero.
    rb = _make_state_buffer(horizon_length=3, gamma=0.9, buffer_size=20)
    for i in range(6):
        _add_state(rb, float(i), float(i) * 10.0, 1.0, done=(i == 1))
    valid = rb._valid_nstep_batch(torch.tensor([0]), torch.tensor([0]))
    assert bool(valid.item())


def test_sample_returns_correctly_shaped_dict_obs_batch():
    rb = _make_buffer(horizon_length=3, gamma=0.9, buffer_size=20)
    for i in range(8):
        _add_dict(rb, float(i), float(i) * 10.0, 1.0)

    sample = rb.sample(4)
    assert isinstance(sample.obs, dict)
    assert sample.obs["rgb_cam"].shape == (4, 8, 8, 3)
    assert sample.obs["rgb_cam"].dtype == torch.uint8
    assert sample.obs["state"].shape == (4, 1)
    assert isinstance(sample.next_obs, dict)
    assert sample.next_obs["rgb_cam"].shape == (4, 8, 8, 3)
    assert sample.next_obs["state"].shape == (4, 1)
    assert sample.actions.shape == (4, 3, 1)
    assert sample.rewards.shape == (4,)
    assert sample.discounts.shape == (4,)
    assert sample.dones.shape == (4,)
    assert sample.valid.shape == (4, 3)
    assert torch.isfinite(sample.obs["state"]).all()


def test_add_without_episode_end_falls_back_to_done():
    rb = _make_buffer(horizon_length=2, gamma=0.9, buffer_size=10)
    _add_dict(rb, 0.0, 1.0, 1.0, done=True, episode_end=None)
    assert bool(rb.episode_ends[0, 0].item()) is True


def test_h5_loader_fills_chunked_buffer():
    """Pre-implementation check from the plan: chunked + Dict + H5-loaded
    had never existed in this repo before ChunkedReplayBuffer. Confirms
    _add_flat_transitions/_slice's existing recursive-dict handling (already
    proven for MCReplayBuffer, see test_h5_dataset_loader.py's own
    test_load_dict_h5_to_dict_replay_buffer) also fills this buffer
    end-to-end, not just constructed-buffer unit tests."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo_dict.h5"
        with h5py.File(path, "w") as f:
            group = f.create_group("traj_0")
            obs = group.create_group("obs")
            obs.create_dataset("state", data=np.ones((5, 1), dtype=np.float32))
            obs.create_dataset("rgb_cam", data=np.ones((5, 8, 8, 3), dtype=np.uint8))
            group.create_dataset("actions", data=np.ones((4, 1), dtype=np.float32))
            group.create_dataset("rewards", data=np.ones(4, dtype=np.float32))
            group.create_dataset("dones", data=np.array([False, False, False, True]))

        rb = ChunkedReplayBuffer(
            observation_space=_DICT_OBS_SPACE,
            action_space=spaces.Box(-1.0, 1.0, (1,), np.float32),
            num_envs=2,
            buffer_size=10,
            horizon_length=2,
            gamma=0.9,
            storage_device="cpu",
            sample_device="cpu",
        )
        loaded = load_h5_dataset_to_replay_buffer(rb, path)
        assert loaded == 4
        sample = rb.sample(4)
        assert sample.obs["state"].shape == (4, 1)
        assert sample.obs["rgb_cam"].shape == (4, 8, 8, 3)
        assert sample.obs["rgb_cam"].dtype == torch.uint8
        assert torch.all(sample.rewards > 0)
