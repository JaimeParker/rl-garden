from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers.chunked_dict_replay_buffer import ChunkedDictReplayBuffer
from rl_garden.buffers.chunked_replay_buffer import ChunkedTensorReplayBuffer
from rl_garden.buffers.h5_dataset import load_h5_dataset_to_replay_buffer

_DICT_OBS_SPACE = spaces.Dict(
    {
        "rgb": spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
        "state": spaces.Box(-np.inf, np.inf, (1,), np.float32),
    }
)


def _make_dict_buffer(horizon_length: int = 3, gamma: float = 0.9, buffer_size: int = 10):
    return ChunkedDictReplayBuffer(
        observation_space=_DICT_OBS_SPACE,
        action_space=spaces.Box(-1.0, 1.0, (1,), np.float32),
        num_envs=1,
        buffer_size=buffer_size,
        horizon_length=horizon_length,
        gamma=gamma,
        storage_device="cpu",
        sample_device="cpu",
    )


def _make_tensor_buffer(horizon_length: int = 3, gamma: float = 0.9, buffer_size: int = 10):
    return ChunkedTensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (1,), np.float32),
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
        "rgb": torch.randint(0, 256, (1, 8, 8, 3), dtype=torch.uint8),
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


def _add_tensor(rb, obs_val, action_val, reward, done=False, episode_end=None):
    rb.add(
        torch.tensor([[obs_val]], dtype=torch.float32),
        torch.tensor([[obs_val + 1.0]], dtype=torch.float32),
        torch.tensor([[action_val]], dtype=torch.float32),
        torch.tensor([reward], dtype=torch.float32),
        torch.tensor([done]),
        None if episode_end is None else torch.tensor([episode_end]),
    )


def test_accumulate_chunk_matches_tensor_buffer_no_terminal():
    """Parity check: ChunkedDictReplayBuffer._accumulate_chunk is a verbatim
    copy of ChunkedTensorReplayBuffer's (never touches obs/next_obs) -- given
    identical action/reward/done sequences, every non-obs output must match
    exactly."""
    dict_rb = _make_dict_buffer(horizon_length=3, gamma=0.9)
    tensor_rb = _make_tensor_buffer(horizon_length=3, gamma=0.9)
    for state_val, action_val, reward in [(0.0, 10.0, 1.0), (1.0, 20.0, 2.0), (2.0, 30.0, 3.0), (3.0, 40.0, 4.0)]:
        _add_dict(dict_rb, state_val, action_val, reward)
        _add_tensor(tensor_rb, state_val, action_val, reward)

    batch_inds = torch.tensor([0])
    env_inds = torch.tensor([0])
    d_rewards, d_discounts, d_next_inds, d_action_chunk, d_valid = dict_rb._accumulate_chunk(
        batch_inds, env_inds
    )
    t_rewards, t_discounts, t_next_inds, t_action_chunk, t_valid = tensor_rb._accumulate_chunk(
        batch_inds, env_inds
    )

    assert torch.allclose(d_rewards, t_rewards)
    assert torch.allclose(d_discounts, t_discounts)
    assert torch.equal(d_next_inds, t_next_inds)
    assert torch.equal(d_action_chunk, t_action_chunk)
    assert torch.equal(d_valid, t_valid)


def test_accumulate_chunk_matches_tensor_buffer_with_terminal():
    dict_rb = _make_dict_buffer(horizon_length=3, gamma=0.9)
    tensor_rb = _make_tensor_buffer(horizon_length=3, gamma=0.9)
    for state_val, action_val, reward, done in [
        (0.0, 10.0, 1.0, False),
        (1.0, 20.0, 2.0, True),
        (2.0, 30.0, 100.0, False),
    ]:
        _add_dict(dict_rb, state_val, action_val, reward, done=done)
        _add_tensor(tensor_rb, state_val, action_val, reward, done=done)

    batch_inds = torch.tensor([0])
    env_inds = torch.tensor([0])
    d_out = dict_rb._accumulate_chunk(batch_inds, env_inds)
    t_out = tensor_rb._accumulate_chunk(batch_inds, env_inds)
    for d, t in zip(d_out, t_out):
        assert torch.equal(d, t) if d.dtype == torch.bool else torch.allclose(d, t)


def test_sample_returns_correctly_shaped_dict_obs_batch():
    rb = _make_dict_buffer(horizon_length=3, gamma=0.9, buffer_size=20)
    for i in range(8):
        _add_dict(rb, float(i), float(i) * 10.0, 1.0)

    sample = rb.sample(4)
    assert isinstance(sample.obs, dict)
    assert sample.obs["rgb"].shape == (4, 8, 8, 3)
    assert sample.obs["rgb"].dtype == torch.uint8
    assert sample.obs["state"].shape == (4, 1)
    assert isinstance(sample.next_obs, dict)
    assert sample.next_obs["rgb"].shape == (4, 8, 8, 3)
    assert sample.next_obs["state"].shape == (4, 1)
    assert sample.actions.shape == (4, 3, 1)
    assert sample.rewards.shape == (4,)
    assert sample.discounts.shape == (4,)
    assert sample.dones.shape == (4,)
    assert sample.valid.shape == (4, 3)
    assert torch.isfinite(sample.obs["state"]).all()


def test_add_without_episode_end_falls_back_to_done():
    rb = _make_dict_buffer(horizon_length=2, gamma=0.9, buffer_size=10)
    _add_dict(rb, 0.0, 1.0, 1.0, done=True, episode_end=None)
    assert bool(rb.episode_ends[0, 0].item()) is True


def test_h5_loader_fills_chunked_dict_buffer():
    """Pre-implementation check from the plan: chunked + Dict + H5-loaded
    had never existed in this repo before ChunkedDictReplayBuffer. Confirms
    _add_flat_transitions/_slice's existing recursive-dict handling (already
    proven for MCDictReplayBuffer, see test_h5_dataset_loader.py's own
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
            obs.create_dataset("rgb", data=np.ones((5, 8, 8, 3), dtype=np.uint8))
            group.create_dataset("actions", data=np.ones((4, 1), dtype=np.float32))
            group.create_dataset("rewards", data=np.ones(4, dtype=np.float32))
            group.create_dataset("dones", data=np.array([False, False, False, True]))

        rb = ChunkedDictReplayBuffer(
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
        assert sample.obs["rgb"].shape == (4, 8, 8, 3)
        assert sample.obs["rgb"].dtype == torch.uint8
        assert torch.all(sample.rewards > 0)
