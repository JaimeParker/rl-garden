"""Unit tests for rl_garden replay buffers.

Runs on CPU so it's safe in CI without a GPU.
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers import DictReplayBuffer, TensorReplayBuffer


def test_tensor_replay_buffer_add_and_sample():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    num_envs, buffer_size = 4, 32
    device = torch.device("cpu")

    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=num_envs, buffer_size=buffer_size,
        storage_device=device, sample_device=device,
    )
    assert rb.per_env_buffer_size == buffer_size // num_envs

    for _ in range(3):
        rb.add(
            obs=torch.randn(num_envs, 7),
            next_obs=torch.randn(num_envs, 7),
            action=torch.randn(num_envs, 3),
            reward=torch.randn(num_envs),
            done=torch.zeros(num_envs),
        )

    assert rb.pos == 3 and not rb.full
    assert len(rb) == 3 * num_envs

    batch = rb.sample(batch_size=16)
    assert batch.obs.shape == (16, 7)
    assert batch.next_obs.shape == (16, 7)
    assert batch.actions.shape == (16, 3)
    assert batch.rewards.shape == (16,)
    assert batch.dones.shape == (16,)
    assert batch.obs.device == device


def test_tensor_replay_buffer_wraps_when_full():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    num_envs, buffer_size = 2, 8  # per_env = 4
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=num_envs, buffer_size=buffer_size,
        storage_device="cpu", sample_device="cpu",
    )
    for i in range(5):
        rb.add(
            obs=torch.full((num_envs, 2), float(i)),
            next_obs=torch.full((num_envs, 2), float(i + 1)),
            action=torch.full((num_envs, 1), float(i)),
            reward=torch.full((num_envs,), float(i)),
            done=torch.zeros(num_envs),
        )
    assert rb.full and rb.pos == 1


def test_dict_replay_buffer_add_and_sample():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    num_envs, buffer_size = 2, 8
    device = torch.device("cpu")

    rb = DictReplayBuffer(
        obs_space, act_space, num_envs=num_envs, buffer_size=buffer_size,
        storage_device=device, sample_device=device,
    )

    def make_obs():
        return {
            "rgb": torch.randint(0, 256, (num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(num_envs, 5),
        }

    for _ in range(3):
        rb.add(
            obs=make_obs(),
            next_obs=make_obs(),
            action=torch.randn(num_envs, 4),
            reward=torch.randn(num_envs),
            done=torch.zeros(num_envs),
        )

    batch = rb.sample(batch_size=5)
    assert isinstance(batch.obs, dict)
    assert batch.obs["rgb"].shape == (5, 64, 64, 3)
    assert batch.obs["rgb"].dtype == torch.uint8
    assert batch.obs["state"].shape == (5, 5)
    assert batch.next_obs["rgb"].shape == (5, 64, 64, 3)
    assert batch.actions.shape == (5, 4)
    assert batch.rewards.shape == (5,)
    assert batch.dones.shape == (5,)


# ----------------------------------------------------------------------------
# sample_without_repeat
# ----------------------------------------------------------------------------


def _fill_tensor_buffer(buf, num_steps):
    obs_dim = buf.obs.shape[-1]
    act_dim = buf.actions.shape[-1]
    n = buf.num_envs
    for step in range(num_steps):
        buf.add(
            obs=torch.full((n, obs_dim), float(step)),
            next_obs=torch.full((n, obs_dim), float(step + 1)),
            action=torch.zeros(n, act_dim),
            reward=torch.full((n,), float(step)),
            done=torch.zeros(n),
        )


def test_sample_without_repeat_no_duplicates_within_epoch():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=4, buffer_size=20,
        storage_device="cpu", sample_device="cpu",
    )
    _fill_tensor_buffer(rb, num_steps=5)  # 5 * 4 = 20 transitions
def test_sample_without_repeat_visits_all_indices_in_epoch():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=4, buffer_size=20,
        storage_device="cpu", sample_device="cpu",
    )
    # Fill with per-env distinct values so we can identify each transition
    for step in range(5):
        rb.add(
            obs=torch.tensor([[step, e] for e in range(4)], dtype=torch.float32),
            next_obs=torch.zeros(4, 2),
            action=torch.zeros(4, 1),
            reward=torch.zeros(4),
            done=torch.zeros(4),
        )
    seen = set()
    epoch = rb.epoch_size
    batch_size = 5
    for _ in range(epoch // batch_size):
        sample = rb.sample_without_repeat(batch_size)
        for o in sample.obs:
            seen.add((int(o[0].item()), int(o[1].item())))
    assert len(seen) == 20  # all distinct (t, env) pairs visited


def test_sample_without_repeat_reshuffles_after_exhaustion():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=2, buffer_size=8,
        storage_device="cpu", sample_device="cpu",
    )
    _fill_tensor_buffer(rb, num_steps=4)  # 8 transitions
    # Exhaust epoch: 8/2 = 4 batches
    for _ in range(4):
        rb.sample_without_repeat(2)
    # Next sample triggers reshuffle (not a new add); should still return valid sample
    sample = rb.sample_without_repeat(2)
    assert sample.obs.shape == (2, 2)


def test_sample_without_repeat_invalidates_after_pos_change():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=2, buffer_size=20,
        storage_device="cpu", sample_device="cpu",
    )
    _fill_tensor_buffer(rb, num_steps=5)  # 10 transitions
    sample1 = rb.sample_without_repeat(2)
    pos_before = rb.pos
    # Add more data → pos changes → permutation should rebuild on next call
    _fill_tensor_buffer(rb, num_steps=2)
    assert rb.pos != pos_before
    sample2 = rb.sample_without_repeat(2)
    assert sample2.obs.shape == (2, 2)


def test_sample_without_repeat_empty_buffer_raises():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=2, buffer_size=10,
        storage_device="cpu", sample_device="cpu",
    )
    import pytest
    with pytest.raises(RuntimeError, match="empty"):
        rb.sample_without_repeat(1)


def test_dict_buffer_sample_without_repeat():
    obs_space = spaces.Dict({
        "state": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
    })
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    rb = DictReplayBuffer(
        obs_space, act_space, num_envs=2, buffer_size=8,
        storage_device="cpu", sample_device="cpu",
    )
    for step in range(4):
        rb.add(
            obs={"state": torch.full((2, 3), float(step))},
            next_obs={"state": torch.zeros(2, 3)},
            action=torch.zeros(2, 2),
            reward=torch.zeros(2),
            done=torch.zeros(2),
        )
    sample = rb.sample_without_repeat(4)
    assert isinstance(sample.obs, dict)
    assert sample.obs["state"].shape == (4, 3)
