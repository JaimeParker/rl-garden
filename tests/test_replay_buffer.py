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
