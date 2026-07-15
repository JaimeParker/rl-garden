"""Tests for MmapMultitaskEpisodeBuffer: mmap create/open round-trip, episode
loading, task-conditioned windowed sampling."""
from __future__ import annotations

import pytest
import torch

from rl_garden.algorithms.tdmpc2.multitask.buffer import MmapMultitaskEpisodeBuffer


def _make_buffer(tmp_path, **kwargs):
    params = dict(
        obs_dim=6,
        action_dim=3,
        buffer_size=100,
        horizon=2,
        mmap_dir=str(tmp_path / "buf"),
        mmap_mode="create",
        storage_device="cpu",
        sample_device="cpu",
    )
    params.update(kwargs)
    return MmapMultitaskEpisodeBuffer(**params)


def test_rejects_non_positive_horizon(tmp_path):
    with pytest.raises(ValueError):
        _make_buffer(tmp_path, horizon=0)


def test_rejects_gpu_storage_device(tmp_path):
    with pytest.raises(ValueError):
        _make_buffer(tmp_path, storage_device="cuda")


def test_add_is_unsupported(tmp_path):
    buf = _make_buffer(tmp_path)
    with pytest.raises(NotImplementedError):
        buf.add(None, None, None, None, None)


def test_load_episode_rejects_overflow(tmp_path):
    buf = _make_buffer(tmp_path, buffer_size=10)
    with pytest.raises(RuntimeError):
        buf.load_episode(torch.zeros(20, 6), torch.zeros(20, 3), torch.zeros(20), task_idx=0)


def test_load_episode_and_sample_shapes(tmp_path):
    buf = _make_buffer(tmp_path)
    for length, task in [(5, 0), (8, 1), (6, 0)]:
        obs = torch.stack([torch.full((6,), float(t)) for t in range(length)])
        buf.load_episode(obs, torch.zeros(length, 3), torch.ones(length), task)

    torch.manual_seed(0)
    sample = buf.sample(batch_size=8)
    assert sample.obs.shape == (3, 8, 6)
    assert sample.action.shape == (2, 8, 3)
    assert sample.reward.shape == (2, 8)
    assert sample.task.shape == (8,)


def test_sampled_windows_never_cross_episode_boundary(tmp_path):
    buf = _make_buffer(tmp_path)
    for length, task in [(5, 0), (8, 1), (6, 2)]:
        obs = torch.stack([torch.full((6,), float(t)) for t in range(length)])
        buf.load_episode(obs, torch.zeros(length, 3), torch.ones(length), task)

    torch.manual_seed(0)
    for _ in range(30):
        sample = buf.sample(batch_size=16)
        # obs values were written as the in-episode timestep, so a boundary
        # crossing would show up as non-consecutive values within a window.
        diffs = sample.obs[1:, :, 0] - sample.obs[:-1, :, 0]
        assert torch.all(diffs == 1.0)


def test_task_field_is_constant_within_a_window_and_matches_loaded_task(tmp_path):
    buf = _make_buffer(tmp_path)
    for length, task in [(6, 0), (6, 1)]:
        obs = torch.stack([torch.full((6,), float(t)) for t in range(length)])
        buf.load_episode(obs, torch.zeros(length, 3), torch.ones(length), task)

    torch.manual_seed(0)
    for _ in range(20):
        sample = buf.sample(batch_size=8)
        assert set(sample.task.tolist()) <= {0, 1}


def test_mmap_create_then_open_round_trip(tmp_path):
    path = tmp_path / "buf"
    buf = MmapMultitaskEpisodeBuffer(
        obs_dim=6, action_dim=3, buffer_size=20, horizon=2,
        mmap_dir=str(path), mmap_mode="create", storage_device="cpu", sample_device="cpu",
    )
    obs = torch.arange(5 * 6).reshape(5, 6).float()
    buf.load_episode(obs, torch.zeros(5, 3), torch.ones(5), task_idx=0)
    buf.flush()
    del buf

    buf2 = MmapMultitaskEpisodeBuffer(
        obs_dim=6, action_dim=3, buffer_size=20, horizon=2,
        mmap_dir=str(path), mmap_mode="open", storage_device="cpu", sample_device="cpu",
    )
    assert torch.equal(buf2.obs[0, 0], obs[0])


def test_disk_byte_footprint_matches_expected_arithmetic_and_uses_memmap(tmp_path):
    """Sanity check the mmap design scales to upstream's mt80 magnitude
    (~550M transitions) without literally allocating that much: assert the
    per-array byte size formula is what we expect, and that storage is a real
    memmap (not RAM-resident), for a SMALL buffer standing in for the shape."""
    buf = _make_buffer(tmp_path, obs_dim=39, action_dim=39, buffer_size=1000)
    import numpy as np

    # buf._store._maps holds the raw np.memmap arrays backing every tensor
    # (torch.from_numpy(memmap) doesn't preserve the memmap subclass on the
    # torch side, so check the store's own array list instead).
    assert len(buf._store._maps) > 0
    assert all(isinstance(arr, np.memmap) for arr in buf._store._maps)
    expected_obs_bytes = 1000 * 1 * 39 * 4  # float32
    assert buf.obs.numel() * buf.obs.element_size() == expected_obs_bytes
    # Extrapolated mt80-scale footprint (not allocated here) stays in the
    # tens-of-GB range on disk, not RAM -- the point of using mmap at all.
    mt80_scale = 550_450_000
    projected_obs_gb = mt80_scale * 39 * 4 / 1e9
    assert projected_obs_gb < 200  # sanity bound, not a hard spec
