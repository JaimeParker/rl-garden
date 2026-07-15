"""Tests for tools/conversion/convert_tdmpc2_multitask_dataset.py.

``test_reindex_*`` tests the pure reindexing logic directly (no tensordict
needed). ``test_convert_task_file_round_trip`` exercises the full
tensordict-dependent ``torch.load`` path against a SYNTHETIC file built to
our own assumed schema -- this proves the converter's internal logic is
correct and self-consistent, NOT that it matches TD-MPC2's real released
mt30/mt80 files (untested -- see the script's module docstring for the
known-risk callout). Skipped entirely if ``tensordict`` isn't installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "conversion"))

from convert_tdmpc2_multitask_dataset import (  # noqa: E402
    convert_dataset,
    reindex_episode_tensors,
)


def test_reindex_shifts_action_reward_by_one_and_drops_final_obs():
    num_eps, window_len, obs_dim, action_dim = 3, 6, 4, 2
    obs = torch.arange(num_eps * window_len * obs_dim).reshape(num_eps, window_len, obs_dim).float()
    action = torch.arange(num_eps * window_len * action_dim).reshape(num_eps, window_len, action_dim).float()
    reward = torch.arange(num_eps * window_len).reshape(num_eps, window_len).float()

    my_obs, my_action, my_reward = reindex_episode_tensors(obs, action, reward)

    assert my_obs.shape == (num_eps, window_len - 1, obs_dim)
    assert my_action.shape == (num_eps, window_len - 1, action_dim)
    assert my_reward.shape == (num_eps, window_len - 1)
    assert torch.equal(my_obs, obs[:, : window_len - 1])
    assert torch.equal(my_action, action[:, 1:])
    assert torch.equal(my_reward, reward[:, 1:])


def test_reindex_rejects_too_short_window():
    with pytest.raises(ValueError):
        reindex_episode_tensors(torch.zeros(2, 1, 4), torch.zeros(2, 1, 2), torch.zeros(2, 1))


def test_convert_task_file_round_trip(tmp_path):
    tensordict = pytest.importorskip("tensordict")

    num_eps, window_len, obs_dim, action_dim = 4, 7, 5, 3
    obs = torch.randn(num_eps, window_len, obs_dim)
    action = torch.randn(num_eps, window_len, action_dim)
    reward = torch.randn(num_eps, window_len)

    td = tensordict.TensorDict(
        {"obs": obs, "action": action, "reward": reward},
        batch_size=(num_eps, window_len),
    )
    src_dir = tmp_path / "official"
    src_dir.mkdir()
    torch.save(td, src_dir / "fake-task.pt")

    dst_dir = tmp_path / "converted"
    manifest = convert_dataset(src_dir, dst_dir, ["fake-task"])

    assert manifest["tasks"] == ["fake-task"]
    assert manifest["obs_dims"] == [obs_dim]
    assert manifest["action_dims"] == [action_dim]
    assert manifest["episode_lengths"] == [window_len - 1]

    loaded = torch.load(dst_dir / "fake-task" / "episodes.pt", weights_only=True)
    assert torch.equal(loaded["obs"], obs[:, : window_len - 1])
    assert torch.equal(loaded["action"], action[:, 1:])
    assert torch.equal(loaded["reward"], reward[:, 1:])

    from rl_garden.algorithms.tdmpc2.multitask.dataset import (
        infer_multitask_dataset_specs,
        load_multitask_dataset,
    )
    from rl_garden.algorithms.tdmpc2.multitask.buffer import MmapMultitaskEpisodeBuffer

    tasks, obs_dims, action_dims, episode_lengths = infer_multitask_dataset_specs(dst_dir)
    assert tasks == ["fake-task"]
    buf = MmapMultitaskEpisodeBuffer(
        obs_dim=max(obs_dims), action_dim=max(action_dims), buffer_size=1000, horizon=2,
        mmap_dir=str(tmp_path / "mmapbuf"), mmap_mode="create", storage_device="cpu", sample_device="cpu",
    )
    total = load_multitask_dataset(buf, dst_dir)
    assert total == num_eps * (window_len - 1)
