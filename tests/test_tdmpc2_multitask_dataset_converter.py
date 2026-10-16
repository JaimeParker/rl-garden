"""Tests for tools/conversion/convert_tdmpc2_multitask_dataset.py.

Synthetic chunks mirror the real mt30 schema validated on 2026-10-15 (see the
script's module docstring): mixed-task ``chunk_*.pt`` files keyed
``task``/``obs``/``action``/``reward``, obs/action zero-padded to the task-set
max dims, garbage action row 0 and NaN reward row 0. The stub-unpickler path
is tested without ``tensordict`` by pickling stand-in classes that claim the
real tensordict module/class names; the real-tensordict round trip is
skipped when ``tensordict`` isn't installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "conversion"))

from convert_tdmpc2_multitask_dataset import (  # noqa: E402
    MT30_TASKS,
    convert_dataset,
    infer_unpadded_dim,
    load_chunk_without_tensordict,
    reindex_episode_tensors,
    split_chunk_by_task,
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


def _synthetic_chunk(task_ids, task_dims, window_len=7, max_obs_dim=5, max_action_dim=3, seed=0):
    """Mixed-task chunk with the real file's quirks: padded dims, garbage
    action row 0, NaN reward row 0."""
    gen = torch.Generator().manual_seed(seed)
    num_eps = len(task_ids)
    obs = torch.zeros(num_eps, window_len, max_obs_dim)
    action = torch.zeros(num_eps, window_len, max_action_dim)
    for ep, task_id in enumerate(task_ids):
        obs_dim, action_dim = task_dims[task_id]
        obs[ep, :, :obs_dim] = torch.randn(window_len, obs_dim, generator=gen)
        action[ep, :, :action_dim] = torch.rand(window_len, action_dim, generator=gen) * 2 - 1
    action[:, 0] = 1e11  # uninitialized-memory placeholder
    reward = torch.rand(num_eps, window_len, generator=gen)
    reward[:, 0] = float("nan")
    task = torch.tensor(task_ids, dtype=torch.int32)[:, None].expand(num_eps, window_len).contiguous()
    return {"task": task, "obs": obs, "action": action, "reward": reward}


def test_split_chunk_groups_by_task_and_drops_placeholders():
    chunk = _synthetic_chunk([2, 0, 2, 1], {0: (5, 3), 1: (2, 1), 2: (4, 2)})
    grouped = split_chunk_by_task(chunk)

    assert sorted(grouped) == [0, 1, 2]
    obs, action, reward = grouped[2]
    assert obs.shape == (2, 6, 5) and action.shape == (2, 6, 3) and reward.shape == (2, 6)
    assert torch.equal(obs, chunk["obs"][[0, 2], :6])
    assert torch.equal(action, chunk["action"][[0, 2], 1:])
    assert torch.equal(reward, chunk["reward"][[0, 2], 1:])


def test_split_chunk_rejects_task_change_within_episode():
    chunk = _synthetic_chunk([0, 1], {0: (5, 3), 1: (5, 3)})
    chunk["task"][0, 3] = 1
    with pytest.raises(ValueError, match="within an episode"):
        split_chunk_by_task(chunk)


def test_split_chunk_rejects_nan_outside_row_zero():
    chunk = _synthetic_chunk([0, 1], {0: (5, 3), 1: (5, 3)})
    chunk["reward"][1, 4] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        split_chunk_by_task(chunk)


def test_infer_unpadded_dim():
    padded = torch.zeros(3, 4, 6)
    padded[..., :4] = 1.0
    padded[0, 0, 2] = 0.0
    assert infer_unpadded_dim(padded) == 4
    with pytest.raises(ValueError):
        infer_unpadded_dim(torch.zeros(2, 3))


class _FakeTensorDict:
    def __init__(self, tensors, batch_size):
        self._tensordict = _FakeStringOnlyDict(tensors)
        self._batch_size = torch.Size(batch_size)
        self._device = torch.device("cpu")


class _FakeStringOnlyDict(dict):
    pass


def _save_fake_tensordict(path, tensors, monkeypatch):
    """Pickle a stand-in under the real tensordict module/class names
    (``GLOBAL 'tensordict.tensordict TensorDict'`` + ``NEWOBJ``/``BUILD``,
    the opcode pattern of the official files)."""
    import types

    td_module = types.ModuleType("tensordict.tensordict")
    utils_module = types.ModuleType("tensordict.utils")
    _FakeTensorDict.__module__, _FakeTensorDict.__qualname__ = "tensordict.tensordict", "TensorDict"
    _FakeStringOnlyDict.__module__, _FakeStringOnlyDict.__qualname__ = "tensordict.utils", "_StringOnlyDict"
    td_module.TensorDict = _FakeTensorDict
    utils_module._StringOnlyDict = _FakeStringOnlyDict
    monkeypatch.setitem(sys.modules, "tensordict", types.ModuleType("tensordict"))
    monkeypatch.setitem(sys.modules, "tensordict.tensordict", td_module)
    monkeypatch.setitem(sys.modules, "tensordict.utils", utils_module)
    torch.save(_FakeTensorDict(tensors, tensors["reward"].shape), path)


def test_stub_unpickler_reads_tensordict_pickle_without_tensordict(tmp_path, monkeypatch):
    chunk = _synthetic_chunk([0, 1], {0: (5, 3), 1: (2, 1)})
    path = tmp_path / "chunk_0.pt"
    with monkeypatch.context() as patch:
        _save_fake_tensordict(path, chunk, patch)

    loaded = load_chunk_without_tensordict(path)
    assert sorted(loaded) == sorted(chunk)
    for key in chunk:
        torch.testing.assert_close(loaded[key], chunk[key], rtol=0, atol=0, equal_nan=True)


def _write_chunks(src_dir, chunks, monkeypatch):
    src_dir.mkdir()
    for i, chunk in enumerate(chunks):
        try:
            import tensordict

            td = tensordict.TensorDict(chunk, batch_size=chunk["reward"].shape)
            torch.save(td, src_dir / f"chunk_{i}.pt")
        except ImportError:
            with monkeypatch.context() as patch:
                _save_fake_tensordict(src_dir / f"chunk_{i}.pt", chunk, patch)


def test_convert_dataset_merges_chunks_crops_padding_and_loads(tmp_path, monkeypatch):
    task_dims = {0: (5, 3), 1: (2, 1), 3: (4, 2)}
    chunks = [
        _synthetic_chunk([3, 0, 1], task_dims, seed=0),
        _synthetic_chunk([1, 3], task_dims, seed=1),
    ]
    src_dir = tmp_path / "official"
    _write_chunks(src_dir, chunks, monkeypatch)
    dst_dir = tmp_path / "converted"

    with pytest.raises(ValueError, match="tasks missing"):
        convert_dataset(src_dir, dst_dir)
    manifest = convert_dataset(src_dir, dst_dir, allow_missing_tasks=True)

    assert manifest["tasks"] == [MT30_TASKS[0], MT30_TASKS[1], MT30_TASKS[3]]
    assert manifest["obs_dims"] == [5, 2, 4]
    assert manifest["action_dims"] == [3, 1, 2]
    assert manifest["episode_lengths"] == [6, 6, 6]
    assert manifest["num_episodes"] == [1, 2, 2]

    loaded = torch.load(dst_dir / MT30_TASKS[3] / "episodes.pt", weights_only=True)
    expected_obs = torch.cat([chunks[0]["obs"][[0], :6, :4], chunks[1]["obs"][[1], :6, :4]])
    expected_action = torch.cat([chunks[0]["action"][[0], 1:, :2], chunks[1]["action"][[1], 1:, :2]])
    assert torch.equal(loaded["obs"], expected_obs)
    assert torch.equal(loaded["action"], expected_action)

    from rl_garden.buffers.mmap_multitask_episode_buffer import MmapMultitaskEpisodeBuffer
    from rl_garden.buffers.tdmpc2_multitask_dataset import (
        infer_multitask_dataset_specs,
        load_multitask_dataset,
    )

    tasks, obs_dims, action_dims, _ = infer_multitask_dataset_specs(dst_dir)
    assert tasks == manifest["tasks"]
    buf = MmapMultitaskEpisodeBuffer(
        obs_dim=max(obs_dims), action_dim=max(action_dims), buffer_size=1000, horizon=2,
        mmap_dir=str(tmp_path / "mmapbuf"), mmap_mode="create", storage_device="cpu", sample_device="cpu",
    )
    assert load_multitask_dataset(buf, dst_dir) == 5 * 6


def test_load_multitask_dataset_task_subset_and_episode_cap_keep_manifest_indices(tmp_path):
    import json

    from rl_garden.buffers.mmap_multitask_episode_buffer import MmapMultitaskEpisodeBuffer
    from rl_garden.buffers.tdmpc2_multitask_dataset import load_multitask_dataset

    tasks, obs_dims, action_dims, num_episodes, episode_length = ["a", "b", "c"], [3, 2, 4], [1, 2, 2], [3, 2, 4], 5
    dataset_dir = tmp_path / "converted"
    for task, obs_dim, action_dim, n in zip(tasks, obs_dims, action_dims, num_episodes):
        (dataset_dir / task).mkdir(parents=True)
        torch.save(
            {
                "obs": torch.randn(n, episode_length, obs_dim),
                "action": torch.randn(n, episode_length, action_dim),
                "reward": torch.arange(n, dtype=torch.float32)[:, None].expand(n, episode_length).clone(),
            },
            dataset_dir / task / "episodes.pt",
        )
    (dataset_dir / "manifest.json").write_text(json.dumps({
        "tasks": tasks, "obs_dims": obs_dims, "action_dims": action_dims,
        "episode_lengths": [episode_length] * 3, "num_episodes": num_episodes,
    }))

    def make_buffer(name):
        return MmapMultitaskEpisodeBuffer(
            obs_dim=max(obs_dims), action_dim=max(action_dims), buffer_size=100, horizon=2,
            mmap_dir=str(tmp_path / name), mmap_mode="create", storage_device="cpu", sample_device="cpu",
        )

    assert load_multitask_dataset(make_buffer("all"), dataset_dir) == sum(num_episodes) * episode_length

    buf = make_buffer("subset")
    loaded = load_multitask_dataset(buf, dataset_dir, tasks=("c", "b"), max_episodes_per_task=2)
    assert loaded == 4 * episode_length
    stored_tasks = buf.task[: buf.pos, 0]
    assert stored_tasks.tolist() == [1] * 2 * episode_length + [2] * 2 * episode_length
    # The cap keeps each task's first episodes (reward encodes the episode index).
    assert buf.rewards[: buf.pos, 0].tolist() == ([0.0] * episode_length + [1.0] * episode_length) * 2

    with pytest.raises(ValueError, match="not in"):
        load_multitask_dataset(make_buffer("bad"), dataset_dir, tasks=("missing",))
