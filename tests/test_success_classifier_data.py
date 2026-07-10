from __future__ import annotations

import os
import pickle

import torch

from rl_garden.models.reward.success.data import SuccessClassifierDataset, collate_obs_label


def _write_pkl(path: str, entries: list[dict]) -> None:
    with open(path, "wb") as f:
        pickle.dump(entries, f)


def test_dataset_assigns_labels_by_glob_pattern(tmp_path):
    success_entries = [{"obs": {"wrist": torch.zeros(3, 4, 4)}} for _ in range(3)]
    failure_entries = [{"obs": {"wrist": torch.ones(3, 4, 4)}} for _ in range(2)]
    _write_pkl(os.path.join(tmp_path, "a_success_1.pkl"), success_entries)
    _write_pkl(os.path.join(tmp_path, "a_failure_1.pkl"), failure_entries)

    dataset = SuccessClassifierDataset(
        [os.path.join(tmp_path, "*success*.pkl")],
        [os.path.join(tmp_path, "*failure*.pkl")],
    )

    assert len(dataset) == 5
    labels = sorted(label for _, label in dataset)
    assert labels == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_collate_obs_label_stacks_batch(tmp_path):
    entries = [{"obs": {"wrist": torch.full((3, 4, 4), float(i))}} for i in range(4)]
    _write_pkl(os.path.join(tmp_path, "s.pkl"), entries)

    dataset = SuccessClassifierDataset([os.path.join(tmp_path, "s.pkl")], [])
    batch = [dataset[i] for i in range(len(dataset))]
    obs, labels = collate_obs_label(batch)

    assert obs["wrist"].shape == (4, 3, 4, 4)
    assert torch.all(labels == 1.0)
