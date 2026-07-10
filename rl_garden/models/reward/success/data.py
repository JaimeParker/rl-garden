"""Dataset for :mod:`rl_garden.models.reward.success.train`.

Data collection convention (see ``collect_data.py``): each pickled file is a
list of ``{"obs": obs_dict}`` entries (CPU tensors), split into two
directories/naming patterns by label -- success and failure -- mirroring
HIL-SERL's own ``classifier_data/*_{success,failure}_*.pkl`` convention
(``3rd_party/hil-serl/examples/record_success_fail.py``). Only ``obs`` is
kept (not action/reward/done) since the classifier only ever consumes
images.
"""
from __future__ import annotations

import glob
import pickle
from typing import Sequence

import torch
from torch.utils.data import Dataset


class SuccessClassifierDataset(Dataset):
    def __init__(self, success_paths: Sequence[str], failure_paths: Sequence[str]) -> None:
        self.samples: list[tuple[dict, float]] = []
        for pattern in success_paths:
            self.samples.extend(self._load(pattern, label=1.0))
        for pattern in failure_paths:
            self.samples.extend(self._load(pattern, label=0.0))

    @staticmethod
    def _load(pattern: str, label: float) -> list[tuple[dict, float]]:
        samples = []
        for path in sorted(glob.glob(pattern)):
            with open(path, "rb") as f:
                entries = pickle.load(f)
            samples.extend((entry["obs"], label) for entry in entries)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[dict, float]:
        return self.samples[index]


def collate_obs_label(batch: Sequence[tuple[dict, float]]) -> tuple[dict, torch.Tensor]:
    obs_list, labels = zip(*batch)
    keys = obs_list[0].keys()
    obs = {k: torch.stack([o[k] for o in obs_list]) for k in keys}
    return obs, torch.tensor(labels, dtype=torch.float32)
