"""Tests for tools/conversion/convert_ncrl_dataset.py on synthetic NCRL npz
episodes (see ``tests/_ncrl_synthetic.py`` for the recognisable values)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "conversion"))

from _ncrl_synthetic import obs_value, write_benchmark, write_episode  # noqa: E402
from convert_ncrl_dataset import convert_dataset  # noqa: E402

EMBODIMENTS = {"cheetah": (6, (4, 6, 5)), "quadruped": (12, (5, 3, 7))}


@pytest.fixture()
def converted(tmp_path):
    src = write_benchmark(tmp_path / "src", EMBODIMENTS, image_size=8)
    manifests = convert_dataset(src, tmp_path / "dst")
    return tmp_path / "dst", manifests


def test_manifest_fields(converted):
    dst, manifests = converted
    assert [m["embodiment"] for m in manifests] == ["cheetah", "quadruped"]
    for manifest, (name, (action_dim, lengths)) in zip(manifests, EMBODIMENTS.items()):
        on_disk = json.loads((dst / name / "manifest.json").read_text())
        assert on_disk == manifest
        assert manifest["format"] == "rl_garden_sequence_v1"
        assert manifest["obs_key"] == "rgb_default"
        assert manifest["action_dim"] == action_dim
        assert manifest["padded_action_dim"] == 12
        assert manifest["num_episodes"] == 3
        assert manifest["episode_lengths"] == list(lengths)
        assert manifest["image_shape"] == [8, 8, 3]
        assert len(manifest["source_files_sha256_sample"]) == 3


def test_obs_hwc_and_one_step_alignment(converted):
    dst, _ = converted
    for emb, (name, (action_dim, lengths)) in enumerate(EMBODIMENTS.items()):
        obs = np.load(dst / name / "obs.npy", mmap_mode="r")
        episodes = torch.load(dst / name / "episodes.pt", weights_only=True)
        assert obs.dtype == np.uint8 and obs.shape == (sum(lengths), 8, 8, 3)
        assert episodes["action"].shape == (sum(lengths), 12)
        offset = 0
        for ep, length in enumerate(lengths):
            for t in range(length):
                row = offset + t
                # obs[t] is the official observation[t] (final frame dropped) ...
                assert np.all(obs[row] == obs_value(emb, ep, t))
                # ... and action/reward[t] are the official rows t + 1.
                assert torch.all(episodes["action"][row, :action_dim] == ep * 100 + t + 1)
                assert torch.all(episodes["action"][row, action_dim:] == 0)
                assert episodes["reward"][row] == t + 1.5
                assert episodes["episode_id"][row] == ep
                assert bool(episodes["is_first"][row]) == (t == 0)
            offset += length


def test_padded_action_dim_override_and_embodiment_filter(tmp_path):
    src = write_benchmark(tmp_path / "src", EMBODIMENTS, image_size=8)
    (manifest,) = convert_dataset(src, tmp_path / "dst", embodiments=["cheetah"], padded_action_dim=21)
    assert manifest["padded_action_dim"] == 21
    assert torch.load(tmp_path / "dst" / "cheetah" / "episodes.pt")["action"].shape[1] == 21
    assert not (tmp_path / "dst" / "quadruped").exists()
    with pytest.raises(ValueError, match="padded_action_dim"):
        convert_dataset(src, tmp_path / "dst2", padded_action_dim=6)


def test_filename_length_mismatch_raises(tmp_path):
    path = write_episode(tmp_path / "src" / "walker", 0, 0, 4, 6, 8)
    path.rename(path.with_name(path.name.replace("-0-4.npz", "-0-5.npz")))
    with pytest.raises(ValueError, match="file name says"):
        convert_dataset(tmp_path / "src", tmp_path / "dst")
