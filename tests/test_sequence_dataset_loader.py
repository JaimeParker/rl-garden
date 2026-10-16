"""Tests for ``rl_garden.buffers.sequence_dataset``: ``load_sequence_dataset``
into a uniform-mode ``SequenceReplayBuffer`` and ``MmapSequenceDataset``,
on a converted synthetic two-embodiment NCRL dataset."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "conversion"))

from _ncrl_synthetic import write_benchmark  # noqa: E402
from convert_ncrl_dataset import convert_dataset  # noqa: E402

from rl_garden.buffers.sequence_dataset import (  # noqa: E402
    MmapSequenceDataset,
    infer_sequence_dataset_spec,
    load_sequence_dataset,
)
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer  # noqa: E402

EMBODIMENTS = {"cheetah": (6, (9, 12, 10)), "quadruped": (12, (11, 8, 10))}
HORIZON = 4
CARRY_SPEC = {"deter": (5,), "stoch": (6,)}


def _decode(pixel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pixel value -> (embodiment, episode, official row t); see _ncrl_synthetic."""
    value = pixel.long()
    return value // 128, (value % 128) // 20, (value % 128) % 20


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("ncrl")
    write_benchmark(root / "src", EMBODIMENTS, image_size=8)
    convert_dataset(root / "src", root / "dst")
    return root / "dst"


def test_spec_from_manifests(dataset_dir):
    spec = infer_sequence_dataset_spec(dataset_dir)
    assert spec.obs_key == "rgb_default"
    assert spec.image_shape == (8, 8, 3)
    assert spec.padded_action_dim == 12
    assert spec.embodiments == ("cheetah", "quadruped")
    assert spec.action_dims == (6, 12)
    assert spec.num_episodes == 6
    assert spec.num_steps == sum(sum(lengths) for _, lengths in EMBODIMENTS.values())


def _buffer(dataset_dir) -> SequenceReplayBuffer:
    spec = infer_sequence_dataset_spec(dataset_dir)
    return SequenceReplayBuffer(
        observation_space=spaces.Dict({"rgb_default": spaces.Box(0, 255, spec.image_shape, np.uint8)}),
        action_space=spaces.Box(-1, 1, (spec.padded_action_dim,), np.float32),
        num_envs=1,
        buffer_size=spec.num_steps,
        cross_episode=True,
        priority=False,
        horizon=HORIZON,
        carry_spec=CARRY_SPEC,
        storage_device="cpu",
        sample_device="cpu",
    )


def test_load_into_buffer_flags(dataset_dir):
    buffer = _buffer(dataset_dir)
    written = load_sequence_dataset(dataset_dir, buffer)
    spec = infer_sequence_dataset_spec(dataset_dir)
    assert written == spec.num_steps
    lengths = [length for _, ls in EMBODIMENTS.values() for length in ls]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    ends = np.cumsum(lengths) - 1
    expected_first = torch.zeros(spec.num_steps, dtype=torch.bool)
    expected_first[torch.as_tensor(starts)] = True
    expected_end = torch.zeros(spec.num_steps, dtype=torch.bool)
    expected_end[torch.as_tensor(ends)] = True
    assert torch.equal(buffer._is_first[:, 0], expected_first)
    assert torch.equal(buffer.episode_ends[:, 0], expected_end)
    assert not buffer.dones.any()
    for key in CARRY_SPEC:
        assert torch.count_nonzero(buffer._carry[key]) == 0


def test_buffer_sample_alignment_and_mixing(dataset_dir):
    buffer = _buffer(dataset_dir)
    load_sequence_dataset(dataset_dir, buffer)
    torch.manual_seed(0)
    batch = buffer.sample(256)
    assert batch.obs["rgb_default"].shape == (HORIZON + 1, 256, 8, 8, 3)
    assert batch.action.shape == (HORIZON, 256, 12)
    emb, ep, t = _decode(batch.obs["rgb_default"][..., 0, 0, 0])
    assert set(emb[0].tolist()) == {0, 1}  # both embodiments in the B dim
    # is_first marks exactly the loss rows that are an episode's first step.
    assert torch.equal(batch.is_first, t[1:] == 0)
    # action[k] is the transition into loss row k: official action row t.
    inside = ~batch.is_first
    assert torch.equal(batch.action[..., 0][inside], (ep[1:] * 100 + t[1:]).float()[inside])
    # cheetah rows keep zero padding beyond its 6 real action dims.
    assert torch.all(batch.action[..., 6:][(emb[:-1] == 0)] == 0)
    for key, shape in CARRY_SPEC.items():
        assert batch.carry[key].shape == (256, *shape)
        assert torch.count_nonzero(batch.carry[key]) == 0


def test_mmap_dataset_windows(dataset_dir):
    dataset = MmapSequenceDataset(dataset_dir, HORIZON, carry_spec=CARRY_SPEC, sample_device="cpu")
    assert dataset.num_episodes == 6
    torch.manual_seed(0)
    batch = dataset.sample(256)
    obs = batch.obs["rgb_default"]
    assert obs.shape == (HORIZON + 1, 256, 8, 8, 3) and obs.dtype == torch.uint8
    emb, ep, t = _decode(obs[..., 0, 0, 0])
    assert set(emb[1].tolist()) == {0, 1}
    # Loss rows are consecutive steps of a single episode.
    assert torch.all(emb[1:] == emb[1]) and torch.all(ep[1:] == ep[1])
    assert torch.equal(t[1:], t[1] + torch.arange(HORIZON).unsqueeze(1))
    expected_first = torch.zeros(HORIZON, 256, dtype=torch.bool)
    expected_first[0] = True
    assert torch.equal(batch.is_first, expected_first)
    # Transition into loss row k >= 1: official action row t; reward t + 0.5.
    assert torch.equal(batch.action[1:, :, 0], (ep[2:] * 100 + t[2:]).float())
    assert torch.equal(batch.reward[1:], t[2:].float() + 0.5)
    for key, shape in CARRY_SPEC.items():
        assert torch.count_nonzero(batch.carry[key]) == 0 and batch.carry[key].shape == (256, *shape)
    # Every window start is reachable, including an episode's first step.
    assert bool((t[1] == 0).any())
