"""EpisodeStatistics: per-env return/length accumulation across auto-resets."""
from __future__ import annotations

import torch

from rl_garden.envs.episode_statistics import EpisodeStatistics


def _done(*rows: bool) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.bool)


def test_accumulates_and_restarts_done_rows_across_auto_resets() -> None:
    stats = EpisodeStatistics(num_envs=2, device="cpu")

    first = stats.step(torch.tensor([1.0, 2.0]), _done(False, False))
    second = stats.step(torch.tensor([3.0, 4.0]), _done(True, False))
    third = stats.step(torch.tensor([5.0, 6.0]), _done(False, True))

    assert first["return"].tolist() == [1.0, 2.0] and first["episode_len"].tolist() == [1, 1]
    assert second["return"].tolist() == [4.0, 6.0] and second["episode_len"].tolist() == [2, 2]
    # env 0 restarted after its done step; env 1 ends its 3-step episode.
    assert third["return"].tolist() == [5.0, 12.0] and third["episode_len"].tolist() == [1, 3]
    assert third["return"].dtype == torch.float32 and third["episode_len"].dtype == torch.int64


def test_done_rows_are_zeroed_after_the_step() -> None:
    stats = EpisodeStatistics(num_envs=2, device="cpu")
    stats.step(torch.tensor([1.0, 2.0]), _done(True, False))

    after = stats.step(torch.zeros(2), _done(False, False))

    assert after["return"].tolist() == [0.0, 2.0]
    assert after["episode_len"].tolist() == [1, 2]


def test_returned_tensors_are_clones() -> None:
    stats = EpisodeStatistics(num_envs=2, device="cpu")
    episode = stats.step(torch.tensor([1.0, 2.0]), _done(True, False))

    stats.step(torch.tensor([10.0, 10.0]), _done(False, False))

    assert episode["return"].tolist() == [1.0, 2.0]
    assert episode["episode_len"].tolist() == [1, 1]


def test_reset_zeroes_every_env() -> None:
    stats = EpisodeStatistics(num_envs=2, device="cpu")
    stats.step(torch.tensor([1.0, 2.0]), _done(False, False))

    stats.reset()
    episode = stats.step(torch.tensor([3.0, 4.0]), _done(True, True))

    assert episode["return"].tolist() == [3.0, 4.0]
    assert episode["episode_len"].tolist() == [1, 1]
