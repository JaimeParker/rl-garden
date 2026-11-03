"""ACTTrajectoryDataset: episode splitting, action chunks with padding, norm stats, cameras."""
from __future__ import annotations

import pickle

import pytest
import torch

from rl_garden.models.act.trajectory_dataset import (
    MIN_NORM_STD,
    ACTTrajectoryDataset,
    episode_end_indices,
    load_transition_pkls,
)

STATE_DIM = 3
IMAGE_HW = (4, 5)


def _obs(step: int, cameras: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    obs = {"state": torch.full((1, STATE_DIM), float(step))}
    for camera_index, camera in enumerate(cameras):
        obs[camera] = torch.full((1, *IMAGE_HW, 3), step * 10 + camera_index, dtype=torch.uint8)
    return obs


def _episode(
    first_step: int,
    length: int,
    *,
    done_at_end: bool,
    truncated_at_end: bool = False,
    cameras: tuple[str, ...] = (),
) -> list[dict]:
    transitions = []
    for offset in range(length):
        step = first_step + offset
        is_last = offset == length - 1
        transition = {
            "obs": _obs(step, cameras),
            "next_obs": _obs(step + 1, cameras),
            # The second action dim is constant, so its std is clamped.
            "action": torch.tensor([[float(step), 0.5]]),
            "reward": torch.zeros(1),
            "done": torch.tensor([done_at_end and is_last]),
        }
        if truncated_at_end:
            transition["truncated"] = torch.tensor([is_last])
        transitions.append(transition)
    return transitions


def _three_episodes(*, truncated_middle: bool = False, cameras: tuple[str, ...] = ()) -> list[dict]:
    """Steps 0-2 (ends with done), 10-13 (no done: an observation gap follows), 20-21."""

    return [
        *_episode(0, 3, done_at_end=True, cameras=cameras),
        *_episode(10, 4, done_at_end=False, truncated_at_end=truncated_middle, cameras=cameras),
        *_episode(20, 2, done_at_end=False, cameras=cameras),
    ]


def test_episodes_split_on_done_and_observation_gaps() -> None:
    dataset = ACTTrajectoryDataset(_three_episodes(), num_queries=4)

    assert dataset.episode_lengths == [3, 4, 2]
    assert len(dataset) == 9


def test_without_gap_splitting_only_done_ends_episodes() -> None:
    dataset = ACTTrajectoryDataset(_three_episodes(), num_queries=4, split_on_observation_gaps=False)

    assert dataset.episode_lengths == [3, 6]


def test_truncated_key_ends_an_episode() -> None:
    ends = episode_end_indices(_three_episodes(truncated_middle=True), split_on_observation_gaps=False)

    assert ends == [3, 7, 9]


def test_action_chunk_repeats_last_action_and_marks_padding() -> None:
    dataset = ACTTrajectoryDataset(_three_episodes(), num_queries=4)
    stats = dataset.norm_stats

    batch = dataset.get_batch(torch.tensor([0, 1, 3]))
    raw_chunk = batch.action_chunk * stats["action_std"] + stats["action_mean"]

    assert batch.action_chunk.shape == (3, 4, 2)
    torch.testing.assert_close(raw_chunk[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 2.0]))
    torch.testing.assert_close(raw_chunk[1, :, 0], torch.tensor([1.0, 2.0, 2.0, 2.0]))
    torch.testing.assert_close(raw_chunk[2, :, 0], torch.tensor([10.0, 11.0, 12.0, 13.0]))
    assert batch.is_pad.tolist() == [
        [False, False, False, True],
        [False, False, True, True],
        [False, False, False, False],
    ]


def test_norm_stats_layout_and_normalised_state() -> None:
    transitions = _three_episodes()
    dataset = ACTTrajectoryDataset(transitions, num_queries=4)
    stats = dataset.norm_stats

    assert set(stats) == {"state_mean", "state_std", "action_mean", "action_std"}
    assert stats["state_mean"].shape == (1, STATE_DIM) and stats["state_std"].shape == (1, STATE_DIM)
    assert stats["action_mean"].shape == (1, 2) and stats["action_std"].shape == (1, 2)
    assert all(value.dtype == torch.float32 and value.device.type == "cpu" for value in stats.values())
    assert stats["action_std"][0, 1].item() == pytest.approx(MIN_NORM_STD)

    raw_states = torch.cat([transition["obs"]["state"] for transition in transitions])
    torch.testing.assert_close(raw_states.mean(dim=0, keepdim=True), stats["state_mean"])
    batch = dataset.get_batch(torch.tensor([4]))
    torch.testing.assert_close(batch.state, (raw_states[4:5] - stats["state_mean"]) / stats["state_std"])


def test_camera_keys_are_sorted_and_images_stacked_channel_first() -> None:
    dataset = ACTTrajectoryDataset(_three_episodes(cameras=("rgb_wrist", "rgb_front")), num_queries=2)

    batch = dataset.sample(5)

    assert dataset.camera_keys == ("rgb_front", "rgb_wrist")
    assert batch.images is not None
    assert batch.images.shape == (5, 2, 3, *IMAGE_HW) and batch.images.dtype == torch.uint8
    first = dataset.get_batch(torch.tensor([1])).images
    # Step 1: rgb_front was created as camera 1 (value 11), rgb_wrist as camera 0 (value 10).
    assert torch.all(first[0, 0] == 11) and torch.all(first[0, 1] == 10)


def test_state_only_when_cameras_are_disabled() -> None:
    dataset = ACTTrajectoryDataset(
        _three_episodes(cameras=("rgb_front",)), num_queries=2, use_cameras=False
    )

    assert dataset.camera_keys == ()
    assert dataset.get_batch(torch.tensor([0])).images is None


def test_flat_observation_is_the_state() -> None:
    transitions = [
        {"obs": torch.full((1, STATE_DIM), float(step)), "action": torch.zeros(1, 2), "done": step == 2}
        for step in range(3)
    ]

    dataset = ACTTrajectoryDataset(transitions, num_queries=2)

    assert dataset.state_dim == STATE_DIM and dataset.episode_lengths == [3]


def test_stacked_rgb_next_to_per_camera_keys_is_rejected() -> None:
    transitions = _three_episodes(cameras=("rgb_front",))
    for transition in transitions:
        transition["obs"]["rgb"] = transition["obs"]["rgb_front"]

    with pytest.raises(ValueError, match="per-camera"):
        ACTTrajectoryDataset(transitions, num_queries=2)


def test_load_transition_pkls_concatenates_matches_in_sorted_order(tmp_path) -> None:
    for name, first_step in (("b.pkl", 10), ("a.pkl", 0)):
        with open(tmp_path / name, "wb") as stream:
            pickle.dump(_episode(first_step, 2, done_at_end=True), stream)

    transitions = load_transition_pkls([str(tmp_path / "*.pkl")])

    assert [transition["action"][0, 0].item() for transition in transitions] == [0.0, 1.0, 10.0, 11.0]
    with pytest.raises(FileNotFoundError):
        load_transition_pkls([str(tmp_path / "missing_*.pkl")])
