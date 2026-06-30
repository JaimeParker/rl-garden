from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
import sys
import torch


def _load_tool(name: str):
    path = Path(__file__).parents[1] / "tools" / "diagnostics" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collector = _load_tool("collect_drqv2_success_rollouts")
probe = _load_tool("probe_drqv2_release_q")
Episode = collector.Episode
SuccessRolloutWriter = collector.SuccessRolloutWriter
classify_episode = collector.classify_episode
update_reservoir = collector.update_reservoir
decision_index = probe.decision_index
event_window = probe.event_window
gradient_alignment = probe.gradient_alignment


def _episode(*, success: bool = False, on_cube: bool = False) -> Episode:
    episode = Episode(
        obs=[
            {"rgb_base_camera": torch.zeros(4, 4, 3, dtype=torch.uint8)},
            {"rgb_base_camera": torch.ones(4, 4, 3, dtype=torch.uint8)},
        ],
        actions=[torch.zeros(8)],
        actor_mean_actions=[torch.ones(8)],
        rewards=[torch.tensor(1.0)],
        terminated=[torch.tensor(False)],
        truncated=[torch.tensor(True)],
    )
    episode.stages["is_cubeA_grasped"] = [not success]
    episode.stages["is_cubeA_on_cubeB"] = [on_cube or success]
    episode.stages["is_cubeA_static"] = [success]
    episode.stages["success"] = [success]
    return episode


def test_classify_episode_distinguishes_success_and_near_success():
    assert classify_episode(_episode(success=True)) == "success"
    assert classify_episode(_episode(on_cube=True)) == "near_success"
    assert classify_episode(_episode()) == "failure"


def test_success_writer_preserves_temporal_schema(tmp_path):
    import h5py

    path = tmp_path / "rollouts.h5"
    writer = SuccessRolloutWriter(path, {"stddev": 0.1})
    writer.write(_episode(success=True), "success")
    writer.close(episodes_seen=3, successes=1, near_successes=0)

    with h5py.File(path, "r") as handle:
        traj = handle["traj_0"]
        assert traj["obs/rgb_base_camera"].shape == (2, 4, 4, 3)
        assert traj["actions"].shape == (1, 8)
        assert traj["actor_mean_actions"].shape == (1, 8)
        assert bool(traj["infos/success"][0])
        assert traj.attrs["label"] == "success"
        assert handle.attrs["episodes_seen"] == 3


def test_near_success_reservoir_is_bounded():
    rng = np.random.default_rng(7)
    reservoir: list[Episode] = []
    episodes = [_episode(on_cube=True) for _ in range(10)]
    for seen, episode in enumerate(episodes, start=1):
        update_reservoir(reservoir, episode, seen, 3, rng)
    assert len(reservoir) == 3
    assert all(classify_episode(episode) == "near_success" for episode in reservoir)


def test_decision_index_prefers_on_cube_release():
    index, kind = decision_index(
        np.array([False, True, True, False, False]),
        np.array([False, False, True, True, True]),
    )
    assert (index, kind) == (3, "release")


def test_decision_index_falls_back_to_first_on_cube_state():
    index, kind = decision_index(
        np.array([False, True, True, True]),
        np.array([False, False, True, True]),
    )
    assert (index, kind) == (2, "on_cube")


def test_event_window_is_aligned_to_first_on_cube_state():
    event, indices = event_window(
        np.array([False, False, False, True, True]), pre_event_steps=2
    )
    assert event == 3
    assert list(indices) == [1, 2, 3]


def test_event_window_rejects_missing_event():
    with np.testing.assert_raises_regex(ValueError, "no on-cube"):
        event_window(np.zeros(4, dtype=bool), pre_event_steps=2)


def test_gradient_alignment_reports_direction_and_zero_norm():
    assert gradient_alignment(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])) == 1.0
    assert gradient_alignment(torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.0])) == -1.0
    assert gradient_alignment(torch.zeros(2), torch.ones(2)) == 0.0
