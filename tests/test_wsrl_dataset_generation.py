"""Tests for SAC-to-WSRL offline dataset generation utilities."""
from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers import (
    MCDictReplayBuffer,
    MCTensorReplayBuffer,
    load_maniskill_h5_to_replay_buffer,
)
from rl_garden.datasets import (
    CheckpointScore,
    PolicySource,
    WSRLTrajectoryWriter,
    collect_policy_dataset,
    discover_checkpoints,
    normalize_mix,
    select_policy_sources,
)


class FakePolicy:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, obs, deterministic: bool = True):
        del deterministic
        batch_size = obs.shape[0]
        return torch.full((batch_size, 1), self.value)


class FakeAgent:
    def __init__(self, value: float) -> None:
        self.policy = FakePolicy(value)
        self.device = torch.device("cpu")


class FakeVecEnv:
    def __init__(self, num_envs: int = 2, episode_len: int = 3) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self.single_observation_space = spaces.Box(
            low=-10, high=10, shape=(2,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_envs, 1), dtype=np.float32
        )
        self.steps = torch.zeros(num_envs, dtype=torch.long)

    def _obs(self) -> torch.Tensor:
        return self.steps.float().unsqueeze(1).repeat(1, 2)

    def reset(self):
        self.steps.zero_()
        return self._obs(), {}

    def step(self, actions):
        self.steps += 1
        final_obs = self._obs()
        rewards = actions[:, 0].float()
        terminated = self.steps >= self.episode_len
        truncated = torch.zeros_like(terminated)
        success = actions[:, 0] > 0.5
        next_obs = final_obs.clone()
        self.steps[terminated] = 0
        next_obs[terminated] = self._obs()[terminated]
        infos = {
            "final_observation": final_obs,
            "final_info": {"episode": {"success_at_end": success}},
            "_final_info": terminated,
        }
        return next_obs, rewards, terminated, truncated, infos


def _source(target_transitions: int = 2) -> PolicySource:
    return PolicySource(
        tier="success",
        name="checkpoint_1",
        path=None,
        target_transitions=target_transitions,
        success_rate=1.0,
    )


def test_discover_checkpoints_and_select_sources(tmp_path):
    (tmp_path / "checkpoint_400.pt").touch()
    (tmp_path / "checkpoint_200.pt").touch()
    (tmp_path / "final.pt").touch()

    assert [p.name for p in discover_checkpoints(tmp_path)] == [
        "checkpoint_200.pt",
        "checkpoint_400.pt",
        "final.pt",
    ]
    assert normalize_mix((3, 3, 4)) == (0.3, 0.3, 0.4)

    scores = [
        CheckpointScore(tmp_path / "failure.pt", 0.05, 1.0, 10.0, 5),
        CheckpointScore(tmp_path / "near.pt", 0.55, 2.0, 10.0, 5),
        CheckpointScore(tmp_path / "success.pt", 0.95, 3.0, 10.0, 5),
    ]
    sources = select_policy_sources(scores, total_transitions=100)
    assert [s.path.name for s in sources if s.path is not None] == [
        "failure.pt",
        "near.pt",
        "success.pt",
    ]
    assert [s.target_transitions for s in sources] == [30, 30, 40]


def test_select_sources_can_use_random_failure_fallback(tmp_path):
    scores = [
        CheckpointScore(tmp_path / "near.pt", 0.45, 2.0, 10.0, 5),
        CheckpointScore(tmp_path / "success.pt", 0.95, 3.0, 10.0, 5),
    ]
    sources = select_policy_sources(scores, total_transitions=10)
    assert sources[0].tier == "failure"
    assert sources[0].path is None
    assert sources[0].name == "random"
    assert sources[0].fallback_reason is not None


def test_writer_outputs_state_h5_compatible_with_wsrl_loader(tmp_path):
    path = tmp_path / "state.h5"
    with WSRLTrajectoryWriter(path) as writer:
        writer.write_episode(
            obs=[torch.zeros(2), torch.ones(2), torch.ones(2) * 2],
            actions=[torch.zeros(1), torch.ones(1)],
            rewards=[torch.tensor(1.0), torch.tensor(2.0)],
            terminated=[False, True],
            truncated=[False, False],
            source=_source(),
            success=True,
        )

    buffer = MCTensorReplayBuffer(
        spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
        spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=8,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    assert load_maniskill_h5_to_replay_buffer(buffer, path) == 2
    assert torch.equal(buffer.obs[:2, 0], torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert torch.equal(buffer.next_obs[:2, 0], torch.tensor([[1.0, 1.0], [2.0, 2.0]]))


def test_writer_outputs_dict_h5_compatible_with_wsrl_loader(tmp_path):
    path = tmp_path / "rgb.h5"
    obs = [
        {"state": torch.zeros(2), "rgb": torch.zeros(4, 4, 3, dtype=torch.uint8)},
        {"state": torch.ones(2), "rgb": torch.ones(4, 4, 3, dtype=torch.uint8)},
    ]
    with WSRLTrajectoryWriter(path) as writer:
        writer.write_episode(
            obs=obs,
            actions=[torch.zeros(1)],
            rewards=[torch.tensor(1.0)],
            terminated=[True],
            truncated=[False],
            source=_source(target_transitions=1),
            success=True,
        )

    buffer = MCDictReplayBuffer(
        spaces.Dict(
            {
                "state": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
                "rgb": spaces.Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8),
            }
        ),
        spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=8,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    assert load_maniskill_h5_to_replay_buffer(buffer, path) == 1
    assert buffer.obs["rgb"][0, 0].dtype == torch.uint8


def test_collect_policy_dataset_writes_complete_episodes_and_final_obs(tmp_path):
    path = tmp_path / "collected.h5"
    env = FakeVecEnv(num_envs=2, episode_len=3)
    source = _source(target_transitions=5)
    with WSRLTrajectoryWriter(path) as writer:
        stats = collect_policy_dataset(
            agent=FakeAgent(1.0),
            env=env,
            writer=writer,
            source=source,
            deterministic=True,
            device="cpu",
        )

    assert stats.transitions == 6
    assert stats.episodes == 2
    assert stats.successes == 2
    with h5py.File(path, "r") as f:
        assert sorted(f.keys()) == ["traj_0", "traj_1"]
        assert f["traj_0"]["obs"].shape == (4, 2)
        assert np.all(f["traj_0"]["obs"][-1] == np.array([3.0, 3.0]))
        assert f["traj_0"].attrs["final_success"]
