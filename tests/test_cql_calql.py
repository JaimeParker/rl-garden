from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import CQL, CalQL, OfflineCQL, OfflineCalQL, OfflineEnvSpec, WSRL
from rl_garden.buffers.mc_buffer import MCTensorReplayBuffer
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer


class DummyVecEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


def _kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": 8,
        "learning_starts": 0,
        "training_freq": 1,
        "eval_freq": 0,
        "net_arch": {"pi": [16], "qf": [16]},
        "n_critics": 4,
        "critic_subsample_size": 2,
        "cql_n_actions": 3,
        "cql_alpha": 1.0,
    }


def _offline_kwargs() -> dict[str, object]:
    params = _kwargs()
    params.pop("learning_starts")
    params.pop("training_freq")
    params.pop("eval_freq")
    return params


def _fill(agent, steps: int = 8) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _offline_env() -> OfflineEnvSpec:
    env = DummyVecEnv()
    return OfflineEnvSpec(
        env.single_observation_space,
        env.single_action_space,
        num_envs=env.num_envs,
    )


def test_cql_standalone_train_step_without_calql_bound():
    agent = CQL(env=DummyVecEnv(), **_kwargs())
    _fill(agent)

    info = agent.train(1)

    assert agent.use_cql_loss
    assert not agent.use_calql
    assert "cql_loss" in info
    assert "calql_bound_rate" not in info
    assert torch.isfinite(torch.tensor(info["critic_loss"]))


def test_cql_does_not_own_calql_or_wsrl_flow_state():
    agent = CQL(env=DummyVecEnv(), **_kwargs())

    assert isinstance(agent.replay_buffer, TensorReplayBuffer)
    assert not isinstance(agent.replay_buffer, MCTensorReplayBuffer)
    assert not hasattr(agent, "switch_to_online_mode")
    assert not hasattr(agent, "offline_replay_buffer")
    assert not hasattr(agent, "offline_data_ratio")


def test_offline_cql_train_step_and_checkpoint(tmp_path):
    agent = OfflineCQL(env=_offline_env(), checkpoint_dir=str(tmp_path), **_offline_kwargs())
    _fill(agent)

    info = agent.train(1)
    result = agent.learn_offline(2, save_filename="offline_cql.pt")

    assert isinstance(agent.replay_buffer, TensorReplayBuffer)
    assert not isinstance(agent.replay_buffer, MCTensorReplayBuffer)
    assert "cql_loss" in info
    assert "calql_bound_rate" not in info
    assert result.final_checkpoint == tmp_path / "offline_cql.pt"
    assert (tmp_path / "offline_cql.pt").exists()


def test_calql_standalone_train_step_logs_bound_rate():
    agent = CalQL(env=DummyVecEnv(), **_kwargs())
    _fill(agent)

    info = agent.train(1)

    assert agent.use_calql
    assert "cql_loss" in info
    assert "calql_bound_rate" in info
    assert torch.isfinite(torch.tensor(info["critic_loss"]))


def test_calql_owns_mc_replay_without_wsrl_flow_state():
    agent = CalQL(env=DummyVecEnv(), **_kwargs())

    assert isinstance(agent.replay_buffer, MCTensorReplayBuffer)
    assert not hasattr(agent, "switch_to_online_mode")
    assert not hasattr(agent, "offline_replay_buffer")
    assert not hasattr(agent, "offline_data_ratio")


def test_offline_calql_train_step_logs_bound_rate():
    agent = OfflineCalQL(
        env=_offline_env(),
        sparse_reward_mc=True,
        sparse_negative_reward=-1.0,
        success_threshold=0.5,
        **_offline_kwargs(),
    )
    _fill(agent)

    info = agent.train(1)

    assert isinstance(agent.replay_buffer, MCTensorReplayBuffer)
    assert agent.replay_buffer.sparse_reward_mc
    assert agent.replay_buffer.sparse_negative_reward == -1.0
    assert "cql_loss" in info
    assert "calql_bound_rate" in info


def test_wsrl_inherits_calql_layer():
    assert issubclass(WSRL, CalQL)


def _write_demo_h5(path):
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=np.zeros((7, 4), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((6, 2), dtype=np.float32))
        group.create_dataset("rewards", data=np.ones((6,), dtype=np.float32))
        dones = np.zeros((6,), dtype=np.float32)
        dones[-1] = 1.0
        group.create_dataset("dones", data=dones)


def test_pretrain_offline_cli_algorithm_selection(tmp_path):
    dataset = tmp_path / "demo.h5"
    _write_demo_h5(dataset)

    for algorithm in ("cql", "calql", "wsrl-calql"):
        checkpoint_dir = tmp_path / algorithm
        cmd = [
            sys.executable,
            "examples/pretrain_offline.py",
            "--algorithm",
            algorithm,
            "--offline_dataset_path",
            str(dataset),
            "--num_offline_steps",
            "2",
            "--buffer_device",
            "cpu",
            "--device",
            "cpu",
            "--log_type",
            "none",
            "--no-std-log",
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--batch_size",
            "4",
            "--buffer_size",
            "32",
            "--n_critics",
            "4",
            "--critic_subsample_size",
            "2",
            "--cql_n_actions",
            "2",
        ]
        subprocess.run(cmd, check=True)
        expected = f"{algorithm.replace('-', '_')}_offline_pretrained.pt"
        assert (checkpoint_dir / expected).exists()


def test_pretrain_cql_offline_cli_legacy_agent_alias(tmp_path):
    dataset = tmp_path / "demo.h5"
    _write_demo_h5(dataset)

    checkpoint_dir = tmp_path / "legacy_cql"
    cmd = [
        sys.executable,
        "examples/pretrain_cql_offline.py",
        "--agent",
        "cql",
        "--offline_dataset_path",
        str(dataset),
        "--num_offline_steps",
        "1",
        "--buffer_device",
        "cpu",
        "--device",
        "cpu",
        "--log_type",
        "none",
        "--no-std-log",
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--batch_size",
        "4",
        "--buffer_size",
        "32",
        "--n_critics",
        "4",
        "--critic_subsample_size",
        "2",
        "--cql_n_actions",
        "2",
    ]
    subprocess.run(cmd, check=True)
    assert (checkpoint_dir / "cql_offline_pretrained.pt").exists()


def test_pretrain_wsrl_offline_cli_legacy_filename(tmp_path):
    dataset = tmp_path / "demo.h5"
    _write_demo_h5(dataset)

    checkpoint_dir = tmp_path / "legacy_wsrl"
    cmd = [
        sys.executable,
        "examples/pretrain_wsrl_offline.py",
        "--offline_dataset_path",
        str(dataset),
        "--num_offline_steps",
        "1",
        "--buffer_device",
        "cpu",
        "--device",
        "cpu",
        "--log_type",
        "none",
        "--no-std-log",
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--batch_size",
        "4",
        "--buffer_size",
        "32",
        "--n_critics",
        "4",
        "--critic_subsample_size",
        "2",
        "--cql_n_actions",
        "2",
    ]
    subprocess.run(cmd, check=True)
    assert (checkpoint_dir / "offline_pretrained.pt").exists()


def test_pretrain_cql_offline_cli_requires_dataset():
    cmd = [
        sys.executable,
        "examples/pretrain_offline.py",
        "--algorithm",
        "cql",
        "--num_offline_steps",
        "1",
        "--log_type",
        "none",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--offline_dataset_path is required" in result.stderr
