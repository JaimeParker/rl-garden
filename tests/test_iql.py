from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import IQL, OfflineEnvSpec
from rl_garden.buffers import DictReplayBuffer, TensorReplayBuffer


def _state_env(num_envs: int = 2) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def _dict_env(num_envs: int = 2) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "rgb": spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
                "state": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def _fill_state(agent: IQL, steps: int = 8) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _fill_dict(agent: IQL, steps: int = 4) -> None:
    env = agent.env
    for _ in range(steps):
        obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        next_obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_iql_state_train_step_and_checkpoint(tmp_path):
    agent = IQL(
        env=_state_env(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        net_arch={"pi": [16], "qf": [16], "vf": [16]},
        n_critics=3,
        critic_subsample_size=2,
        checkpoint_dir=str(tmp_path),
        std_log=False,
    )
    _fill_state(agent)

    info = agent.train(1)
    result = agent.learn_offline(2, save_filename="iql.pt")

    assert isinstance(agent.replay_buffer, TensorReplayBuffer)
    assert torch.isfinite(torch.tensor(info["loss"]))
    assert "value_loss" in info
    assert "behavior_log_prob" in info
    assert result.final_checkpoint == tmp_path / "iql.pt"
    assert (tmp_path / "iql.pt").exists()


def test_iql_dict_uses_dict_replay_and_combined_encoder():
    agent = IQL(
        env=_dict_env(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=4,
        net_arch=[16],
        n_critics=2,
        critic_subsample_size=2,
        image_keys=("rgb",),
        image_fusion_mode="stack_channels",
        std_log=False,
    )
    _fill_dict(agent)

    info = agent.train(1)

    assert isinstance(agent.replay_buffer, DictReplayBuffer)
    assert torch.isfinite(torch.tensor(info["critic_loss"]))
    assert agent.policy.features_extractor.features_dim > 0
