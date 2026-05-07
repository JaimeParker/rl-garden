"""Checkpoint roundtrip tests for off-policy agents and replay buffers."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import RGBDSAC, SAC, WSRL
from rl_garden.buffers import DictReplayBuffer, TensorReplayBuffer
from rl_garden.buffers.mc_buffer import MCTensorReplayBuffer


class DummyVecEnv:
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box) -> None:
        self.num_envs = 2
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(action_space.low, (self.num_envs,) + action_space.shape),
            high=np.broadcast_to(action_space.high, (self.num_envs,) + action_space.shape),
            dtype=action_space.dtype,
        )


class DummyStepVecEnv(DummyVecEnv):
    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(self.num_envs, *self.single_observation_space.shape), {}

    def step(self, actions):
        obs = torch.randn(self.num_envs, *self.single_observation_space.shape)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}


def _state_env() -> DummyVecEnv:
    return DummyVecEnv(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )


def _state_step_env() -> DummyStepVecEnv:
    return DummyStepVecEnv(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )


def _rgbd_env() -> DummyVecEnv:
    return DummyVecEnv(
        spaces.Dict(
            {
                "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )


def _sac_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 16,
        "batch_size": 4,
        "training_freq": 4,
        "learning_starts": 4,
        "eval_freq": 0,
        "net_arch": [16],
    }


def _wsrl_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 16,
        "batch_size": 4,
        "training_freq": 4,
        "learning_starts": 4,
        "eval_freq": 0,
        "net_arch": {"pi": [16], "qf": [16]},
        "n_critics": 3,
        "critic_subsample_size": 2,
        "cql_n_actions": 2,
        "cql_alpha": 1.5,
    }


def _add_state_transitions(agent, steps: int = 4) -> None:
    env = agent.env
    for i in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        action = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        reward = torch.full((env.num_envs,), float(i))
        done = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, action, reward, done)


def _add_rgbd_transitions(agent, steps: int = 4) -> None:
    env = agent.env
    for i in range(steps):
        obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        next_obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        action = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        reward = torch.full((env.num_envs,), float(i))
        done = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, action, reward, done)


def _assert_state_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_sac_checkpoint_roundtrip_with_replay_buffer(tmp_path):
    agent = SAC(env=_state_env(), **_sac_kwargs())
    _add_state_transitions(agent)
    agent.train(1)
    agent._global_step = 8

    path = tmp_path / "checkpoint_8.pt"
    agent.save(path, include_replay_buffer=True)

    loaded = SAC(env=_state_env(), **_sac_kwargs())
    loaded.load(path)

    _assert_state_dict_equal(agent.policy.state_dict(), loaded.policy.state_dict())
    assert loaded._global_step == 8
    assert loaded._global_update == agent._global_update
    assert loaded.replay_buffer.pos == agent.replay_buffer.pos
    assert torch.equal(loaded.replay_buffer.obs, agent.replay_buffer.obs)
    assert torch.equal(loaded.replay_buffer.actions, agent.replay_buffer.actions)


def test_learn_writes_periodic_and_final_checkpoints(tmp_path):
    agent = SAC(
        env=_state_step_env(),
        **_sac_kwargs(),
        checkpoint_dir=str(tmp_path),
        checkpoint_freq=2,
        save_replay_buffer=True,
    )

    agent.learn(total_timesteps=4)

    assert (tmp_path / "checkpoint_4.pt").exists()
    assert (tmp_path / "replay_buffer_4.pt").exists()
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "replay_buffer_final.pt").exists()


def test_rgbdsac_checkpoint_roundtrip(tmp_path):
    agent = RGBDSAC(
        env=_rgbd_env(),
        **_sac_kwargs(),
        image_keys=("rgb",),
    )
    _add_rgbd_transitions(agent)
    agent._global_step = 4

    path = tmp_path / "rgbd_final.pt"
    agent.save(path, include_replay_buffer=True)

    loaded = RGBDSAC(env=_rgbd_env(), **_sac_kwargs(), image_keys=("rgb",))
    loaded.load(path)

    _assert_state_dict_equal(agent.policy.state_dict(), loaded.policy.state_dict())
    assert loaded.replay_buffer.pos == agent.replay_buffer.pos
    assert torch.equal(loaded.replay_buffer.obs["rgb"], agent.replay_buffer.obs["rgb"])
    assert torch.equal(loaded.replay_buffer.obs["state"], agent.replay_buffer.obs["state"])


def test_wsrl_checkpoint_restores_extra_state(tmp_path):
    agent = WSRL(env=_state_env(), **_wsrl_kwargs())
    _add_state_transitions(agent)
    agent.train(1)
    agent.switch_to_online_mode()
    agent.use_td_loss = False
    agent._global_step = 12

    path = tmp_path / "wsrl.pt"
    agent.save(path)

    loaded = WSRL(env=_state_env(), **_wsrl_kwargs())
    loaded.load(path)

    _assert_state_dict_equal(agent.policy.state_dict(), loaded.policy.state_dict())
    assert loaded._global_step == 12
    assert loaded.use_cql_loss == agent.use_cql_loss
    assert loaded.use_td_loss == agent.use_td_loss
    assert torch.allclose(loaded._current_alpha(), agent._current_alpha())


def test_tensor_replay_buffer_file_roundtrip(tmp_path):
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    source = TensorReplayBuffer(obs_space, act_space, 2, 8, "cpu", "cpu")
    target = TensorReplayBuffer(obs_space, act_space, 2, 8, "cpu", "cpu")
    for _ in range(3):
        source.add(
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 2),
            torch.randn(2),
            torch.zeros(2),
        )

    path = tmp_path / "buffer.pt"
    from rl_garden.common.checkpoint import load_replay_buffer_file, save_replay_buffer_file

    save_replay_buffer_file(path, source)
    load_replay_buffer_file(path, target)
    assert target.pos == source.pos
    assert target.full == source.full
    assert torch.equal(target.obs, source.obs)
    assert torch.equal(target.next_obs, source.next_obs)


def test_dict_and_mc_replay_buffer_file_roundtrip(tmp_path):
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    dict_source = DictReplayBuffer(obs_space, act_space, 2, 8, "cpu", "cpu")
    dict_target = DictReplayBuffer(obs_space, act_space, 2, 8, "cpu", "cpu")
    mc_source = MCTensorReplayBuffer(
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        act_space,
        2,
        8,
        gamma=0.7,
        storage_device="cpu",
        sample_device="cpu",
    )
    mc_target = MCTensorReplayBuffer(
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        act_space,
        2,
        8,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    for _ in range(3):
        dict_source.add(
            {
                "rgb": torch.randint(0, 256, (2, 4, 4, 3), dtype=torch.uint8),
                "state": torch.randn(2, 3),
            },
            {
                "rgb": torch.randint(0, 256, (2, 4, 4, 3), dtype=torch.uint8),
                "state": torch.randn(2, 3),
            },
            torch.randn(2, 1),
            torch.randn(2),
            torch.zeros(2),
        )
        mc_source.add(
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 1),
            torch.randn(2),
            torch.zeros(2),
        )
    mc_source.sample(2)

    from rl_garden.common.checkpoint import load_replay_buffer_file, save_replay_buffer_file

    dict_path = tmp_path / "dict_buffer.pt"
    save_replay_buffer_file(dict_path, dict_source)
    load_replay_buffer_file(dict_path, dict_target)
    assert torch.equal(dict_target.obs["rgb"], dict_source.obs["rgb"])
    assert torch.equal(dict_target.obs["state"], dict_source.obs["state"])

    mc_path = tmp_path / "mc_buffer.pt"
    save_replay_buffer_file(mc_path, mc_source)
    load_replay_buffer_file(mc_path, mc_target)
    assert mc_target.gamma == 0.7
    assert mc_target._mc_table is not None
    assert torch.equal(mc_target._mc_table, mc_source._mc_table)
