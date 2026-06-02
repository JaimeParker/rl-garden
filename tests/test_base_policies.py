from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.policies.base_policies import SACBasePolicy, ZeroBasePolicy


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


def _state_env() -> DummyVecEnv:
    return DummyVecEnv(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )


def _rgb_env() -> DummyVecEnv:
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


def test_zero_base_policy_returns_env_space_zero_actions() -> None:
    env = _state_env()
    provider = ZeroBasePolicy(
        env.single_observation_space,
        env.single_action_space,
        device="cpu",
    )

    output = provider.select_action(torch.ones(env.num_envs, 4))

    assert output.actions.shape == (env.num_envs, 2)
    assert output.actions.device.type == "cpu"
    assert torch.equal(output.actions, torch.zeros(env.num_envs, 2))


def test_sac_base_policy_loads_state_checkpoint(tmp_path) -> None:
    env = _state_env()
    agent = SAC(env=env, **_sac_kwargs())
    path = tmp_path / "sac_state.pt"
    agent.save(path)

    provider = SACBasePolicy.from_checkpoint(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        checkpoint_path=path,
        device="cpu",
    )
    output = provider.select_action(torch.zeros(env.num_envs, 4))

    assert output.actions.shape == (env.num_envs, 2)
    assert torch.all(output.actions <= 1.0)
    assert torch.all(output.actions >= -1.0)


def test_sac_base_policy_loads_rgb_checkpoint(tmp_path) -> None:
    env = _rgb_env()
    agent = SAC(env=env, **_sac_kwargs(), image_keys=("rgb",))
    path = tmp_path / "sac_rgb.pt"
    agent.save(path)

    provider = SACBasePolicy.from_checkpoint(
        observation_space=env.single_observation_space,
        action_space=env.single_action_space,
        checkpoint_path=path,
        device="cpu",
    )
    obs = {
        "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
        "state": torch.zeros(env.num_envs, 4),
    }
    output = provider.select_action(obs)

    assert output.actions.shape == (env.num_envs, 2)


def test_sac_base_policy_rejects_mismatched_action_space(tmp_path) -> None:
    env = _state_env()
    agent = SAC(env=env, **_sac_kwargs())
    path = tmp_path / "sac_state.pt"
    agent.save(path)
    mismatched_action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(3,),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="action_space metadata"):
        SACBasePolicy.from_checkpoint(
            observation_space=env.single_observation_space,
            action_space=mismatched_action_space,
            checkpoint_path=path,
            device="cpu",
        )
