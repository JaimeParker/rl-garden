from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms.residual_hil_serl import ResidualHilSerlSAC
from rl_garden.policies.base_policies import BasePolicyOutput, BasePolicyProvider


class _Env:
    num_envs = 1

    def __init__(self) -> None:
        self.single_observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1, 2), dtype=np.float32)

    def reset(self, seed=None):
        del seed
        return torch.zeros(1, 3), {}

    def step(self, actions):
        del actions
        return (
            torch.zeros(1, 3),
            torch.ones(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )


class _BaseProvider(BasePolicyProvider):
    def select_action(self, obs):
        n = obs.shape[0]
        return BasePolicyOutput(actions=torch.zeros(n, 2, device=obs.device))


def _agent() -> ResidualHilSerlSAC:
    env = _Env()
    return ResidualHilSerlSAC(
        env=env,
        base_action_provider=_BaseProvider(env.single_observation_space, env.single_action_space),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=4,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
    )


def test_init_demo_buffer_sets_residual_offline_slot():
    agent = _agent()

    agent.init_demo_buffer(buffer_size=8, demo_data_ratio=0.25)

    assert agent.offline_replay_buffer is not None
    assert agent.offline_data_ratio == 0.25
    assert len(agent.offline_replay_buffer) == 0


def test_add_demo_transition_requires_and_stores_base_action_fields():
    agent = _agent()
    agent.init_demo_buffer(buffer_size=8, demo_data_ratio=1.0)

    obs = torch.zeros(1, 3)
    action = torch.zeros(1, 2)
    reward = torch.ones(1)
    done = torch.zeros(1, dtype=torch.bool)
    agent.add_demo_transition(
        obs,
        obs,
        action,
        reward,
        done,
        base_actions=torch.full((1, 2), 0.1),
        next_base_actions=torch.full((1, 2), 0.2),
    )

    assert len(agent.offline_replay_buffer) == 1
    sample = agent.offline_replay_buffer.sample(1)
    assert sample.base_actions.shape == (1, 2)
    assert sample.next_base_actions.shape == (1, 2)


def test_init_demo_buffer_rejects_existing_offline_buffer():
    agent = _agent()
    agent.offline_replay_buffer = agent._make_residual_replay_buffer(8)

    with pytest.raises(RuntimeError, match="offline_replay_buffer"):
        agent.init_demo_buffer(buffer_size=8)
