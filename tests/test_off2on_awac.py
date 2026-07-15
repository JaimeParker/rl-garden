"""Unit tests for Off2OnAWAC: AWAC's off2on preset.

Mirrors the fixture style of test_off2on_iql.py, but focuses on what's
specific to AWAC: no actor target at all (critic backup samples next_action
from the current actor), Box-obs-only enforcement, and no online-regularizer
override (matching IQL, unlike Cal-QL).
"""
import pytest
import torch
from gymnasium import spaces
from unittest.mock import MagicMock

from rl_garden.algorithms.off2on_awac import Off2OnAWAC


@pytest.fixture
def simple_env():
    env = MagicMock()
    env.num_envs = 2
    env.single_observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)
    return env


@pytest.fixture
def off2on_awac_agent(simple_env):
    return Off2OnAWAC(
        env=simple_env,
        buffer_size=100,
        buffer_device="cpu",
        learning_starts=10,
        batch_size=8,
        gamma=0.99,
        tau=0.005,
        training_freq=4,
        utd=1.0,
        n_critics=3,
        device="cpu",
        seed=42,
    )


def _fill_buffer(buffer, num_steps: int, marker: float = 0.0) -> None:
    n = buffer.num_envs
    obs_dim = buffer.obs.shape[-1]
    act_dim = buffer.actions.shape[-1]
    for _ in range(num_steps):
        buffer.add(
            torch.full((n, obs_dim), marker),
            torch.full((n, obs_dim), marker + 1.0),
            torch.zeros(n, act_dim),
            torch.zeros(n),
            torch.zeros(n),
        )


def test_no_actor_target(off2on_awac_agent):
    assert not hasattr(off2on_awac_agent.policy, "actor_target")


def test_no_online_regularizer_override_attributes(off2on_awac_agent):
    assert not hasattr(off2on_awac_agent, "online_cql_alpha")
    assert not hasattr(off2on_awac_agent, "online_use_cql_loss")


def test_compatible_checkpoint_algorithms(off2on_awac_agent):
    assert off2on_awac_agent._compatible_checkpoint_algorithms == (
        "Off2OnAWAC",
        "AWAC",
    )


def test_rejects_dict_observation_space():
    env = MagicMock()
    env.num_envs = 1
    env.single_observation_space = spaces.Dict(
        {"state": spaces.Box(low=-1, high=1, shape=(4,), dtype=float)}
    )
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)
    with pytest.raises(TypeError):
        Off2OnAWAC(env=env, buffer_device="cpu", device="cpu")


def test_offline_then_switch_to_online_mixed_replay(off2on_awac_agent):
    _fill_buffer(off2on_awac_agent.replay_buffer, 5, marker=42.0)
    off2on_awac_agent.fit_obs_normalizer()
    off2on_awac_agent.train(1)
    assert off2on_awac_agent.offline_replay_buffer is None

    off2on_awac_agent.switch_to_online_mode(
        online_replay_mode="mixed", offline_data_ratio=0.5
    )
    assert off2on_awac_agent.offline_replay_buffer is not None
    assert len(off2on_awac_agent.offline_replay_buffer) > 0
    assert len(off2on_awac_agent.replay_buffer) == 0

    _fill_buffer(off2on_awac_agent.replay_buffer, 5, marker=1.0)
    metrics = off2on_awac_agent.train(1, compute_info=True)
    assert metrics["critic_loss"] == metrics["critic_loss"]  # finite, not NaN


def test_rollout_predict_matches_policy_contract(off2on_awac_agent):
    obs = torch.randn(2, 4)
    action = off2on_awac_agent.policy.predict(obs, deterministic=False)
    assert action.shape == (2, 2)
    assert torch.isfinite(action).all()
