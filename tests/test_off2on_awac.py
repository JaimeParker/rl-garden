"""Unit tests for Off2OnAWAC: AWAC's off2on preset.

Mirrors the fixture style of test_off2on_iql.py, but focuses on what's
specific to AWAC: no actor target at all (critic backup samples next_action
from the current actor), Box-obs-only enforcement, and no online-regularizer
override (matching IQL, unlike Cal-QL).
"""
import numpy as np
import pytest
import torch
from gymnasium import spaces
from unittest.mock import MagicMock

from rl_garden.algorithms.off2on_awac import Off2OnAWAC
from rl_garden.observations import ObservationContractError


@pytest.fixture
def simple_env():
    env = MagicMock()
    env.num_envs = 2
    env.single_observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
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
    # buffer.obs is a DictArray({"state": Box}) -- boundary normalization
    # (always on) wraps the env's bare Box into Dict.
    n = buffer.num_envs
    obs_dim = buffer.obs["state"].shape[-1]
    act_dim = buffer.actions.shape[-1]
    for _ in range(num_steps):
        buffer.add(
            {"state": torch.full((n, obs_dim), marker)},
            {"state": torch.full((n, obs_dim), marker + 1.0)},
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


def test_accepts_dict_state_only_observation_space():
    # A pure-state Dict ({"state": Box}, no image keys) is accepted -- AWAC
    # only rejects Dict spaces that carry images. See test_awac.py's
    # equivalent offline-class coverage for the construction-level check;
    # this just confirms the off2on shell doesn't re-add a stricter gate.
    env = MagicMock()
    env.num_envs = 1
    env.single_observation_space = spaces.Dict(
        {"state": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)}
    )
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    agent = Off2OnAWAC(env=env, buffer_device="cpu", device="cpu")
    from rl_garden.buffers.replay_buffer import ReplayBuffer

    assert isinstance(agent.replay_buffer, ReplayBuffer)


def test_rejects_dict_observation_space_with_images():
    env = MagicMock()
    env.num_envs = 1
    env.single_observation_space = spaces.Dict(
        {
            "state": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype="uint8"),
        }
    )
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    with pytest.raises(ObservationContractError):
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
    obs = {"state": torch.randn(2, 4)}
    action = off2on_awac_agent.policy.predict(obs, deterministic=False)
    assert action.shape == (2, 2)
    assert torch.isfinite(action).all()
