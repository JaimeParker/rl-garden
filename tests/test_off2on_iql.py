"""Unit tests for Off2OnIQL: IQL's off2on preset.

Mirrors the fixture style of test_off2on_calql.py, but targets Off2OnIQL and
focuses on what's specific to it: no online-regularizer-override concept at
all (unlike Cal-QL), no warmup by default, mixed replay with adaptive ratio.
"""
import pytest
import torch
from gymnasium import spaces
from unittest.mock import MagicMock

from rl_garden.algorithms.off2on_iql import Off2OnIQL


@pytest.fixture
def simple_env():
    env = MagicMock()
    env.num_envs = 2
    env.single_observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)
    return env


@pytest.fixture
def off2on_iql_agent(simple_env):
    return Off2OnIQL(
        env=simple_env,
        buffer_size=100,
        buffer_device="cpu",
        learning_starts=10,
        batch_size=8,
        gamma=0.99,
        tau=0.005,
        training_freq=4,
        utd=1.0,
        net_arch={"pi": [32, 32], "qf": [32, 32], "vf": [32, 32]},
        n_critics=3,
        critic_subsample_size=2,
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


class TestOff2OnIQLDefaults:
    def test_no_warmup_by_default(self, off2on_iql_agent):
        assert off2on_iql_agent.initial_training_phase is None

    def test_no_online_regularizer_override_attributes(self, off2on_iql_agent):
        # Unlike Cal-QL, IQL has no online_cql_alpha/online_use_cql_loss concept.
        assert not hasattr(off2on_iql_agent, "online_cql_alpha")
        assert not hasattr(off2on_iql_agent, "online_use_cql_loss")

    def test_compatible_checkpoint_algorithms(self, off2on_iql_agent):
        assert off2on_iql_agent._compatible_checkpoint_algorithms == (
            "Off2OnIQL",
            "IQL",
        )


class TestOff2OnIQLMixedBatchSampling:
    def test_switch_to_online_mode_mixed_freezes_offline_buffer(self, off2on_iql_agent):
        _fill_buffer(off2on_iql_agent.replay_buffer, 5, marker=42.0)
        assert off2on_iql_agent.offline_replay_buffer is None

        off2on_iql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio=0.5
        )

        assert off2on_iql_agent.offline_replay_buffer is not None
        assert len(off2on_iql_agent.offline_replay_buffer) > 0
        assert len(off2on_iql_agent.replay_buffer) == 0
        assert off2on_iql_agent.offline_data_ratio == 0.5

    def test_mixed_batch_combines_online_and_offline(self, off2on_iql_agent):
        _fill_buffer(off2on_iql_agent.replay_buffer, 5, marker=10.0)
        off2on_iql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio=0.25
        )
        _fill_buffer(off2on_iql_agent.replay_buffer, 5, marker=99.0)
        sample = off2on_iql_agent._sample_batch(off2on_iql_agent.batch_size)
        offline_count = (sample.obs[:, 0] == 10.0).sum().item()
        online_count = (sample.obs[:, 0] == 99.0).sum().item()
        assert offline_count + online_count == off2on_iql_agent.batch_size
        assert offline_count == 2
        assert online_count == 6

    def test_mixed_batch_auto_ratio_matches_official_formula(self, off2on_iql_agent):
        _fill_buffer(off2on_iql_agent.replay_buffer, 5, marker=10.0)
        off2on_iql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio="auto"
        )
        _fill_buffer(off2on_iql_agent.replay_buffer, 15, marker=99.0)

        assert off2on_iql_agent._resolve_offline_data_ratio() == pytest.approx(0.25)
        sample = off2on_iql_agent._sample_batch(off2on_iql_agent.batch_size)
        offline_count = (sample.obs[:, 0] == 10.0).sum().item()
        online_count = (sample.obs[:, 0] == 99.0).sum().item()
        assert offline_count == 2
        assert online_count == 6

    def test_mixed_batch_auto_ratio_all_offline_when_online_empty(self, off2on_iql_agent):
        _fill_buffer(off2on_iql_agent.replay_buffer, 5, marker=42.0)
        off2on_iql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio="auto"
        )

        assert off2on_iql_agent._resolve_offline_data_ratio() == 1.0
        sample = off2on_iql_agent._sample_batch(off2on_iql_agent.batch_size)
        assert torch.all(sample.obs[:, 0] == 42.0)

    def test_mixed_batch_invalid_ratio_raises(self, off2on_iql_agent):
        with pytest.raises(ValueError, match="offline_data_ratio"):
            off2on_iql_agent.switch_to_online_mode(
                online_replay_mode="mixed", offline_data_ratio=1.5
            )


def test_off2on_iql_one_update_smoke(off2on_iql_agent):
    _fill_buffer(off2on_iql_agent.replay_buffer, 5, marker=1.0)
    metrics = off2on_iql_agent.train(1, compute_info=True)
    assert "critic_loss" in metrics
    assert "value_loss" in metrics


def test_off2on_iql_checkpoint_roundtrip_restores_weights(tmp_path):
    def _make(**overrides):
        kwargs = dict(
            env=MagicMock(
                num_envs=2,
                single_observation_space=spaces.Box(-1, 1, shape=(4,), dtype=float),
                single_action_space=spaces.Box(-1, 1, shape=(2,), dtype=float),
            ),
            buffer_size=64,
            buffer_device="cpu",
            batch_size=8,
            net_arch={"pi": [16], "qf": [16], "vf": [16]},
            n_critics=3,
            critic_subsample_size=2,
            device="cpu",
            seed=42,
        )
        kwargs.update(overrides)
        return Off2OnIQL(**kwargs)

    agent = _make()
    _fill_buffer(agent.replay_buffer, 5, marker=1.0)
    agent.train(2)

    path = agent.save(tmp_path / "off2on_iql.pt")

    loaded = _make()
    loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key
