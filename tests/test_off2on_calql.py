"""Unit tests for Off2OnCalQL: the original Cal-QL paper's off2on preset.

Mirrors the fixture style of test_wsrl.py, but targets Off2OnCalQL and
focuses on what actually differs from WSRL's defaults: no warmup, mixed
replay retained by default, adaptive mixing ratio, CQL retained online.
"""
import pytest
import torch
from gymnasium import spaces
from unittest.mock import MagicMock

from rl_garden.algorithms.off2on_calql import Off2OnCalQL


@pytest.fixture
def simple_env():
    env = MagicMock()
    env.num_envs = 2
    env.single_observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)
    return env


@pytest.fixture
def off2on_calql_agent(simple_env):
    return Off2OnCalQL(
        env=simple_env,
        buffer_size=100,
        buffer_device="cpu",
        learning_starts=10,
        batch_size=8,
        gamma=0.99,
        tau=0.005,
        training_freq=4,
        utd=1.0,
        net_arch={"pi": [32, 32], "qf": [32, 32]},
        n_critics=4,
        critic_subsample_size=2,
        use_cql_loss=True,
        cql_n_actions=4,
        cql_alpha=1.0,
        cql_autotune_alpha=False,
        use_calql=True,
        calql_bound_random_actions=False,
        device="cpu",
        seed=42,
    )


def _fill_buffer(buffer, num_steps: int, marker: float = 0.0) -> None:
    """Fill buffer with deterministic values; obs[0] = marker for identification."""
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


class TestOff2OnCalQLDefaults:
    """Off2OnCalQL's defaults diverge from WSRL's: no warmup, CQL retained
    online, mixed replay with an adaptive ratio."""

    def test_no_warmup_by_default(self, off2on_calql_agent):
        assert off2on_calql_agent.initial_training_phase is None

    def test_online_cql_retained_by_default(self, off2on_calql_agent):
        assert off2on_calql_agent.online_use_cql_loss is True
        assert off2on_calql_agent.online_cql_alpha == 5.0

    def test_compatible_checkpoint_algorithms_includes_wsrl_lineage(self, off2on_calql_agent):
        assert off2on_calql_agent._compatible_checkpoint_algorithms == (
            "Off2OnCalQL",
            "WSRL",
            "CalQL",
            "CQL",
        )


class TestOff2OnCalQLMixedBatchSampling:
    """Mixed-batch online sampling, including the adaptive ("auto") ratio."""

    def test_switch_to_online_mode_mixed_freezes_offline_buffer(self, off2on_calql_agent):
        _fill_buffer(off2on_calql_agent.replay_buffer, 5, marker=42.0)
        assert off2on_calql_agent.offline_replay_buffer is None

        off2on_calql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio=0.5
        )

        assert off2on_calql_agent.offline_replay_buffer is not None
        assert len(off2on_calql_agent.offline_replay_buffer) > 0
        assert len(off2on_calql_agent.replay_buffer) == 0
        assert off2on_calql_agent.offline_data_ratio == 0.5

    def test_mixed_batch_sample_when_online_empty_uses_all_offline(self, off2on_calql_agent):
        _fill_buffer(off2on_calql_agent.replay_buffer, 5, marker=42.0)
        off2on_calql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio=0.5
        )
        sample = off2on_calql_agent._sample_batch(off2on_calql_agent.batch_size)
        assert sample.obs.shape[0] == off2on_calql_agent.batch_size
        assert torch.all(sample.obs[:, 0] == 42.0)

    def test_mixed_batch_combines_online_and_offline(self, off2on_calql_agent):
        _fill_buffer(off2on_calql_agent.replay_buffer, 5, marker=10.0)
        off2on_calql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio=0.25
        )
        _fill_buffer(off2on_calql_agent.replay_buffer, 5, marker=99.0)
        sample = off2on_calql_agent._sample_batch(off2on_calql_agent.batch_size)
        offline_count = (sample.obs[:, 0] == 10.0).sum().item()
        online_count = (sample.obs[:, 0] == 99.0).sum().item()
        assert offline_count + online_count == off2on_calql_agent.batch_size
        assert offline_count == 2
        assert online_count == 6

    def test_mixed_batch_zero_ratio_uses_only_online(self, off2on_calql_agent):
        _fill_buffer(off2on_calql_agent.replay_buffer, 5, marker=10.0)
        off2on_calql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio=0.0
        )
        _fill_buffer(off2on_calql_agent.replay_buffer, 5, marker=99.0)
        sample = off2on_calql_agent._sample_batch(off2on_calql_agent.batch_size)
        assert torch.all(sample.obs[:, 0] == 99.0)

    def test_mixed_batch_invalid_ratio_raises(self, off2on_calql_agent):
        with pytest.raises(ValueError, match="offline_data_ratio"):
            off2on_calql_agent.switch_to_online_mode(
                online_replay_mode="mixed", offline_data_ratio=1.5
            )

    def test_switch_to_online_mode_accepts_auto_ratio(self, off2on_calql_agent):
        off2on_calql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio="auto"
        )

        assert off2on_calql_agent.offline_data_ratio == "auto"

    def test_mixed_batch_auto_ratio_matches_official_formula(self, off2on_calql_agent):
        # Offline data with marker=10.0 (5 steps * num_envs=2 = 10 transitions)
        _fill_buffer(off2on_calql_agent.replay_buffer, 5, marker=10.0)
        off2on_calql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio="auto"
        )
        # Online data with marker=99.0 (15 steps * num_envs=2 = 30 transitions)
        _fill_buffer(off2on_calql_agent.replay_buffer, 15, marker=99.0)

        # official formula: offline_n / (offline_n + online_n) = 10 / 40 = 0.25
        assert off2on_calql_agent._resolve_offline_data_ratio() == pytest.approx(0.25)

        sample = off2on_calql_agent._sample_batch(off2on_calql_agent.batch_size)
        offline_count = (sample.obs[:, 0] == 10.0).sum().item()
        online_count = (sample.obs[:, 0] == 99.0).sum().item()
        assert offline_count + online_count == off2on_calql_agent.batch_size
        assert offline_count == 2
        assert online_count == 6

    def test_mixed_batch_auto_ratio_all_offline_when_online_empty(self, off2on_calql_agent):
        _fill_buffer(off2on_calql_agent.replay_buffer, 5, marker=42.0)
        off2on_calql_agent.switch_to_online_mode(
            online_replay_mode="mixed", offline_data_ratio="auto"
        )

        assert off2on_calql_agent._resolve_offline_data_ratio() == 1.0
        sample = off2on_calql_agent._sample_batch(off2on_calql_agent.batch_size)
        assert torch.all(sample.obs[:, 0] == 42.0)

    def test_mixed_batch_invalid_ratio_string_raises(self, off2on_calql_agent):
        with pytest.raises(ValueError, match="offline_data_ratio"):
            off2on_calql_agent.switch_to_online_mode(
                online_replay_mode="mixed", offline_data_ratio="bogus"
            )

    def test_concat_replay_samples_preserves_mc_returns(self, off2on_calql_agent):
        from rl_garden.common.types import MCReplayBufferSample

        a = MCReplayBufferSample(
            obs=torch.zeros(2, 4),
            next_obs=torch.zeros(2, 4),
            actions=torch.zeros(2, 2),
            rewards=torch.tensor([1.0, 2.0]),
            dones=torch.zeros(2),
            mc_returns=torch.tensor([10.0, 20.0]),
        )
        b = MCReplayBufferSample(
            obs=torch.ones(3, 4),
            next_obs=torch.ones(3, 4),
            actions=torch.ones(3, 2),
            rewards=torch.tensor([3.0, 4.0, 5.0]),
            dones=torch.zeros(3),
            mc_returns=torch.tensor([30.0, 40.0, 50.0]),
        )
        out = Off2OnCalQL._concat_replay_samples(a, b)
        assert out.obs.shape == (5, 4)
        torch.testing.assert_close(
            out.mc_returns, torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        )


def test_off2on_calql_one_update_smoke(off2on_calql_agent):
    _fill_buffer(off2on_calql_agent.replay_buffer, 5, marker=1.0)
    metrics = off2on_calql_agent.train(1, compute_info=True)
    assert "critic_loss" in metrics
    assert "calql_bound_rate" in metrics
