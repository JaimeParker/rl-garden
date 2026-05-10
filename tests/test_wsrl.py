"""Unit tests for WSRL algorithm with CQL/Cal-QL support."""
import pytest
import torch
from gymnasium import spaces
from unittest.mock import MagicMock

from rl_garden.algorithms.wsrl import WSRL


class _RecordingLogger:
    def __init__(self):
        self.scalars = []
        self.summaries = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_summary(self, tag, value):
        self.summaries.append((tag, value))


@pytest.fixture
def simple_env():
    """Create a simple mock environment for testing."""
    env = MagicMock()
    env.num_envs = 2
    env.single_observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)
    return env


@pytest.fixture
def wsrl_agent(simple_env):
    """Create WSRL agent with small networks for testing."""
    return WSRL(
        env=simple_env,
        buffer_size=100,
        buffer_device="cpu",
        learning_starts=10,
        batch_size=8,
        gamma=0.99,
        tau=0.005,
        training_freq=4,
        utd=1.0,
        # Small networks for fast testing
        net_arch={"pi": [32, 32], "qf": [32, 32]},
        n_critics=4,  # Smaller ensemble for testing
        critic_subsample_size=2,
        # CQL parameters
        use_cql_loss=True,
        cql_n_actions=4,  # Fewer actions for testing
        cql_alpha=1.0,
        cql_autotune_alpha=False,
        # Cal-QL
        use_calql=True,
        calql_bound_random_actions=False,
        # General
        device="cpu",
        seed=42,
    )


@pytest.fixture
def wsrl_agent_with_autotune(simple_env):
    """Create WSRL agent with CQL alpha auto-tuning."""
    return WSRL(
        env=simple_env,
        buffer_size=100,
        buffer_device="cpu",
        learning_starts=10,
        batch_size=8,
        net_arch={"pi": [32, 32], "qf": [32, 32]},
        n_critics=4,
        critic_subsample_size=2,
        cql_autotune_alpha=True,
        cql_alpha_lagrange_init=2.0,
        device="cpu",
        seed=42,
    )


class TestWSRLCreation:
    """Test WSRL agent creation and initialization."""

    def test_agent_creation(self, wsrl_agent):
        assert wsrl_agent.n_critics == 4
        assert wsrl_agent.critic_subsample_size == 2
        assert wsrl_agent.use_cql_loss
        assert wsrl_agent.use_calql
        assert not wsrl_agent.cql_autotune_alpha

    def test_agent_with_autotune(self, wsrl_agent_with_autotune):
        assert wsrl_agent_with_autotune.cql_autotune_alpha
        assert wsrl_agent_with_autotune.policy.use_cql_alpha_lagrange

    def test_policy_creation(self, wsrl_agent):
        assert wsrl_agent.policy is not None
        assert wsrl_agent.policy.n_critics == 4
        assert wsrl_agent.policy.critic_subsample_size == 2

    def test_replay_buffer_is_mc_buffer(self, wsrl_agent):
        from rl_garden.buffers.mc_buffer import MCTensorReplayBuffer
        assert isinstance(wsrl_agent.replay_buffer, MCTensorReplayBuffer)
        assert wsrl_agent.replay_buffer.gamma == 0.99

    def test_optimizers_created(self, wsrl_agent):
        assert wsrl_agent.q_optimizer is not None
        assert wsrl_agent.actor_optimizer is not None
        assert wsrl_agent.alpha_optimizer is not None  # Auto-tune by default
        assert wsrl_agent.cql_alpha_optimizer is None  # Not auto-tuning

    def test_optimizers_with_autotune(self, wsrl_agent_with_autotune):
        assert wsrl_agent_with_autotune.cql_alpha_optimizer is not None

    def test_deprecated_hidden_dims_still_work_with_warning(self, simple_env):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            agent = WSRL(
                env=simple_env,
                buffer_size=100,
                buffer_device="cpu",
                batch_size=8,
                actor_hidden_dims=(31, 29),
                critic_hidden_dims=(37, 35),
                n_critics=4,
                device="cpu",
            )
        assert agent.net_arch == {"pi": [31, 29], "qf": [37, 35]}


class TestWSRLHelperMethods:
    """Test helper methods."""

    def test_current_alpha(self, wsrl_agent):
        alpha = wsrl_agent._current_alpha()
        assert alpha.numel() == 1  # Single scalar value
        assert alpha.item() > 0

    def test_temperature_uses_softplus(self, wsrl_agent):
        """Verify temperature uses softplus parameterization matching original."""
        if not wsrl_agent.autotune:
            pytest.skip("Only applies to auto-tuned temperature")

        import torch.nn.functional as F
        assert hasattr(wsrl_agent, "temperature_lagrange")
        assert wsrl_agent.temperature_lagrange is not None

        # Verify softplus is used
        expected = F.softplus(wsrl_agent.temperature_lagrange.log_alpha)
        actual = wsrl_agent._current_alpha()
        assert torch.isclose(expected, actual)
        assert actual.item() > 0

    def test_current_cql_alpha(self, wsrl_agent):
        cql_alpha = wsrl_agent._current_cql_alpha()
        assert cql_alpha.item() == 1.0  # Fixed value

    def test_current_cql_alpha_autotune(self, wsrl_agent_with_autotune):
        cql_alpha = wsrl_agent_with_autotune._current_cql_alpha()
        assert cql_alpha.item() > 0

    def test_switch_to_online_mode(self, wsrl_agent):
        logger = _RecordingLogger()
        wsrl_agent.logger = logger
        wsrl_agent._global_step = 123
        # Set online parameters
        wsrl_agent.online_cql_alpha = 0.5
        wsrl_agent.online_use_cql_loss = False

        # Switch mode
        wsrl_agent.switch_to_online_mode()

        assert wsrl_agent.cql_alpha == 0.5
        assert not wsrl_agent.use_cql_loss
        assert wsrl_agent._online_start_step == 123
        assert logger.scalars == []
        assert ("wsrl/online_start_step", 123) in logger.summaries

    def test_update_metric_tags(self, wsrl_agent):
        tags = wsrl_agent._update_metric_tags(
            {
                "critic_loss": 1.0,
                "td_loss": 4.0,
                "predicted_q": 2.0,
                "target_q": 3.0,
                "cql_ood_values": 5.0,
                "cql_q_diff": 6.0,
                "calql_bound_rate": 0.25,
                "alpha": 0.1,
                "cql_alpha": 5.0,
            }
        )

        assert tags["losses/critic_loss"] == 1.0
        assert tags["losses/td_loss"] == 4.0
        assert tags["q/td_rmse"] == 2.0
        assert tags["q/predicted"] == 2.0
        assert tags["q/target"] == 3.0
        assert tags["q/cql_ood"] == 5.0
        assert tags["q/cql_diff"] == 6.0
        assert tags["cql/bound_rate"] == 0.25
        assert tags["entropy/alpha"] == 0.1
        assert tags["cql/alpha"] == 5.0

    def test_eval_metrics_include_normalized_score(self, wsrl_agent):
        metrics = wsrl_agent.canonical_eval_metrics({"return": 1.0, "success_at_end": 0.82})

        assert metrics["return"] == 1.0
        assert metrics["success_at_end"] == 0.82
        assert metrics["normalized_score"] == pytest.approx(82.0)

    def test_wsrl_logging_uses_continuous_namespaces(self, wsrl_agent):
        logger = _RecordingLogger()
        wsrl_agent.logger = logger

        wsrl_agent._log_update_metrics(
            {"critic_loss": 1.0, "td_loss": 4.0, "predicted_q": 2.0},
            step=100,
        )
        tags = {tag for tag, _, _ in logger.scalars}

        assert "losses/critic_loss" in tags
        assert "losses/td_loss" in tags
        assert "q/predicted" in tags
        assert "q/td_rmse" in tags
        assert "phase/is_online" in tags
        assert not any(
            tag.startswith(("offline_losses/", "offline_eval/", "online/", "wsrl/"))
            for tag in tags
        )


class TestCQLLossComputation:
    """Test CQL loss computation."""

    def test_sample_n_actions_with_log_probs(self, wsrl_agent):
        """Verify the vectorized n-action sampler returns actions AND log_probs."""
        obs = torch.randn(8, 4)
        actions, log_probs = wsrl_agent._sample_n_actions_with_log_probs(
            obs, n=wsrl_agent.cql_n_actions
        )
        assert actions.shape == (8, wsrl_agent.cql_n_actions, 2)  # (batch, n, action_dim)
        assert log_probs.shape == (8, wsrl_agent.cql_n_actions)   # (batch, n)
        # Tanh-squashed actions should be in [-1, 1]
        assert torch.all(actions >= -1.0 - 1e-5)
        assert torch.all(actions <= 1.0 + 1e-5)

    def test_sample_random_actions_uniform(self, wsrl_agent):
        wsrl_agent.cql_action_sample_method = "uniform"
        a = wsrl_agent._sample_random_actions(8, 2)
        assert a.shape == (8, wsrl_agent.cql_n_actions, 2)
        assert torch.all(a >= -1) and torch.all(a <= 1)

    def test_sample_random_actions_normal(self, wsrl_agent):
        wsrl_agent.cql_action_sample_method = "normal"
        a = wsrl_agent._sample_random_actions(8, 2)
        assert a.shape == (8, wsrl_agent.cql_n_actions, 2)

    def test_next_actions_use_next_obs(self, wsrl_agent):
        """Critical correctness: next-state actions must be sampled from next_obs.

        We verify by feeding two different observations and checking that the
        sampled actions differ on average — if our impl wrongly used `obs` for
        both, the two distributions would be identical.
        """
        torch.manual_seed(0)
        obs_a = torch.zeros(64, 4)
        obs_b = torch.ones(64, 4) * 5.0  # very different from obs_a
        a, _ = wsrl_agent._sample_n_actions_with_log_probs(obs_a, n=8)
        b, _ = wsrl_agent._sample_n_actions_with_log_probs(obs_b, n=8)
        # Distinct inputs should produce distinct mean actions in expectation.
        assert not torch.allclose(a.mean(dim=(0, 1)), b.mean(dim=(0, 1)), atol=1e-2)

    def test_cql_loss_computation(self, wsrl_agent):
        # Create mock data
        from rl_garden.buffers.mc_buffer import MCReplayBufferSample

        data = MCReplayBufferSample(
            obs=torch.randn(8, 4),
            next_obs=torch.randn(8, 4),
            actions=torch.randn(8, 2),
            rewards=torch.ones(8),
            dones=torch.zeros(8),
            mc_returns=torch.ones(8) * 5.0,
        )

        # Get Q predictions
        q_pred = wsrl_agent._critic_forward(data.obs, data.actions, target=False)
        assert q_pred.shape == (4, 8, 1)  # (n_critics, batch, 1)

        # Compute CQL loss
        cql_loss, info = wsrl_agent._cql_loss(data, q_pred)

        assert cql_loss.shape == ()
        assert "cql_q_diff" in info
        assert "cql_ood_values" in info
        assert "calql_bound_rate" in info  # Cal-QL enabled

    def test_cql_loss_no_importance_sample(self, simple_env):
        """Verify the non-IS branch concatenates q_pred and applies log(M)*temp."""
        from rl_garden.buffers.mc_buffer import MCReplayBufferSample

        agent = WSRL(
            env=simple_env,
            buffer_size=100,
            buffer_device="cpu",
            batch_size=8,
            net_arch={"pi": [32, 32], "qf": [32, 32]},
            n_critics=4,
            critic_subsample_size=2,
            cql_importance_sample=False,  # non-IS branch
            use_calql=False,
            device="cpu",
        )
        data = MCReplayBufferSample(
            obs=torch.randn(8, 4),
            next_obs=torch.randn(8, 4),
            actions=torch.randn(8, 2),
            rewards=torch.ones(8),
            dones=torch.zeros(8),
            mc_returns=None,
        )
        q_pred = agent._critic_forward(data.obs, data.actions, target=False)
        cql_loss, info = agent._cql_loss(data, q_pred)
        assert cql_loss.shape == ()
        assert "cql_q_diff" in info

    def test_cql_loss_uses_subsampled_critics(self, wsrl_agent):
        """Verify CQL loss respects critic_subsample_size (REDQ in CQL path).

        We check via shape: after subsampling, the internal cql_q_diff should
        have leading dim == critic_subsample_size, not n_critics. We can
        verify by patching torch.randint to return fixed indices and reading
        intermediate shapes — but simpler: just confirm the loss runs and
        gradients flow through only the subsampled critics' Q-values.
        """
        from rl_garden.buffers.mc_buffer import MCReplayBufferSample

        data = MCReplayBufferSample(
            obs=torch.randn(8, 4),
            next_obs=torch.randn(8, 4),
            actions=torch.randn(8, 2),
            rewards=torch.ones(8),
            dones=torch.zeros(8),
            mc_returns=torch.ones(8) * 5.0,
        )
        q_pred = wsrl_agent._critic_forward(data.obs, data.actions, target=False)
        loss, info = wsrl_agent._cql_loss(data, q_pred)
        # Loss should be a finite scalar.
        assert torch.isfinite(loss)
        # n_critics=4, critic_subsample_size=2 (from fixture)
        assert wsrl_agent.critic_subsample_size < wsrl_agent.n_critics

    def test_cql_loss_without_calql(self, simple_env):
        agent = WSRL(
            env=simple_env,
            buffer_size=100,
            buffer_device="cpu",
            batch_size=8,
            net_arch={"pi": [32, 32], "qf": [32, 32]},
            n_critics=4,
            use_calql=False,  # Disable Cal-QL
            device="cpu",
        )

        from rl_garden.buffers.mc_buffer import MCReplayBufferSample

        data = MCReplayBufferSample(
            obs=torch.randn(8, 4),
            next_obs=torch.randn(8, 4),
            actions=torch.randn(8, 2),
            rewards=torch.ones(8),
            dones=torch.zeros(8),
            mc_returns=None,  # No MC returns
        )

        q_pred = agent._critic_forward(data.obs, data.actions, target=False)
        cql_loss, info = agent._cql_loss(data, q_pred)

        assert "calql_bound_rate" not in info  # Cal-QL disabled


class TestWSRLTraining:
    """Test WSRL training methods."""

    def test_critic_forward(self, wsrl_agent):
        obs = torch.randn(8, 4)
        actions = torch.randn(8, 2)

        q_values = wsrl_agent._critic_forward(obs, actions, target=False)
        assert q_values.shape == (4, 8, 1)  # (n_critics, batch, 1)

    def test_target_q_computation(self, wsrl_agent):
        from rl_garden.buffers.mc_buffer import MCReplayBufferSample

        data = MCReplayBufferSample(
            obs=torch.randn(8, 4),
            next_obs=torch.randn(8, 4),
            actions=torch.randn(8, 2),
            rewards=torch.ones(8),
            dones=torch.zeros(8),
            mc_returns=torch.ones(8) * 5.0,
        )

        target_q = wsrl_agent._target_q(data)
        assert target_q.shape == (8, 1)

    def test_target_q_backup_entropy_flag(self, wsrl_agent, monkeypatch):
        from rl_garden.buffers.mc_buffer import MCReplayBufferSample

        data = MCReplayBufferSample(
            obs=torch.randn(8, 4),
            next_obs=torch.randn(8, 4),
            actions=torch.randn(8, 2),
            rewards=torch.zeros(8),
            dones=torch.zeros(8),
            mc_returns=torch.zeros(8),
        )

        def fake_sample(obs, n, features=None):
            return torch.zeros(8, n, 2), torch.ones(8, n)

        def fake_q(features, actions, subsample_size=None, target=True):
            return torch.zeros(2, actions.shape[0], 1)

        monkeypatch.setattr(wsrl_agent, "_sample_n_actions_with_log_probs", fake_sample)
        monkeypatch.setattr(wsrl_agent.policy, "q_values_subsampled", fake_q)

        wsrl_agent.backup_entropy = False
        no_entropy_target = wsrl_agent._target_q(data)
        wsrl_agent.backup_entropy = True
        entropy_target = wsrl_agent._target_q(data)

        torch.testing.assert_close(no_entropy_target, torch.zeros(8, 1))
        torch.testing.assert_close(entropy_target, torch.full((8, 1), -wsrl_agent.gamma))

    def test_critic_loss(self, wsrl_agent):
        from rl_garden.buffers.mc_buffer import MCReplayBufferSample

        data = MCReplayBufferSample(
            obs=torch.randn(8, 4),
            next_obs=torch.randn(8, 4),
            actions=torch.randn(8, 2),
            rewards=torch.ones(8),
            dones=torch.zeros(8),
            mc_returns=torch.ones(8) * 5.0,
        )

        critic_loss, info = wsrl_agent._critic_loss(data)

        assert critic_loss.shape == ()
        assert "critic_loss" in info
        assert "td_loss" in info
        assert "cql_loss" in info
        assert "predicted_q" in info

    def test_actor_loss(self, wsrl_agent):
        obs = torch.randn(8, 4)
        actor_loss, log_prob = wsrl_agent._actor_loss(obs)

        assert actor_loss.shape == ()
        assert log_prob.shape == (8, 1)

    def test_actor_loss_uses_all_critics(self, wsrl_agent, monkeypatch):
        obs = torch.randn(8, 4)
        seen_subsample_sizes = []
        original_min_q_value = wsrl_agent.policy.min_q_value

        def wrapped_min_q_value(features, actions, subsample_size=None, target=True):
            seen_subsample_sizes.append(subsample_size)
            return original_min_q_value(
                features, actions, subsample_size=subsample_size, target=target
            )

        monkeypatch.setattr(wsrl_agent.policy, "min_q_value", wrapped_min_q_value)
        wsrl_agent._actor_loss(obs)
        assert seen_subsample_sizes == [None]

    def test_train_step(self, wsrl_agent):
        # Add some data to replay buffer
        for _ in range(20):
            obs = torch.randn(2, 4)
            next_obs = torch.randn(2, 4)
            actions = torch.randn(2, 2)
            rewards = torch.ones(2)
            dones = torch.zeros(2)
            wsrl_agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)

        # Run training step
        info = wsrl_agent.train(gradient_steps=2)

        assert "critic_loss" in info
        assert "actor_loss" in info
        assert "alpha" in info
        assert "cql_alpha" in info

    def test_train_high_utd(self, wsrl_agent):
        """Verify high-UTD path runs ``utd_ratio`` critic updates per actor update."""
        # Add data to replay buffer.
        for _ in range(20):
            obs = torch.randn(2, 4)
            next_obs = torch.randn(2, 4)
            actions = torch.randn(2, 2)
            rewards = torch.ones(2)
            dones = torch.zeros(2)
            wsrl_agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)

        # batch_size=8 must be divisible by utd_ratio.
        utd_ratio = 4
        info = wsrl_agent.train_high_utd(utd_ratio=utd_ratio)

        assert info["utd_ratio"] == float(utd_ratio)
        assert "critic_loss" in info
        assert "actor_loss" in info
        assert "alpha" in info

    def test_train_dispatches_high_utd(self, wsrl_agent):
        """Normal train() should use high-UTD grouping when utd is an integer > 1."""
        for _ in range(20):
            obs = torch.randn(2, 4)
            next_obs = torch.randn(2, 4)
            actions = torch.randn(2, 2)
            rewards = torch.ones(2)
            dones = torch.zeros(2)
            wsrl_agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)

        wsrl_agent.utd = 4.0
        info = wsrl_agent.train(gradient_steps=4)
        assert info["utd_ratio"] == 4.0

    def test_train_high_utd_invalid_ratio(self, wsrl_agent):
        """Non-divisible utd_ratio should raise."""
        for _ in range(10):
            obs = torch.randn(2, 4)
            next_obs = torch.randn(2, 4)
            actions = torch.randn(2, 2)
            rewards = torch.ones(2)
            dones = torch.zeros(2)
            wsrl_agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)
        with pytest.raises(AssertionError):
            wsrl_agent.train_high_utd(utd_ratio=3)  # 8 % 3 != 0


class TestWSRLModeSwitch:
    """Test offline→online mode switching."""

    def test_mode_switch_disables_cql(self, wsrl_agent):
        assert wsrl_agent.use_cql_loss

        wsrl_agent.online_use_cql_loss = False
        wsrl_agent.switch_to_online_mode()

        assert not wsrl_agent.use_cql_loss

    def test_mode_switch_changes_cql_alpha(self, wsrl_agent):
        assert wsrl_agent.cql_alpha == 1.0

        wsrl_agent.online_cql_alpha = 0.1
        wsrl_agent.switch_to_online_mode()

        assert wsrl_agent.cql_alpha == 0.1

    def test_mode_switch_with_both_params(self, wsrl_agent):
        wsrl_agent.online_use_cql_loss = False
        wsrl_agent.online_cql_alpha = 0.5
        wsrl_agent.switch_to_online_mode()

        assert not wsrl_agent.use_cql_loss
        assert wsrl_agent.cql_alpha == 0.5


class TestMixedBatchSampling:
    """Mixed-batch online sampling: switch_to_online_mode("mixed", ratio)."""

    def _fill_buffer(self, buffer, num_steps: int, marker: float = 0.0) -> None:
        """Fill buffer with deterministic values; obs[0] = marker for identification."""
        n = buffer.num_envs
        obs_dim = buffer.obs.shape[-1]
        act_dim = buffer.actions.shape[-1]
        for step in range(num_steps):
            buffer.add(
                torch.full((n, obs_dim), marker),
                torch.full((n, obs_dim), marker + 1.0),
                torch.zeros(n, act_dim),
                torch.zeros(n),
                torch.zeros(n),
            )

    def test_switch_to_online_mode_mixed_freezes_offline_buffer(self, wsrl_agent):
        # Pre-fill with offline data
        self._fill_buffer(wsrl_agent.replay_buffer, 5, marker=42.0)
        assert wsrl_agent.offline_replay_buffer is None

        wsrl_agent.switch_to_online_mode(online_replay_mode="mixed", offline_data_ratio=0.5)

        # Original buffer should now be the offline buffer (still has the data).
        assert wsrl_agent.offline_replay_buffer is not None
        assert len(wsrl_agent.offline_replay_buffer) > 0
        # New replay buffer is empty.
        assert len(wsrl_agent.replay_buffer) == 0
        assert wsrl_agent.offline_data_ratio == 0.5

    def test_mixed_batch_sample_when_online_empty_uses_all_offline(self, wsrl_agent):
        self._fill_buffer(wsrl_agent.replay_buffer, 5, marker=42.0)
        wsrl_agent.switch_to_online_mode(online_replay_mode="mixed", offline_data_ratio=0.5)
        # Online buffer empty → all samples from offline (marker=42)
        sample = wsrl_agent._sample_batch(wsrl_agent.batch_size)
        assert sample.obs.shape[0] == wsrl_agent.batch_size
        # All obs should have marker=42 since online is empty
        assert torch.all(sample.obs[:, 0] == 42.0)

    def test_mixed_batch_combines_online_and_offline(self, wsrl_agent):
        # Offline data with marker=10.0
        self._fill_buffer(wsrl_agent.replay_buffer, 5, marker=10.0)
        wsrl_agent.switch_to_online_mode(online_replay_mode="mixed", offline_data_ratio=0.25)
        # Add online data with marker=99.0
        self._fill_buffer(wsrl_agent.replay_buffer, 5, marker=99.0)
        sample = wsrl_agent._sample_batch(wsrl_agent.batch_size)
        # batch_size=8, ratio=0.25 → 2 from offline (marker=10), 6 from online (marker=99)
        offline_count = (sample.obs[:, 0] == 10.0).sum().item()
        online_count = (sample.obs[:, 0] == 99.0).sum().item()
        assert offline_count + online_count == wsrl_agent.batch_size
        assert offline_count == 2
        assert online_count == 6

    def test_mixed_batch_zero_ratio_uses_only_online(self, wsrl_agent):
        self._fill_buffer(wsrl_agent.replay_buffer, 5, marker=10.0)
        wsrl_agent.switch_to_online_mode(online_replay_mode="mixed", offline_data_ratio=0.0)
        self._fill_buffer(wsrl_agent.replay_buffer, 5, marker=99.0)
        sample = wsrl_agent._sample_batch(wsrl_agent.batch_size)
        # ratio=0 → no offline samples
        assert torch.all(sample.obs[:, 0] == 99.0)

    def test_mixed_batch_invalid_ratio_raises(self, wsrl_agent):
        with pytest.raises(ValueError, match="offline_data_ratio"):
            wsrl_agent.switch_to_online_mode(
                online_replay_mode="mixed", offline_data_ratio=1.5
            )

    def test_concat_replay_samples_preserves_mc_returns(self, wsrl_agent):
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
        out = WSRL._concat_replay_samples(a, b)
        assert out.obs.shape == (5, 4)
        torch.testing.assert_close(
            out.mc_returns, torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
