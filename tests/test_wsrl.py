"""Unit tests for WSRL algorithm with CQL/Cal-QL support."""
import pytest
import torch
from gymnasium import spaces
from unittest.mock import MagicMock

from rl_garden.algorithms.wsrl import WSRL


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
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
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
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
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


class TestWSRLHelperMethods:
    """Test helper methods."""

    def test_current_alpha(self, wsrl_agent):
        alpha = wsrl_agent._current_alpha()
        assert alpha.numel() == 1  # Single scalar value
        assert alpha.item() > 0

    def test_current_cql_alpha(self, wsrl_agent):
        cql_alpha = wsrl_agent._current_cql_alpha()
        assert cql_alpha.item() == 1.0  # Fixed value

    def test_current_cql_alpha_autotune(self, wsrl_agent_with_autotune):
        cql_alpha = wsrl_agent_with_autotune._current_cql_alpha()
        assert cql_alpha.item() > 0

    def test_switch_to_online_mode(self, wsrl_agent):
        # Set online parameters
        wsrl_agent.online_cql_alpha = 0.5
        wsrl_agent.online_use_cql_loss = False

        # Switch mode
        wsrl_agent.switch_to_online_mode()

        assert wsrl_agent.cql_alpha == 0.5
        assert not wsrl_agent.use_cql_loss


class TestCQLLossComputation:
    """Test CQL loss computation."""

    def test_sample_ood_actions(self, wsrl_agent):
        obs = torch.randn(8, 4)
        random_actions, current_actions, next_actions = wsrl_agent._sample_ood_actions(obs, 8)

        assert random_actions.shape == (8, 4, 2)  # (batch, n_actions, action_dim)
        assert current_actions.shape == (8, 4, 2)
        assert next_actions.shape == (8, 4, 2)

        # Random actions should be in [-1, 1]
        assert torch.all(random_actions >= -1) and torch.all(random_actions <= 1)

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

    def test_cql_loss_without_calql(self, simple_env):
        agent = WSRL(
            env=simple_env,
            buffer_size=100,
            buffer_device="cpu",
            batch_size=8,
            actor_hidden_dims=(32, 32),
            critic_hidden_dims=(32, 32),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
