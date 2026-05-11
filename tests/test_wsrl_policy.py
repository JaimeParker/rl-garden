"""Unit tests for WSRLPolicy with Q-ensemble and CQL support."""
import pytest
import torch
from gymnasium import spaces

from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.sac_policy import SACPolicy
from rl_garden.policies.wsrl_policy import CQLAlphaLagrange, WSRLPolicy


@pytest.fixture
def obs_space():
    return spaces.Box(low=-1, high=1, shape=(10,), dtype=float)


@pytest.fixture
def action_space():
    return spaces.Box(low=-1, high=1, shape=(3,), dtype=float)


@pytest.fixture
def wsrl_policy(obs_space, action_space):
    features_extractor = FlattenExtractor(obs_space)
    return WSRLPolicy(
        observation_space=obs_space,
        action_space=action_space,
        features_extractor=features_extractor,
        net_arch={"pi": [64, 64], "qf": [64, 64]},
        n_critics=10,
        critic_subsample_size=2,
        use_cql_alpha_lagrange=False,
    )


@pytest.fixture
def wsrl_policy_with_legacy_lagrange_args(obs_space, action_space):
    features_extractor = FlattenExtractor(obs_space)
    return WSRLPolicy(
        observation_space=obs_space,
        action_space=action_space,
        features_extractor=features_extractor,
        net_arch={"pi": [64, 64], "qf": [64, 64]},
        n_critics=10,
        critic_subsample_size=2,
        use_cql_alpha_lagrange=True,
        cql_alpha_lagrange_init=5.0,
    )


class TestWSRLPolicyBasics:
    """Test basic policy functionality."""

    def test_policy_creation(self, wsrl_policy):
        assert wsrl_policy.n_critics == 10
        assert wsrl_policy.critic_subsample_size == 2
        assert not wsrl_policy.use_cql_alpha_lagrange

    def test_policy_uniform_std(self, obs_space, action_space):
        features_extractor = FlattenExtractor(obs_space)
        policy = WSRLPolicy(
            observation_space=obs_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch={"pi": [32], "qf": [32]},
            std_parameterization="uniform",
        )
        obs = torch.randn(5, 10)
        action, log_prob, _ = policy.actor_action_log_prob(obs)
        assert action.shape == (5, 3)
        assert log_prob.shape == (5, 1)
        assert policy.actor.log_stds is not None

    def test_policy_layer_norm_options(self, obs_space, action_space):
        features_extractor = FlattenExtractor(obs_space)
        policy = WSRLPolicy(
            observation_space=obs_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch={"pi": [32], "qf": [32]},
            actor_use_layer_norm=True,
            critic_use_layer_norm=True,
        )
        assert any(isinstance(module, torch.nn.LayerNorm) for module in policy.actor.modules())
        # The vmap-fused EnsembleQCritic flattens stacked params and hides the
        # prototype on the meta device, so ``modules()`` won't surface
        # LayerNorm directly. Verify presence by counting parameters against
        # a no-norm baseline — LayerNorm adds 2 params (weight, bias) per
        # hidden layer.
        no_norm_policy = WSRLPolicy(
            observation_space=obs_space,
            action_space=action_space,
            features_extractor=FlattenExtractor(obs_space),
            net_arch={"pi": [32], "qf": [32]},
            actor_use_layer_norm=False,
            critic_use_layer_norm=False,
        )
        critic_params_with = sum(1 for _ in policy.critic.parameters())
        critic_params_without = sum(1 for _ in no_norm_policy.critic.parameters())
        assert critic_params_with == critic_params_without + 2, (
            f"Expected LayerNorm to add 2 params; got {critic_params_with} vs "
            f"{critic_params_without} (baseline)."
        )

    def test_policy_ignores_legacy_cql_alpha_lagrange_args(
        self, wsrl_policy_with_legacy_lagrange_args
    ):
        assert not wsrl_policy_with_legacy_lagrange_args.use_cql_alpha_lagrange
        assert wsrl_policy_with_legacy_lagrange_args.cql_alpha_lagrange is None

    def test_forward_pass(self, wsrl_policy):
        obs = torch.randn(16, 10)
        action = wsrl_policy(obs, deterministic=False)
        assert action.shape == (16, 3)
        assert torch.all(action >= -1) and torch.all(action <= 1)

    def test_deterministic_action(self, wsrl_policy):
        obs = torch.randn(16, 10)
        action = wsrl_policy(obs, deterministic=True)
        assert action.shape == (16, 3)


class TestQEnsemble:
    """Test Q-ensemble functionality."""

    def test_q_values_shape(self, wsrl_policy):
        obs = torch.randn(16, 10)
        actions = torch.randn(16, 3)
        features = wsrl_policy.extract_features(obs)

        q_values = wsrl_policy.q_values(features, actions, target=False)
        assert len(q_values) == 10  # n_critics
        for q in q_values:
            assert q.shape == (16, 1)

    def test_q_values_all(self, wsrl_policy):
        obs = torch.randn(16, 10)
        actions = torch.randn(16, 3)
        features = wsrl_policy.extract_features(obs)

        q_all = wsrl_policy.q_values_all(features, actions, target=False)
        assert q_all.shape == (10, 16, 1)  # (n_critics, batch, 1)

    def test_q_values_target(self, wsrl_policy):
        obs = torch.randn(16, 10)
        actions = torch.randn(16, 3)
        features = wsrl_policy.extract_features(obs)

        q_target = wsrl_policy.q_values_all(features, actions, target=True)
        assert q_target.shape == (10, 16, 1)


class TestCriticSubsampling:
    """Test REDQ critic subsampling."""

    def test_subsampled_shape(self, wsrl_policy):
        obs = torch.randn(16, 10)
        actions = torch.randn(16, 3)
        features = wsrl_policy.extract_features(obs)

        q_subsampled = wsrl_policy.q_values_subsampled(
            features, actions, subsample_size=2, target=True
        )
        assert q_subsampled.shape == (2, 16, 1)

    def test_none_subsample_uses_all_critics(self, wsrl_policy):
        obs = torch.randn(16, 10)
        actions = torch.randn(16, 3)
        features = wsrl_policy.extract_features(obs)

        q_values = wsrl_policy.q_values_subsampled(
            features, actions, subsample_size=None, target=True
        )
        assert q_values.shape == (10, 16, 1)

    def test_subsampled_custom_size(self, wsrl_policy):
        obs = torch.randn(16, 10)
        actions = torch.randn(16, 3)
        features = wsrl_policy.extract_features(obs)

        q_subsampled = wsrl_policy.q_values_subsampled(
            features, actions, subsample_size=5, target=True
        )
        assert q_subsampled.shape == (5, 16, 1)

    def test_min_q_value(self, wsrl_policy):
        obs = torch.randn(16, 10)
        actions = torch.randn(16, 3)
        features = wsrl_policy.extract_features(obs)

        min_q = wsrl_policy.min_q_value(features, actions, subsample_size=2, target=True)
        assert min_q.shape == (16, 1)

    def test_min_q_is_minimum(self, wsrl_policy):
        obs = torch.randn(16, 10)
        actions = torch.randn(16, 3)
        features = wsrl_policy.extract_features(obs)

        # Get all Q-values and subsampled
        q_all = wsrl_policy.q_values_all(features, actions, target=True)
        min_q = wsrl_policy.min_q_value(features, actions, subsample_size=10, target=True)

        # Min should be <= all Q-values
        expected_min = q_all.min(dim=0)[0]
        torch.testing.assert_close(min_q, expected_min)

    def test_sac_policy_supports_redq_subsampling(self, obs_space, action_space):
        policy = SACPolicy(
            observation_space=obs_space,
            action_space=action_space,
            features_extractor=FlattenExtractor(obs_space),
            net_arch={"pi": [32], "qf": [32]},
            n_critics=10,
            critic_subsample_size=2,
        )
        obs = torch.randn(16, 10)
        actions = torch.randn(16, 3)
        features = policy.extract_features(obs)

        q_all = policy.q_values_all(features, actions, target=True)
        q_sub = policy.q_values_subsampled(
            features, actions, subsample_size=policy.critic_subsample_size, target=True
        )
        min_q = policy.min_q_value(features, actions, subsample_size=None, target=False)

        assert q_all.shape == (10, 16, 1)
        assert q_sub.shape == (2, 16, 1)
        assert min_q.shape == (16, 1)


class TestActorCriticIntegration:
    """Test actor-critic integration."""

    def test_actor_action_log_prob(self, wsrl_policy):
        obs = torch.randn(16, 10)
        action, log_prob, features = wsrl_policy.actor_action_log_prob(obs)

        assert action.shape == (16, 3)
        assert log_prob.shape == (16, 1)
        assert features.shape == (16, 10)  # FlattenExtractor output

    def test_stop_gradient(self, wsrl_policy):
        obs = torch.randn(16, 10)
        _, _, features_stopped = wsrl_policy.actor_action_log_prob(
            obs, stop_gradient=True
        )

        # Stop-gradient features should not require grad.
        assert not features_stopped.requires_grad


class TestParameterGroups:
    """Test parameter grouping for optimizers."""

    def test_critic_and_encoder_parameters(self, wsrl_policy):
        params = list(wsrl_policy.critic_and_encoder_parameters())
        assert len(params) > 0

        # Should include critic parameters
        critic_params = set(wsrl_policy.critic.parameters())
        encoder_params = set(wsrl_policy.features_extractor.parameters())
        param_set = set(params)

        assert critic_params.issubset(param_set)
        assert encoder_params.issubset(param_set)

    def test_actor_parameters(self, wsrl_policy):
        params = list(wsrl_policy.actor_parameters())
        assert len(params) > 0

        # Should only include actor parameters
        actor_params = set(wsrl_policy.actor.parameters())
        param_set = set(params)

        assert actor_params == param_set

    def test_cql_alpha_lagrange_parameters(self, wsrl_policy_with_legacy_lagrange_args):
        params = list(wsrl_policy_with_legacy_lagrange_args.cql_alpha_lagrange_parameters())
        assert len(params) == 0

    def test_cql_alpha_lagrange_disabled(self, wsrl_policy):
        params = list(wsrl_policy.cql_alpha_lagrange_parameters())
        assert len(params) == 0


class TestCQLAlphaLagrange:
    """Test CQL alpha Lagrange multiplier."""

    def test_lagrange_forward(self):
        lagrange = CQLAlphaLagrange(init_value=5.0)
        alpha = lagrange()
        assert alpha.shape == ()
        assert alpha.item() > 0

    def test_lagrange_gradient(self):
        lagrange = CQLAlphaLagrange(init_value=5.0)
        alpha = lagrange()
        loss = alpha * 2.0
        loss.backward()

        assert lagrange.log_alpha.grad is not None

    def test_lagrange_not_enabled_raises(self, wsrl_policy):
        with pytest.raises(ValueError, match="owned by CQL/CalQL algorithms"):
            wsrl_policy.get_cql_alpha()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
