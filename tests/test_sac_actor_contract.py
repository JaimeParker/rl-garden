"""The SAC family trains any ``rl_garden.policies.contracts.SACActor``: the
in-tree actors implement it, ``actor_factory`` injects one into every
SACPolicy-building algorithm, construction rejects options the actor cannot
serve, and SAC's entropy-off options give a TD3-shaped update."""

from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np
import pytest
import torch
from gymnasium import spaces
from test_cql_calql import _fill, _kwargs, _offline_env, _offline_kwargs
from test_optimizer_ownership import COMMON_KWARGS, SEQUENCE_KWARGS, TinyVecEnv
from test_sac_optimizer_snapshot import fill_random_transitions
from torch import nn

from rl_garden.algorithms import RLPD, SAC, WSRL, CalQL, CQL, RecurrentSAC, SACFlow
from rl_garden.algorithms.acrlpd import ACRLPD
from rl_garden.algorithms.offline_sac import OfflineSAC
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.networks import (
    DeterministicTanhActor,
    FlowMatchingActor,
    SACActorBase,
    SquashedGaussianActor,
)
from rl_garden.networks.actor_critic import UnsquashedGaussianActor
from rl_garden.policies.contracts import SACActor, SACActorCapabilities
from rl_garden.policies.sac_policy import SACPolicy


class ToyTanhActor(SACActorBase):
    """A Gaussian-on-tanh actor implementing only the two required methods."""

    def __init__(self, features_dim: int, action_space: spaces.Box) -> None:
        super().__init__()
        action_dim = int(np.prod(action_space.shape))
        self.mean = nn.Linear(features_dim, action_dim)
        self.log_scale = nn.Parameter(torch.full((action_dim,), -0.5))

    def action_log_prob(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normal = torch.distributions.Normal(self.mean(features), self.log_scale.exp())
        x = normal.rsample()
        y = torch.tanh(x)
        log_prob = (normal.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return y, log_prob

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean(features))


class ToyPathLikelihoodActor(ToyTanhActor):
    log_prob_is_density = False


class ToyDeterministicActor(SACActorBase):
    stochastic = False

    def __init__(self, features_dim: int, action_space: spaces.Box) -> None:
        super().__init__()
        self.net = nn.Linear(features_dim, int(np.prod(action_space.shape)))

    def action_log_prob(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.deterministic_action(features)
        return action, action.new_zeros(action.shape[0], 1)

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(features))


def flat_policy(action_space: spaces.Box | None = None, **overrides: Any) -> SACPolicy:
    observation_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
    return SACPolicy(
        observation_space,
        action_space or spaces.Box(-1.0, 1.0, (3,), np.float32),
        FlattenExtractor(observation_space),
        net_arch=[16],
        **overrides,
    )


# --- the in-tree actors implement the contract ------------------------------------------------


def _reference_sample_actor_actions(policy: SACPolicy, features: torch.Tensor, n: int):
    """``SACPolicy._sample_actor_actions`` as it was before the SAC actor contract."""
    actor_input = policy._transform_features_for_actor(features)
    mean, log_std = policy.actor(actor_input)
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    x_t = normal.rsample(sample_shape=(n,))
    y_t = torch.tanh(x_t)
    action = y_t * policy.actor.action_scale + policy.actor.action_bias
    log_prob = policy.actor._squashed_log_prob(normal, x_t, y_t).squeeze(-1)
    return action.permute(1, 0, 2), log_prob.permute(1, 0)


def _reference_actor_diagnostics(policy: SACPolicy, obs: torch.Tensor) -> dict[str, torch.Tensor]:
    """``SACPolicy.actor_diagnostics`` as it was before the SAC actor contract."""
    with torch.no_grad():
        features = policy.extract_features(obs, stop_gradient=True)
        actor_input = policy._transform_features_for_actor(features)
        mean, log_std = policy.actor(actor_input)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        gaussian_term = normal.log_prob(x_t).sum(-1)
        tanh_correction = torch.log(policy.actor.action_scale * (1 - y_t.pow(2)) + 1e-6).sum(-1)
        return {
            "policy_std": std.mean(),
            "action_saturation": torch.tanh(mean).abs().mean(),
            "entropy_gaussian": -gaussian_term.mean(),
            "tanh_correction": tanh_correction.mean(),
        }


ASYMMETRIC_ACTIONS = spaces.Box(
    np.array([-2.0, 0.0, -0.3], np.float32), np.array([1.0, 5.0, 0.7], np.float32)
)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"log_std_mode": "clamp", "log_std_min": -20.0}, {"std_parameterization": "uniform"}],
)
def test_gaussian_n_samples_are_bit_identical_to_the_previous_policy_helper(overrides):
    policy = flat_policy(ASYMMETRIC_ACTIONS, **overrides)
    features = torch.randn(4, 5)

    torch.manual_seed(7)
    actions, log_probs = policy._sample_actor_actions(features, 6)
    torch.manual_seed(7)
    expected_actions, expected_log_probs = _reference_sample_actor_actions(policy, features, 6)

    assert actions.shape == (4, 6, 3) and log_probs.shape == (4, 6)
    assert torch.equal(actions, expected_actions)
    assert torch.equal(log_probs, expected_log_probs)


def test_gaussian_diagnostics_are_bit_identical_to_the_previous_policy_diagnostics():
    policy = flat_policy(ASYMMETRIC_ACTIONS)
    obs = torch.randn(4, 5)

    torch.manual_seed(3)
    diagnostics = policy.actor_diagnostics(obs)
    torch.manual_seed(3)
    expected = _reference_actor_diagnostics(policy, obs)

    assert diagnostics.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(diagnostics[key], value), key


def test_uniform_action_bounds_equal_the_gaussian_actor_buffers():
    policy = flat_policy(ASYMMETRIC_ACTIONS)
    low, high = policy._action_bounds(torch.device("cpu"))

    assert torch.equal(low, policy.actor.action_bias - policy.actor.action_scale)
    assert torch.equal(high, policy.actor.action_bias + policy.actor.action_scale)


@pytest.mark.parametrize(
    ("build_actor", "capabilities"),
    [
        (
            lambda space: SquashedGaussianActor(5, space, [8]),
            SACActorCapabilities(True, True, True),
        ),
        (
            lambda space: UnsquashedGaussianActor(5, space, [8]),
            SACActorCapabilities(True, True, True),
        ),
        (
            lambda space: DeterministicTanhActor(5, space, [8]),
            SACActorCapabilities(False, False, False),
        ),
        (
            lambda space: FlowMatchingActor(5, space, [8], denoising_steps=2),
            SACActorCapabilities(True, False, False),
        ),
    ],
)
def test_in_tree_actors_implement_the_sac_actor_contract(build_actor, capabilities):
    space = spaces.Box(-1.0, 1.0, (3,), np.float32)
    actor = build_actor(space)
    assert isinstance(actor, SACActor)
    policy = flat_policy(space, actor_factory=lambda features_dim, action_space: actor)
    assert policy.actor is actor
    assert policy.actor_capabilities == capabilities

    features = torch.randn(4, 5)
    actions, log_probs = actor.sample_actions_with_log_prob(features, 3)
    assert actions.shape == (4, 3, 3) and log_probs.shape == (4, 3)
    assert all(torch.isfinite(value).all() for value in actor.diagnostics(features).values())


def test_deterministic_tanh_actor_samples_its_action_with_zero_log_prob():
    actor = DeterministicTanhActor(5, spaces.Box(-1.0, 1.0, (3,), np.float32), [8])
    features = torch.randn(4, 5)
    action, log_prob = actor.action_log_prob(features)
    assert torch.equal(action, actor.deterministic_action(features))
    assert torch.equal(log_prob, torch.zeros(4, 1))


def test_flow_actor_diagnostics_keep_the_sac_flow_keys():
    actor = FlowMatchingActor(5, spaces.Box(-1.0, 1.0, (3,), np.float32), [8], denoising_steps=2)
    assert set(actor.diagnostics(torch.randn(4, 5))) == {
        "log_prob_mean",
        "log_prob_std",
        "action_norm",
    }


def test_default_policy_keeps_its_checkpoint_keys_and_capabilities():
    policy = flat_policy()
    keys = set(policy.state_dict())
    assert {"actor.action_scale", "actor.action_bias", "actor.fc_mean.weight"} <= keys
    assert isinstance(policy.actor, SquashedGaussianActor)
    assert policy.actor_capabilities == SACActorCapabilities(True, True, True)


def test_factory_actor_skips_the_gaussian_options():
    received: list[tuple[int, spaces.Box]] = []

    def factory(features_dim: int, action_space: spaces.Box) -> nn.Module:
        received.append((features_dim, action_space))
        return ToyTanhActor(features_dim, action_space)

    policy = flat_policy(
        log_std_mode="clamp", log_std_min=-20.0, actor_use_layer_norm=True, actor_factory=factory
    )
    assert isinstance(policy.actor, ToyTanhActor)
    assert received == [(5, policy.action_space)]


def test_factory_returning_a_non_actor_fails_at_construction():
    with pytest.raises(TypeError, match="SAC actor contract"):
        flat_policy(actor_factory=lambda features_dim, action_space: nn.Linear(features_dim, 3))


# --- actor_factory through every SACPolicy-building algorithm ---------------------------------


def _online(algorithm_cls: type, **kwargs: Any):
    agent = algorithm_cls(env=TinyVecEnv(), **COMMON_KWARGS, **kwargs)
    fill_random_transitions(agent)
    return agent


def _offline(algorithm_cls: type, **kwargs: Any):
    agent = algorithm_cls(env=_offline_env(), **_offline_kwargs(), **kwargs)
    _fill(agent)
    return agent


def _offline_sac(**kwargs: Any) -> OfflineSAC:
    agent = OfflineSAC(
        env=_offline_env(), buffer_size=64, buffer_device="cpu", batch_size=8,
        device="cpu", net_arch=[16], **kwargs,
    )
    _fill(agent)
    return agent


def _wsrl(**kwargs: Any) -> WSRL:
    agent = WSRL(env=_offline_env(), **_kwargs(), **kwargs)
    _fill(agent)
    return agent


FACTORY_FAMILIES: dict[str, Callable[..., Any]] = {
    "sac": functools.partial(_online, SAC),
    "rlpd": functools.partial(_online, RLPD),
    "cql": functools.partial(_offline, CQL),
    "calql": functools.partial(_offline, CalQL),
    "wsrl": _wsrl,
    "offline_sac": _offline_sac,
}


@pytest.mark.parametrize("family", sorted(FACTORY_FAMILIES))
def test_factory_actor_trains_and_round_trips_a_checkpoint(family, tmp_path):
    build = FACTORY_FAMILIES[family]
    agent = build(actor_factory=ToyTanhActor)
    assert isinstance(agent.policy.actor, ToyTanhActor)
    actor_before = [p.detach().clone() for p in agent.policy.actor.parameters()]

    info = agent.train(1, compute_info=True)

    assert np.isfinite(info["critic_loss"]) and np.isfinite(info["actor_loss"])
    assert {"log_prob_mean", "log_prob_std", "action_abs_mean"} <= set(info)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, agent.policy.actor.parameters())
    )

    path = agent.save(tmp_path / f"{family}.pt")
    loaded = build(actor_factory=ToyTanhActor)
    loaded.load(path)
    expected = agent.policy.state_dict()
    restored = loaded.policy.state_dict()
    assert restored.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(restored[key], value), key


def test_flow_actor_trains_through_calql_without_importance_sampling():
    flow_factory = functools.partial(FlowMatchingActor, hidden_dims=[16], denoising_steps=2)
    agent = _offline(CalQL, actor_factory=flow_factory, cql_importance_sample=False)
    assert isinstance(agent.policy.actor, FlowMatchingActor)
    assert not agent.policy.actor_capabilities.log_prob_is_density

    info = agent.train(1, compute_info=True)

    assert np.isfinite(info["critic_loss"]) and np.isfinite(info["cql_loss"])
    assert "action_norm" in info


# --- construction-time capability checks ------------------------------------------------------


def test_importance_sampled_cql_rejects_a_path_likelihood_actor():
    with pytest.raises(ValueError, match="cql_importance_sample=True.*log_prob_is_density=False"):
        CQL(env=_offline_env(), actor_factory=ToyPathLikelihoodActor, **_offline_kwargs())


def test_acrlpd_bc_regularizer_rejects_an_actor_that_cannot_score_actions():
    with pytest.raises(ValueError, match="bc_alpha=0.1.*supports_evaluate_log_prob=False"):
        ACRLPD(
            env=TinyVecEnv(), horizon_length=2, bc_alpha=0.1,
            actor_factory=ToyTanhActor, **COMMON_KWARGS,
        )


def test_auto_temperature_rejects_a_deterministic_actor():
    with pytest.raises(ValueError, match="ent_coef='auto'.*stochastic=False"):
        SAC(env=TinyVecEnv(), actor_factory=ToyDeterministicActor, **COMMON_KWARGS)


def test_sac_flow_rejects_an_actor_factory():
    with pytest.raises(ValueError, match="actor_factory"):
        SACFlow(env=TinyVecEnv(), actor_factory=ToyTanhActor, flow_hidden_dims=[8], **COMMON_KWARGS)


def test_sequence_sac_rejects_an_actor_factory():
    with pytest.raises(ValueError, match="actor_factory"):
        RecurrentSAC(
            env=TinyVecEnv(), actor_factory=ToyTanhActor, rnn_hidden_size=8,
            **SEQUENCE_KWARGS, **COMMON_KWARGS,
        )


@pytest.mark.parametrize(
    ("option", "match"),
    [
        ({"actor_q_agg": "max"}, "actor_q_agg"),
        ({"target_policy_noise": -0.1}, "target_policy_noise"),
        ({"target_noise_clip": -0.1}, "target_noise_clip"),
    ],
)
def test_sac_rejects_invalid_entropy_off_options(option, match):
    with pytest.raises(ValueError, match=match):
        SAC(env=TinyVecEnv(), **COMMON_KWARGS, **option)


# --- entropy-off (TD3-shaped) SAC -------------------------------------------------------------


def test_deterministic_actor_trains_with_the_entropy_term_off():
    agent = _online(
        SAC,
        actor_factory=ToyDeterministicActor,
        ent_coef=0.0,
        target_policy_noise=0.2,
        actor_q_agg="first",
    )
    assert not agent.autotune
    assert agent._alpha_step(torch.zeros(4, 1)) is None
    actor_before = [p.detach().clone() for p in agent.policy.actor.parameters()]

    info = agent.train(2, compute_info=True)

    assert "alpha_loss" not in info
    assert info["alpha"] == 0.0
    assert np.isfinite(info["critic_loss"]) and np.isfinite(info["actor_loss"])
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, agent.policy.actor.parameters())
    )


def test_target_policy_noise_is_clipped_and_clamped_to_the_action_bounds():
    agent = _online(SAC, target_policy_noise=5.0, target_noise_clip=0.3)
    action = torch.full((256, 2), 0.9)
    action[:, 1] = 0.0

    torch.manual_seed(0)
    noisy = agent._target_policy_action(action)

    assert not torch.equal(noisy, action)
    assert noisy.max().item() <= 1.0
    assert ((noisy - action).abs() <= 0.3 + 1e-6).all()
    assert torch.isclose(noisy[:, 1].abs().max(), torch.tensor(0.3))


def test_zero_target_policy_noise_draws_no_random_numbers():
    agent = _online(SAC)
    action = torch.rand(4, 2)
    state = torch.random.get_rng_state()

    assert agent._target_policy_action(action) is action
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("aggregation", ["min", "mean", "first"])
def test_actor_q_agg_reduces_the_critic_ensemble(aggregation):
    agent = _online(SAC, actor_q_agg=aggregation, n_critics=3)
    features = torch.randn(4, agent.policy.critic_features_dim)
    action = torch.rand(4, 2) * 2 - 1
    q_all = agent.policy.q_values_all(features, action, target=False)
    expected = {
        "min": q_all.min(dim=0).values,
        "mean": q_all.mean(dim=0),
        "first": q_all[0],
    }[aggregation]

    assert torch.equal(agent._actor_q_value(features, action), expected)


# --- diagnostics without actor-owned action bounds --------------------------------------------


def test_q_landscape_diagnostics_need_no_actor_action_bounds():
    agent = _online(SAC, actor_factory=ToyTanhActor, q_landscape_diagnostics=True)
    assert not hasattr(agent.policy.actor, "action_scale")

    info = agent.train(1, compute_info=True)

    for key in ("q_uniform_var", "q_action_grad_norm", "feature_norm"):
        assert np.isfinite(info[key]), key
