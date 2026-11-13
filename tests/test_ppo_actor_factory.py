"""Plain PPO with a user actor head (``actor_factory``): non-Gaussian actors
train through ``PPO``/``RecurrentPPO``, and the construction-time capability
checks (``PPOPolicy.actor_capabilities``, ``PPO._check_actor_capabilities``)
reject the settings an actor cannot serve.

Two toy actors stand in for user heads (``rl_garden.policies.contracts.PPOActor``):

- ``TanhSquashedActor``: a diagonal Gaussian squashed by ``tanh`` into the
  action bounds. Its log-prob factors per action dimension
  (``sum_dims=False`` works); it has no closed-form entropy (``None``).
- ``TwoStepFlowActor``: a 2-step normalizing flow, ``z0 ~ N(0, I)``, step 1
  ``z1 = L z0`` (``L`` learned unit lower-triangular: volume preserving, mixes
  the dimensions), step 2 ``a = shift(f) + exp(log_scale(f)) * sinh(z1)``.
  Both steps invert in closed form, so the log-prob of a STORED action is
  exact (``asinh``, then a triangular solve). The mixing step makes the
  log-prob not factor per dimension, so ``sum_dims=False`` raises
  ``NotImplementedError``; there is no closed-form entropy (``None``).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
from _decision_fakes import NUM_ENVS, NUM_STEPS, _ComponentMaskVecEnv, _FixedHoldVecEnv
from gymnasium import spaces
from test_graph_losses import _learn, _make_agent
from test_ppo_seams import ACTION_3, STATE_SPACE, TWO_SINK_GRAPH, _RecordingLogger

from rl_garden.algorithms import PPO, RecurrentPPO
from rl_garden.networks.graph.builtin import mlp_actor_critic
from rl_garden.policies import GraphPPOPolicy
from rl_garden.policies.contracts import PPOActorCapabilities

_EPS = 1e-6


class TanhSquashedActor(nn.Module):
    """``a = low + (tanh(u) + 1) / 2 * (high - low)``, ``u ~ N(mean(f), exp(log_scale))``."""

    def __init__(self, features_dim: int, action_space: spaces.Box) -> None:
        super().__init__()
        act_dim = int(np.prod(action_space.shape))
        self.mean = nn.Linear(features_dim, act_dim)
        self.log_scale = nn.Parameter(torch.full((act_dim,), -0.5))
        self.register_buffer("low", torch.as_tensor(action_space.low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(action_space.high, dtype=torch.float32))

    def _base(self, features: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean(features)
        return torch.distributions.Normal(mean, self.log_scale.exp().expand_as(mean))

    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        return self.low + (torch.tanh(u) + 1.0) * 0.5 * (self.high - self.low)

    def _log_prob(
        self, base: torch.distributions.Normal, u: torch.Tensor, sum_dims: bool
    ) -> torch.Tensor:
        half_range = 0.5 * (self.high - self.low)
        log_prob = base.log_prob(u) - torch.log(half_range * (1.0 - torch.tanh(u) ** 2) + _EPS)
        return log_prob.sum(-1, keepdim=True) if sum_dims else log_prob

    def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
        base = self._base(features)
        u = base.mean if deterministic else base.sample()
        return self._squash(u), self._log_prob(base, u, sum_dims), None

    def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
        unit = 2.0 * (actions - self.low) / (self.high - self.low) - 1.0
        u = torch.atanh(unit.clamp(-1.0 + _EPS, 1.0 - _EPS))
        return self._log_prob(self._base(features), u, sum_dims), None

    def deterministic_action(self, features):
        return self._squash(self.mean(features))

    def clamp_action(self, actions):
        return torch.max(torch.min(actions, self.high), self.low)


def _log_cosh(x: torch.Tensor) -> torch.Tensor:
    return x.abs() + torch.log1p(torch.exp(-2.0 * x.abs())) - math.log(2.0)


class TwoStepFlowActor(nn.Module):
    """See the module docstring."""

    def __init__(self, features_dim: int, action_space: spaces.Box) -> None:
        super().__init__()
        self.act_dim = int(np.prod(action_space.shape))
        self.shift = nn.Linear(features_dim, self.act_dim)
        self.log_scale = nn.Linear(features_dim, self.act_dim)
        # Only the strictly lower triangle is read.
        self.mix = nn.Parameter(torch.zeros(self.act_dim, self.act_dim))
        self.register_buffer("low", torch.as_tensor(action_space.low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(action_space.high, dtype=torch.float32))

    def _mixing(self) -> torch.Tensor:
        eye = torch.eye(self.act_dim, device=self.mix.device, dtype=self.mix.dtype)
        return torch.tril(self.mix, diagonal=-1) + eye

    def _affine(self, features):
        return self.shift(features), self.log_scale(features).clamp(-5.0, 2.0)

    @staticmethod
    def _path_log_prob(z0, z1, log_scale) -> torch.Tensor:
        base = torch.distributions.Normal(0.0, 1.0).log_prob(z0).sum(-1, keepdim=True)
        return base - (log_scale + _log_cosh(z1)).sum(-1, keepdim=True)

    def _reject_per_dim(self, sum_dims: bool) -> None:
        if not sum_dims:
            raise NotImplementedError(
                "TwoStepFlowActor's log-prob does not factor per action dimension."
            )

    def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
        self._reject_per_dim(sum_dims)
        shape = (features.shape[0], self.act_dim)
        z0 = (
            torch.zeros(shape, device=features.device, dtype=features.dtype)
            if deterministic
            else torch.randn(shape, device=features.device, dtype=features.dtype)
        )
        z1 = z0 @ self._mixing().T
        shift, log_scale = self._affine(features)
        actions = shift + log_scale.exp() * torch.sinh(z1)
        return actions, self._path_log_prob(z0, z1, log_scale), None

    def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
        self._reject_per_dim(sum_dims)
        shift, log_scale = self._affine(features)
        z1 = torch.asinh((actions - shift) / log_scale.exp())
        z0 = torch.linalg.solve_triangular(
            self._mixing(), z1.T, upper=False, unitriangular=True
        ).T
        return self._path_log_prob(z0, z1, log_scale), None

    def deterministic_action(self, features):
        return self._affine(features)[0]

    def clamp_action(self, actions):
        return torch.max(torch.min(actions, self.high), self.low)


class _ActorWithoutClamp(TanhSquashedActor):
    clamp_action = None


def tanh_squashed_actor(features_dim: int, action_space: spaces.Box) -> nn.Module:
    return TanhSquashedActor(features_dim, action_space)


def two_step_flow_actor(features_dim: int, action_space: spaces.Box) -> nn.Module:
    return TwoStepFlowActor(features_dim, action_space)


def actor_without_clamp(features_dim: int, action_space: spaces.Box) -> nn.Module:
    return _ActorWithoutClamp(features_dim, action_space)


ACTOR_FACTORIES = [
    pytest.param(tanh_squashed_actor, TanhSquashedActor, id="tanh_squashed"),
    pytest.param(two_step_flow_actor, TwoStepFlowActor, id="two_step_flow"),
]


def _assert_trained_with_finite_grads(agent, before: dict[str, torch.Tensor]) -> None:
    after = dict(agent.policy.actor.named_parameters())
    for name, param in after.items():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    assert any(not torch.equal(before[name], after[name]) for name in after)


def record_policy_ratios(agent) -> list[torch.Tensor]:
    """The importance ratio of every minibatch ``agent`` updates on, in
    order (a hook on ``PPO._policy_terms``)."""
    ratios: list[torch.Tensor] = []
    policy_terms = agent._policy_terms

    def recording_policy_terms(**kwargs):
        terms = policy_terms(**kwargs)
        ratios.append(terms.ratio.detach().clone())
        return terms

    agent._policy_terms = recording_policy_terms
    return ratios


def assert_first_ratio_is_one(ratios: list[torch.Tensor]) -> None:
    """Before the first gradient step the policy is the rollout's, so
    ``evaluate_action_log_prob`` of the stored actions must reproduce the
    rollout log-probs: every ratio of the first minibatch is 1."""
    assert ratios, "no minibatch update ran"
    assert (ratios[0] - 1.0).abs().max().item() < 1e-5


def _assert_finite_without_entropy(metrics: dict[str, float]) -> None:
    assert "entropy" not in metrics
    for key in ("loss", "policy_loss", "value_loss", "approx_kl"):
        assert math.isfinite(metrics[key]), key


# --- a user actor through plain PPO ------------------------------------------------------------


@pytest.mark.parametrize("factory, actor_cls", ACTOR_FACTORIES)
def test_a_non_gaussian_actor_trains_through_ppo(factory, actor_cls):
    agent = _make_agent(PPO, actor_factory=factory)
    assert type(agent.policy.actor) is actor_cls
    assert agent.policy.actor_capabilities.has_entropy is False
    assert agent.policy.actor_capabilities.is_gaussian is False
    before = {name: p.detach().clone() for name, p in agent.policy.actor.named_parameters()}
    ratios = record_policy_ratios(agent)

    (metrics,) = _learn(agent)

    assert_first_ratio_is_one(ratios)
    _assert_finite_without_entropy(metrics)
    _assert_trained_with_finite_grads(agent, before)


@pytest.mark.parametrize("factory, actor_cls", ACTOR_FACTORIES)
def test_a_non_gaussian_actor_trains_through_recurrent_ppo(factory, actor_cls):
    agent = _make_agent(RecurrentPPO, rnn_hidden_size=8, actor_factory=factory)
    assert type(agent.policy.actor) is actor_cls
    before = {name: p.detach().clone() for name, p in agent.policy.actor.named_parameters()}

    (metrics,) = _learn(agent)

    _assert_finite_without_entropy(metrics)
    _assert_trained_with_finite_grads(agent, before)


def test_the_actor_factory_reads_the_features_width_and_the_action_space():
    received = []

    def recording_factory(features_dim, action_space):
        received.append((features_dim, action_space))
        return TanhSquashedActor(features_dim, action_space)

    agent = _make_agent(PPO, actor_factory=recording_factory)
    ((features_dim, action_space),) = received
    assert features_dim == agent.policy.actor_extractor.features_dim
    assert action_space == agent.env.single_action_space


def test_the_default_gaussian_actor_reports_every_capability():
    capabilities = _make_agent(PPO).policy.actor_capabilities
    assert capabilities == PPOActorCapabilities(
        has_entropy=True, supports_per_dim_log_prob=True, is_gaussian=True
    )


def test_a_split_actor_of_a_graph_policy_reports_every_capability():
    policy = GraphPPOPolicy(STATE_SPACE, ACTION_3, graph=TWO_SINK_GRAPH, net_arch=[8])
    assert policy.actor_capabilities == PPOActorCapabilities(
        has_entropy=True, supports_per_dim_log_prob=True, is_gaussian=True
    )


# --- construction-time rejections --------------------------------------------------------------


def test_an_actor_missing_a_contract_method_fails_at_construction():
    with pytest.raises(TypeError, match="clamp_action"):
        _make_agent(PPO, actor_factory=actor_without_clamp)


@pytest.mark.parametrize("factory", [tanh_squashed_actor, two_step_flow_actor])
def test_adaptive_kl_needs_a_gaussian_actor(factory):
    with pytest.raises(ValueError, match="--lr-schedule adaptive_kl"):
        _make_agent(PPO, actor_factory=factory, lr_schedule="adaptive_kl")


@pytest.mark.parametrize("factory, actor_cls", ACTOR_FACTORIES)
def test_an_entropy_bonus_needs_an_actor_with_an_entropy(factory, actor_cls):
    with pytest.raises(ValueError, match=rf"--ent-coef.*{actor_cls.__name__}"):
        _make_agent(PPO, actor_factory=factory, ent_coef=0.01)
    assert _make_agent(PPO, actor_factory=factory, ent_coef=0.0).ent_coef == 0.0


def test_decision_level_needs_per_dimension_log_probs():
    with pytest.raises(ValueError, match="decision_level"):
        _make_agent(
            PPO, _FixedHoldVecEnv(), actor_factory=two_step_flow_actor, decision_level=True
        )


def test_actor_factory_and_policy_graph_are_exclusive():
    with pytest.raises(ValueError, match="--policy-graph"):
        _make_agent(PPO, actor_factory=tanh_squashed_actor, policy_graph="builtin:mlp")


def test_a_graph_policy_rejects_an_actor_factory():
    with pytest.raises(ValueError, match="actor_factory"):
        GraphPPOPolicy(
            STATE_SPACE,
            ACTION_3,
            graph=mlp_actor_critic(),
            net_arch=[8],
            actor_factory=tanh_squashed_actor,
        )


# --- decision-level PPO with a non-Gaussian actor ----------------------------------------------


def _decision_level_window(env) -> tuple[PPO, dict[str, float]]:
    agent = _make_agent(
        PPO, env, actor_factory=tanh_squashed_actor, decision_level=True, log_freq=1
    )
    agent.logger = _RecordingLogger()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (metrics,) = _learn(agent)
    return agent, metrics


def test_decision_level_trains_and_logs_a_non_gaussian_actor_without_log_std():
    agent, metrics = _decision_level_window(_FixedHoldVecEnv())
    logged = set(agent.logger.scalars)

    _assert_finite_without_entropy(metrics)
    assert {"train/decisions", "losses/decisions_per_minibatch_mean"} <= logged
    assert "losses/entropy" not in logged
    assert not any(key.startswith("policy/log_std") for key in logged)
    assert agent.logger.scalars["train/decisions"][0] < NUM_ENVS * NUM_STEPS


def test_decision_components_train_a_per_dimension_non_gaussian_actor():
    agent, metrics = _decision_level_window(_ComponentMaskVecEnv())
    logged = set(agent.logger.scalars)

    _assert_finite_without_entropy(metrics)
    assert "train/component_usage/0" in logged
    assert not any(key.startswith("policy/log_std") for key in logged)
