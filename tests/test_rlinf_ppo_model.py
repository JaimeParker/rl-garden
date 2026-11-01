from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.encoders import FlattenExtractor
from rl_garden.integrations.rlinf import ppo_model
from rl_garden.integrations.rlinf.ppo_model import (
    RLGardenPPOModel,
    build_policy_from_cfg,
    register_policy,
    resolve_policy,
)
from rl_garden.policies.ppo_policy import PPOPolicy


def _spaces() -> tuple[spaces.Box, spaces.Box]:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return obs_space, action_space


def _build_model() -> RLGardenPPOModel:
    obs_space, action_space = _spaces()
    policy = PPOPolicy(
        obs_space, action_space, FlattenExtractor(obs_space), net_arch=[16]
    )
    return RLGardenPPOModel(policy)


def test_resolve_policy_returns_expected_class():
    assert resolve_policy("ppo") is PPOPolicy


def test_resolve_policy_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_policy("recurrent_ppo")


def test_default_forward_shapes_are_per_action_dimension_not_summed():
    model = _build_model()
    obs_dim, action_dim = 6, 3
    batch = 5
    states = torch.randn(batch, obs_dim)
    actions = torch.randn(batch, action_dim)

    out = model.default_forward({"states": states, "action": actions})

    assert set(out.keys()) == {"logprobs", "entropy", "values"}
    assert out["logprobs"].shape == (batch, action_dim)
    assert out["entropy"].shape == (batch, action_dim)
    assert out["values"].shape == (batch, 1)


def test_default_forward_respects_compute_flags():
    model = _build_model()
    states = torch.randn(4, 6)
    out = model.default_forward(
        {"states": states}, compute_logprobs=False, compute_entropy=False
    )
    assert set(out.keys()) == {"values"}


def test_predict_action_batch_shapes_and_prev_logprobs_match_default_forward():
    model = _build_model()
    batch = 4
    states = torch.randn(batch, 6)

    chunk_actions, result = model.predict_action_batch(
        {"states": states}, mode="train"
    )

    assert chunk_actions.shape == (batch, 1, 3)  # [B, num_action_chunks, action_dim]
    assert result["prev_values"].shape == (batch, 1)
    assert result["forward_inputs"]["action"].shape == (batch, 3)
    assert torch.equal(result["forward_inputs"]["states"], states)

    # prev_logprobs must match default_forward's unsummed shape -- a
    # mismatch here would silently break the PPO ratio computation
    # (exp(logprobs - prev_logprobs)) between rollout and training time.
    train_out = model.default_forward(
        {"states": states, "action": result["forward_inputs"]["action"]}
    )
    assert result["prev_logprobs"].shape == train_out["logprobs"].shape


def test_predict_action_batch_eval_mode_is_deterministic():
    model = _build_model()
    states = torch.randn(3, 6)
    actions1, _ = model.predict_action_batch({"states": states}, mode="eval")
    actions2, _ = model.predict_action_batch({"states": states}, mode="eval")
    assert torch.equal(actions1, actions2)


def test_value_head_attribute_reachable_for_optimizer_split():
    """Regression test for FSDPModelManager.build_optimizer's substring match.

    build_optimizer buckets parameters into a separate LR group by
    substring-matching "value_head"/"model.value_head" against
    model.named_parameters(). Since value_head is an alias for
    policy.value_net's own submodule (the same tensors reachable via two
    attribute paths), named_parameters()'s default duplicate-parameter
    dedup can silently drop the "value_head"-named path entirely depending
    on attribute-assignment order -- this asserts it doesn't.
    """
    model = _build_model()
    names = [name for name, _ in model.named_parameters()]
    value_head_names = [n for n in names if "value_head" in n]
    assert value_head_names, (
        "no parameter name contains 'value_head' -- FSDPModelManager.build_"
        "optimizer's LR-group split would silently misroute the value head "
        "into the actor's optimizer group"
    )
    # every value_head-matched parameter must be a real, non-empty tensor
    # shared with policy.value_net (not an accidental empty/copy alias).
    value_head_params = dict(model.named_parameters())
    for name in value_head_names:
        assert value_head_params[name].numel() > 0


def test_build_policy_from_cfg_constructs_expected_policy():
    class _Cfg:
        """Minimal DictConfig-like fake: attribute and .get() access."""

        def __init__(self, **kwargs):
            self._data = kwargs

        def __getattr__(self, key):
            try:
                return self._data[key]
            except KeyError:
                raise AttributeError(key) from None

        def get(self, key, default=None):
            return self._data.get(key, default)

    cfg = _Cfg(obs_dim=6, action_dim=3, rlgarden_policy="ppo", rlgarden_policy_kwargs={})
    policy = build_policy_from_cfg(cfg)
    assert isinstance(policy, PPOPolicy)
    assert policy.observation_space.shape == (6,)
    assert policy.action_space.shape == (3,)


class _DownstreamPolicy(PPOPolicy):
    """Stands in for a policy class defined in another package."""


def test_register_policy_makes_a_policy_buildable_from_cfg(monkeypatch):
    monkeypatch.setattr(ppo_model, "_POLICIES", dict(ppo_model._POLICIES))

    register_policy("downstream_ppo", _DownstreamPolicy)

    assert resolve_policy("downstream_ppo") is _DownstreamPolicy
    cfg = _GraphCfg(
        obs_dim=6,
        action_dim=3,
        rlgarden_policy="downstream_ppo",
        rlgarden_policy_kwargs={"net_arch": [16]},
    )
    policy = build_policy_from_cfg(cfg)
    assert type(policy) is _DownstreamPolicy
    assert policy.action_space.shape == (3,)


def test_register_policy_rejects_a_taken_name(monkeypatch):
    monkeypatch.setattr(ppo_model, "_POLICIES", dict(ppo_model._POLICIES))

    with pytest.raises(ValueError, match="already registered"):
        register_policy("ppo", _DownstreamPolicy)
    assert resolve_policy("ppo") is PPOPolicy


class _GraphCfg:
    """Minimal DictConfig-like fake: attribute and .get() access."""

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key) from None

    def get(self, key, default=None):
        return self._data.get(key, default)


def _graph_cfg(graph: dict) -> _GraphCfg:
    return _GraphCfg(
        obs_dim=6,
        action_dim=3,
        rlgarden_policy="graph_ppo",
        rlgarden_policy_kwargs={"graph": graph, "net_arch": [16]},
    )


_FLAT_GRAPH = {
    "nodes": {
        "obs": {"type": "input"},
        "encoder": {"type": "encoder"},
        "trunk": {"type": "mlp", "net_arch": [8]},
        "actor": {"type": "actor"},
        "value": {"type": "value"},
    },
    "edges": [
        {"from": "obs", "to": "encoder"},
        {"from": "encoder", "to": "trunk"},
        {"from": "trunk", "to": "actor"},
        {"from": "trunk", "to": "value"},
    ],
}


def test_build_policy_from_cfg_builds_a_graph_policy_over_flat_states():
    from rl_garden.policies.graph_ppo_policy import GraphPPOPolicy

    assert resolve_policy("graph_ppo") is GraphPPOPolicy
    first = build_policy_from_cfg(_graph_cfg(_FLAT_GRAPH))
    second = build_policy_from_cfg(_graph_cfg(_FLAT_GRAPH))
    assert isinstance(first, GraphPPOPolicy)
    assert list(first.state_dict()) == list(second.state_dict())

    batch = 4
    states, actions = torch.randn(batch, 6), torch.randn(batch, 3).clamp(-0.9, 0.9)
    sampled, values, log_prob, _ = first.forward(states, sum_dims=False)
    assert sampled.shape == (batch, 3) and values.shape == (batch, 1)
    assert log_prob.shape == (batch, 3)
    values, log_prob, entropy = first.evaluate_actions(states, actions, sum_dims=False)
    assert log_prob.shape == entropy.shape == (batch, 3)
    assert first.predict_values(states).shape == (batch, 1)

    model = RLGardenPPOModel(first)
    out = model.default_forward({"states": states, "action": actions})
    assert out["logprobs"].shape == (batch, 3) and out["values"].shape == (batch, 1)
    chunk_actions, result = model.predict_action_batch({"states": states})
    assert chunk_actions.shape == (batch, 1, 3)
    assert result["prev_logprobs"].shape == (batch, 3)


_TWO_ACTOR_SINK_GRAPH = {
    "nodes": {
        "obs": {"type": "input"},
        "encoder": {"type": "encoder"},
        "trunk": {"type": "mlp", "net_arch": [8]},
        "arm": {"type": "actor", "action_dims": [0, 2]},
        "gripper": {"type": "actor", "action_dims": [2, 3], "net_arch": []},
        "value": {"type": "value"},
    },
    "edges": [
        {"from": "obs", "to": "encoder"},
        {"from": "encoder", "to": "trunk"},
        {"from": "trunk", "to": "arm"},
        {"from": "encoder", "to": "gripper"},
        {"from": "trunk", "to": "value"},
    ],
}


def test_build_policy_from_cfg_builds_a_graph_policy_with_two_actor_sinks():
    from rl_garden.networks.actor_critic import SplitDiagGaussianActor

    policy = build_policy_from_cfg(_graph_cfg(_TWO_ACTOR_SINK_GRAPH))
    assert isinstance(policy.actor, SplitDiagGaussianActor)
    assert policy.actor.action_slices == ((0, 2), (2, 3))

    batch = 4
    states, actions = torch.randn(batch, 6), torch.randn(batch, 3).clamp(-0.9, 0.9)
    sampled, values, log_prob, _ = policy.forward(states, sum_dims=False)
    assert sampled.shape == (batch, 3) and values.shape == (batch, 1)
    assert log_prob.shape == (batch, 3)
    values, log_prob, entropy = policy.evaluate_actions(states, actions, sum_dims=False)
    assert log_prob.shape == entropy.shape == (batch, 3)
    assert policy.predict_values(states).shape == (batch, 1)

    model = RLGardenPPOModel(policy)
    out = model.default_forward({"states": states, "action": actions})
    assert out["logprobs"].shape == (batch, 3) and out["values"].shape == (batch, 1)
    rollout_actions, result = model.predict_action_batch({"states": states})
    assert rollout_actions.shape == (batch, 1, 3)
    assert result["prev_logprobs"].shape == (batch, 3)


def test_build_policy_from_cfg_rejects_a_stateful_graph():
    from rl_garden.networks.graph.builtin import recurrent_actor_critic

    with pytest.raises(ValueError, match="needs a stateless graph"):
        build_policy_from_cfg(_graph_cfg(recurrent_actor_critic("gru", hidden_size=8).to_dict()))
