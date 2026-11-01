"""Policy graphs with several ``actor`` sinks: each sink owns a contiguous
slice of the action and its own mean head (``SplitDiagGaussianActor``).
A graph with one actor sink stays bit-identical to the reference policies;
slices are ordered by ``action_dims``, validated at parse and at build,
pinned in checkpoints, and gradients stay per head."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest
import torch
import torch.nn as nn
from _decision_fakes import _ComponentMaskVecEnv
from gymnasium import spaces
from test_eval_checkpoint import eval_checkpoint
from test_graph_policy import (
    ACTION_SPACE,
    BATCH,
    HEADS,
    STATE_SPACE,
    WINDOW,
    _actions,
    _as_tuple,
    _assert_same,
    _assert_tensor_dicts_equal,
    _flat_calls,
    _learn_one_window,
    _make_agent,
    _obs,
    _parameter_grads,
    _preflight,
    _recurrent_calls,
    _reference_mlp,
    _reference_recurrent,
    _renamed,
    _seeded,
    _touched,
)

from rl_garden.algorithms import PPO, RecurrentPPO
from rl_garden.common.effective_config import ConfigError
from rl_garden.networks.actor_critic import DiagGaussianActor, SplitDiagGaussianActor
from rl_garden.networks.graph.builtin import mlp_actor_critic, recurrent_actor_critic
from rl_garden.networks.graph.executor import run_flat
from rl_garden.networks.graph.spec import GraphSpec
from rl_garden.policies import GraphPPOPolicy

ACTION_3 = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
ACTION_4 = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
CPU = torch.device("cpu")

# Example (f) of docs/guides/policy-graphs.md: two actor sinks on a GRU, the
# gripper's head a single linear layer. The action is 3-dimensional.
EXAMPLE_F_GRAPH = textwrap.dedent(
    """
    nodes:
      obs:     {type: input}
      encoder: {type: encoder}
      gru:     {type: gru, hidden_size: 8}
      arm:     {type: actor, action_dims: [0, 2]}
      gripper: {type: actor, action_dims: [2, 3], net_arch: []}
      value:   {type: value}
    edges:
      - {from: obs, to: encoder}
      - {from: encoder, to: gru}
      - {from: gru, to: arm}
      - {from: gru, to: gripper}
      - {from: gru, to: value}
    """
)


def _split_graph(
    arm: dict, gripper: dict, *, stateful: bool = False, actor_detach: bool = False
) -> dict:
    """``encoder -> trunk -> core``; ``arm``, ``gripper`` and ``value`` all
    read ``core`` (a GRU when ``stateful``)."""
    core = {"type": "gru", "hidden_size": 8} if stateful else {"type": "mlp", "net_arch": [8]}
    return {
        "nodes": {
            "obs": {"type": "input"},
            "encoder": {"type": "encoder"},
            "trunk": {"type": "mlp", "net_arch": [8]},
            "core": core,
            "arm": {"type": "actor", **arm},
            "gripper": {"type": "actor", **gripper},
            "value": {"type": "value"},
        },
        "edges": [
            {"from": "obs", "to": "encoder"},
            {"from": "encoder", "to": "trunk"},
            {"from": "trunk", "to": "core"},
            {"from": "core", "to": "arm", "detach": actor_detach},
            {"from": "core", "to": "gripper", "detach": actor_detach},
            {"from": "core", "to": "value"},
        ],
    }


ARM = {"action_dims": [0, 2]}
GRIPPER = {"action_dims": [2, 3]}


def _actions3(leading: tuple[int, ...], seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(*leading, 3, generator=generator) * 1.8 - 0.9


def _with_actor_fields(spec: GraphSpec, fields: dict) -> str:
    raw = spec.to_dict()
    raw["nodes"]["actor"].update(fields)
    return GraphSpec.parse(raw).to_yaml()


def _all_zero(module: nn.Module) -> bool:
    return all(
        param.grad is None or torch.count_nonzero(param.grad) == 0
        for param in module.parameters()
    )


# --- 1. one actor sink: bit-identical -----------------------------------------------------


ONE_SINK_FIELDS = [
    pytest.param({}, id="no-fields"),
    pytest.param({"action_dims": [0, 2], "net_arch": [8]}, id="whole-action-and-pi-widths"),
]


@pytest.mark.parametrize("actor_fields", ONE_SINK_FIELDS)
def test_one_actor_sink_flat_policy_is_bit_identical_to_ppo_policy(actor_fields):
    graph = _with_actor_fields(mlp_actor_critic(), actor_fields)
    reference = _seeded(lambda: _reference_mlp(STATE_SPACE, None, "shared_critic_grad"))
    policy = _seeded(lambda: GraphPPOPolicy(STATE_SPACE, ACTION_SPACE, graph=graph, **HEADS))
    assert type(policy.actor) is DiagGaussianActor
    _assert_tensor_dicts_equal(_renamed(policy.state_dict()), reference.state_dict())

    obs, actions = _obs(STATE_SPACE, (BATCH,)), _actions((BATCH,))
    _assert_same(_flat_calls(policy, obs, actions), _flat_calls(reference, obs, actions))
    for candidate in (reference, policy):
        values, log_prob, _ = candidate.evaluate_actions(obs, actions)
        (log_prob.mean() + values.pow(2).mean()).backward()
    _assert_tensor_dicts_equal(_renamed(_parameter_grads(policy)), _parameter_grads(reference))


@pytest.mark.parametrize("actor_fields", ONE_SINK_FIELDS)
def test_one_actor_sink_recurrent_policy_is_bit_identical_to_recurrent_ppo_policy(actor_fields):
    graph = _with_actor_fields(recurrent_actor_critic("gru", hidden_size=8), actor_fields)
    reference = _seeded(lambda: _reference_recurrent("gru", None))
    policy = _seeded(lambda: GraphPPOPolicy(STATE_SPACE, ACTION_SPACE, graph=graph, **HEADS))
    _assert_tensor_dicts_equal(_renamed(policy.state_dict(), "gru"), reference.state_dict())

    obs = _obs(STATE_SPACE, (BATCH,), seed=4)
    hidden = reference.initial_hidden_state(BATCH, CPU)
    with torch.no_grad():
        for _ in range(2):
            *_, hidden = reference.act_recurrent(obs, hidden, torch.zeros(BATCH), True)
    expected = _recurrent_calls(reference, hidden, stop_gradient_actor=True)
    actual = _recurrent_calls(policy, _as_tuple(hidden), stop_gradient_actor=False)
    _assert_same(actual, expected)
    _assert_tensor_dicts_equal(
        _renamed(_parameter_grads(policy), "gru"), _parameter_grads(reference)
    )


@pytest.mark.parametrize("actor_fields", ONE_SINK_FIELDS)
@pytest.mark.parametrize(
    "agent_cls,extra_kwargs,spec,stage",
    [
        pytest.param(PPO, {}, mlp_actor_critic(), None, id="ppo"),
        pytest.param(
            RecurrentPPO,
            {"rnn_type": "gru", "rnn_hidden_size": 8},
            recurrent_actor_critic("gru", hidden_size=8),
            "gru",
            id="recurrent_ppo",
        ),
    ],
)
def test_one_actor_sink_learn_window_is_bit_identical(
    agent_cls, extra_kwargs, spec, stage, actor_fields
):
    def trained(**graph_kwargs) -> dict:
        agent = _make_agent(agent_cls, **extra_kwargs, **graph_kwargs)
        _learn_one_window(agent)
        return agent.policy.state_dict()

    reference = trained()
    with_graph = trained(policy_graph=_with_actor_fields(spec, actor_fields))
    _assert_tensor_dicts_equal(_renamed(with_graph, stage), reference)


# --- 2. the action is the heads' outputs in action_dims order -----------------------------


ORDER_GRAPH = {
    "nodes": {
        "obs": {"type": "input"},
        "encoder": {"type": "encoder"},
        "wide": {"type": "mlp", "net_arch": [6]},
        "narrow": {"type": "mlp", "net_arch": [5]},
        "z_first": {"type": "actor", "action_dims": [0, 3]},
        "a_second": {"type": "actor", "action_dims": [3, 4]},
        "value": {"type": "value"},
    },
    "edges": [
        {"from": "obs", "to": "encoder"},
        {"from": "encoder", "to": "wide"},
        {"from": "encoder", "to": "narrow"},
        {"from": "wide", "to": "z_first"},
        {"from": "narrow", "to": "a_second"},
        {"from": "encoder", "to": "value"},
    ],
}


def test_action_mean_is_the_heads_in_action_dims_order_not_name_order():
    policy = _seeded(lambda: GraphPPOPolicy(STATE_SPACE, ACTION_4, graph=ORDER_GRAPH, **HEADS))
    assert policy.graph_spec.actor_sinks == ("z_first", "a_second")
    actor = policy.actor
    assert isinstance(actor, SplitDiagGaussianActor)
    assert actor.action_slices == ((0, 3), (3, 4))
    assert [head.mean.out_features for head in actor.heads] == [3, 1]

    obs = _obs(STATE_SPACE, (BATCH,))
    outputs = run_flat(policy.graph_spec, policy.graph_nodes, obs, ("wide", "narrow"))
    first_input, second_input = policy.extract_actor_features(obs)
    assert torch.equal(first_input, outputs["wide"])
    assert torch.equal(second_input, outputs["narrow"])
    expected_mean = torch.cat([actor.heads[0](first_input), actor.heads[1](second_input)], dim=-1)

    actions = torch.zeros(BATCH, 4)
    _, _, _, mean, log_std = policy.evaluate_actions_with_dist_params(obs, actions)
    assert torch.equal(mean, expected_mean)
    assert log_std.shape == (BATCH, 4)
    assert torch.equal(policy.predict(obs, deterministic=True), actor.clamp_action(expected_mean))
    _, _, _, _, rollout_mean, _ = policy.act_with_value_logprob_and_dist_params(obs)
    assert torch.equal(rollout_mean, expected_mean)


# --- 3. validation -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm,gripper,match",
    [
        pytest.param(
            {"action_dims": [0, 2]},
            {"action_dims": [1, 3]},
            r"actor sinks 'arm' \(action_dims \[0, 2\]\) and 'gripper' \(action_dims "
            r"\[1, 3\]\) overlap",
            id="overlap",
        ),
        pytest.param(
            {"action_dims": [0, 2]},
            {},
            r"2 actor sinks \['arm', 'gripper'\]: each needs 'action_dims.*missing on "
            r"\['gripper'\]",
            id="missing-on-one-sink",
        ),
        pytest.param(
            {"action_dims": [2, 2]},
            GRIPPER,
            r"actor sink 'arm': 'action_dims' \[start, stop\] needs 0 <= start < stop, "
            r"got \[2, 2\]",
            id="start-not-below-stop",
        ),
        pytest.param(
            {"action_dims": [-1, 2]},
            GRIPPER,
            r"actor sink 'arm': 'action_dims' \[start, stop\] needs 0 <= start < stop",
            id="negative-start",
        ),
        pytest.param(
            ARM,
            {"action_dims": [2, 2.5]},
            r"actor sink 'gripper': 'action_dims' must be \[start, stop\], two integers",
            id="non-integer",
        ),
        pytest.param(
            ARM,
            {"action_dims": [2, 3, 4]},
            r"actor sink 'gripper': 'action_dims' must be \[start, stop\]",
            id="three-numbers",
        ),
        pytest.param(
            ARM,
            {**GRIPPER, "net_arch": [0]},
            r"node 'gripper': 'net_arch' must be a positive integer",
            id="bad-net-arch",
        ),
    ],
)
def test_malformed_action_slices_are_rejected_at_parse(arm, gripper, match):
    with pytest.raises(ConfigError, match=match):
        GraphSpec.parse(_split_graph(arm, gripper))


@pytest.mark.parametrize(
    "arm,gripper,match",
    [
        pytest.param(
            {"action_dims": [0, 1]},
            {"action_dims": [2, 3]},
            r"actor sinks \['arm', 'gripper'\] leave action dims \[1, 2\) uncovered between "
            r"actor sinks 'arm' and 'gripper'",
            id="gap-between",
        ),
        pytest.param(
            {"action_dims": [1, 2]},
            GRIPPER,
            r"leave action dims \[0, 1\) uncovered before actor sink 'arm'",
            id="gap-before",
        ),
        pytest.param(
            {"action_dims": [0, 1]},
            {"action_dims": [1, 2]},
            r"leave action dims \[2, 3\) uncovered after actor sink 'gripper'",
            id="gap-after",
        ),
        pytest.param(
            ARM,
            {"action_dims": [2, 4]},
            r"actor sink 'gripper' has action_dims \[2, 4\], beyond the 3-dim action; "
            r"the action_dims of actor sinks \['arm', 'gripper'\]",
            id="beyond-the-action",
        ),
    ],
)
def test_action_slices_that_do_not_cover_the_action_are_rejected_at_build(arm, gripper, match):
    graph = _split_graph(arm, gripper)
    GraphSpec.parse(graph)
    with pytest.raises(ConfigError, match=match):
        GraphPPOPolicy(STATE_SPACE, ACTION_3, graph=graph, **HEADS)


def test_a_single_actor_sink_must_cover_the_whole_action():
    graph = _with_actor_fields(mlp_actor_critic(), {"action_dims": [0, 2]})
    with pytest.raises(
        ConfigError,
        match=r"actor sink 'actor' has action_dims \[0, 2\], but it is the only actor sink",
    ):
        GraphPPOPolicy(STATE_SPACE, ACTION_3, graph=graph, **HEADS)


def test_every_actor_sink_is_a_sink():
    graph = _split_graph(ARM, GRIPPER)
    graph["nodes"]["extra"] = {"type": "mlp", "net_arch": [4]}
    graph["edges"].append({"from": "gripper", "to": "extra"})
    with pytest.raises(ConfigError, match="sink node 'gripper' has an outgoing edge"):
        GraphSpec.parse(graph)


def test_actor_features_dim_has_no_single_value_with_several_sinks():
    policy = GraphPPOPolicy(STATE_SPACE, ACTION_3, graph=_split_graph(ARM, GRIPPER), **HEADS)
    with pytest.raises(TypeError, match=r"2 actor sinks \['arm', 'gripper'\] of widths \[8, 8\]"):
        policy.actor_features_dim
    assert policy.critic_features_dim == 8


# --- 4. gradient isolation -------------------------------------------------------------------


def test_a_loss_on_one_slice_leaves_the_other_head_untouched():
    policy = _seeded(
        lambda: GraphPPOPolicy(STATE_SPACE, ACTION_3, graph=_split_graph(ARM, GRIPPER), **HEADS)
    )
    obs, actions = _obs(STATE_SPACE, (BATCH,)), _actions3((BATCH,))
    _, log_prob, _ = policy.evaluate_actions(obs, actions, sum_dims=False)
    assert log_prob.shape == (BATCH, 3)
    log_prob[:, 2:].sum().backward()

    arm_head, gripper_head = policy.actor.heads
    assert _all_zero(arm_head)
    assert not _all_zero(gripper_head)
    log_std_grad = policy.actor.log_std.grad
    assert torch.count_nonzero(log_std_grad[0, :2]) == 0
    assert log_std_grad[0, 2] != 0


def _sequence_routing(actor_detach: bool, loss: str) -> dict[str, bool]:
    graph = _split_graph(ARM, GRIPPER, stateful=True, actor_detach=actor_detach)
    policy = _seeded(lambda: GraphPPOPolicy(STATE_SPACE, ACTION_3, graph=graph, **HEADS))
    state = torch.randn(WINDOW, BATCH, 4).requires_grad_()
    values, log_prob, _ = policy.evaluate_actions_sequence(
        {"state": state},
        _actions3((WINDOW, BATCH)),
        policy.initial_hidden_state(BATCH, CPU),
        torch.zeros(WINDOW, BATCH),
    )
    (log_prob.sum() if loss == "actor" else values.sum()).backward()
    return {
        "gru": _touched(policy, "graph_nodes.core."),
        "trunk": _touched(policy, "graph_nodes.trunk."),
        "obs": state.grad is not None and bool(state.grad.abs().sum() > 0),
        "arm": _touched(policy, "actor.heads.0."),
        "gripper": _touched(policy, "actor.heads.1."),
    }


def test_detach_on_both_actor_edges_leaves_the_trunk_to_the_value_loss():
    assert _sequence_routing(actor_detach=True, loss="actor") == {
        "gru": False, "trunk": False, "obs": False, "arm": True, "gripper": True,
    }
    assert _sequence_routing(actor_detach=True, loss="value") == {
        "gru": True, "trunk": True, "obs": True, "arm": False, "gripper": False,
    }
    assert _sequence_routing(actor_detach=False, loss="actor") == {
        "gru": True, "trunk": True, "obs": True, "arm": True, "gripper": True,
    }
    assert _sequence_routing(actor_detach=False, loss="value") == {
        "gru": True, "trunk": True, "obs": True, "arm": False, "gripper": False,
    }


# --- 5. per-sink net_arch ----------------------------------------------------------------------


def _linear_shapes(module: nn.Module) -> list[tuple[int, int]]:
    return [
        (layer.in_features, layer.out_features)
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]


def test_empty_net_arch_is_one_linear_layer_and_the_default_is_the_pi_widths():
    graph = _split_graph(ARM, {**GRIPPER, "net_arch": []})
    policy = GraphPPOPolicy(
        STATE_SPACE, ACTION_3, graph=graph, net_arch={"pi": [16, 6], "vf": [4]}
    )
    arm_head, gripper_head = policy.actor.heads
    assert _linear_shapes(gripper_head) == [(8, 1)]
    assert not any(isinstance(layer, nn.Linear) for layer in gripper_head.trunk.modules())
    assert _linear_shapes(arm_head) == [(8, 16), (16, 6), (6, 2)]
    assert _linear_shapes(policy.value_net) == [(8, 4), (4, 1)]


def test_the_only_actor_sinks_net_arch_replaces_the_pi_widths():
    graph = _with_actor_fields(mlp_actor_critic(), {"net_arch": []})
    policy = GraphPPOPolicy(
        STATE_SPACE, ACTION_SPACE, graph=graph, net_arch={"pi": [16, 6], "vf": [4]}
    )
    assert type(policy.actor) is DiagGaussianActor
    assert _linear_shapes(policy.actor) == [(4, 2)]
    assert _linear_shapes(policy.value_net) == [(4, 4), (4, 1)]


# --- 6. per-dimension initialisation across heads ----------------------------------------------


def test_per_dimension_init_options_land_on_their_global_action_dims():
    policy = GraphPPOPolicy(
        STATE_SPACE,
        ACTION_3,
        graph=_split_graph(ARM, GRIPPER),
        **HEADS,
        log_std_init_per_dim=(-0.5, -1.0, -1.5),
        action_mean_bias_init=(0.1, 0.2, 0.3),
        action_mean_head_gain=0.01,
    )
    arm_head, gripper_head = policy.actor.heads
    assert torch.equal(policy.actor.log_std.detach(), torch.tensor([[-0.5, -1.0, -1.5]]))
    assert torch.equal(arm_head.mean.bias.detach(), torch.tensor([0.1, 0.2]))
    assert torch.equal(gripper_head.mean.bias.detach(), torch.tensor([0.3]))
    for head in (arm_head, gripper_head):
        weight = head.mean.weight.detach()
        rows = weight.shape[0]
        assert torch.allclose(weight @ weight.T, 1e-4 * torch.eye(rows), atol=1e-7)


def test_per_dimension_options_need_the_whole_action_width():
    with pytest.raises(ValueError, match="action_mean_bias_init must have length act_dim=3"):
        GraphPPOPolicy(
            STATE_SPACE,
            ACTION_3,
            graph=_split_graph(ARM, GRIPPER),
            **HEADS,
            action_mean_bias_init=(0.1, 0.2),
        )


# --- 7. checkpoints ------------------------------------------------------------------------------


def _build_recurrent_agent(policy_graph: str):
    from rl_garden.training.online import registry

    builder = eval_checkpoint._builder_from_module(
        eval_checkpoint._training_module("online", "recurrent_ppo"), "recurrent_ppo"
    )
    args = registry.parse_args(
        ["recurrent_ppo", "--policy-graph", policy_graph, "--num-minibatches", "2"]
        + ["--log-type", "none"]
    )
    return builder, builder(args, _ComponentMaskVecEnv(), None, None, None)


def _deterministic_actions(policy) -> torch.Tensor:
    device = next(policy.parameters()).device
    obs = {key: value.to(device) for key, value in _obs(STATE_SPACE, (BATCH,), seed=9).items()}
    hidden = policy.initial_hidden_state(BATCH, device)
    starts = torch.ones(BATCH, device=device)
    actions = []
    with torch.no_grad():
        for _ in range(3):
            action, hidden = policy.predict_recurrent(obs, hidden, starts, deterministic=True)
            actions.append(action.cpu())
            starts = torch.zeros(BATCH, device=device)
    return torch.stack(actions)


def test_split_actor_checkpoint_pins_the_slices_and_round_trips_through_eval_checkpoint(
    tmp_path,
):
    from rl_garden.common.checkpoint import load_checkpoint_file

    builder, agent = _build_recurrent_agent(EXAMPLE_F_GRAPH)
    assert isinstance(agent.policy.actor, SplitDiagGaussianActor)
    path = agent.save(tmp_path / "run" / "checkpoints" / "checkpoint.pt")
    checkpoint = load_checkpoint_file(path, map_location="cpu")
    canonical = checkpoint["metadata"]["hyperparameters"]["policy_graph"]
    assert canonical == agent.policy.graph_spec.to_yaml()
    assert "action_dims" in canonical and "net_arch: []" in canonical
    pinned = GraphSpec.parse(canonical)
    assert pinned.node("arm").params == {"action_dims": [0, 2]}
    assert pinned.node("gripper").params == {"action_dims": [2, 3], "net_arch": []}

    other_slices = EXAMPLE_F_GRAPH.replace("[0, 2]", "[0, 1]").replace("[2, 3]", "[1, 3]")
    _, different = _build_recurrent_agent(other_slices)
    with pytest.raises(ValueError, match="policy graph differs from the current agent's"):
        different.load(path, load_replay_buffer=False)

    eval_args = eval_checkpoint.EvalCheckpointArgs(
        checkpoint_path=str(path), config_path=str(tmp_path / "missing.json")
    )
    restored_args = eval_checkpoint._load_training_args(
        "online", "recurrent_ppo", eval_args, checkpoint
    )
    assert restored_args.policy_graph == canonical
    restored = builder(restored_args, _ComponentMaskVecEnv(), None, None, None)
    restored.load(path, load_replay_buffer=False)
    assert torch.equal(_deterministic_actions(restored.policy), _deterministic_actions(agent.policy))


# --- 8. decision-level recurrent PPO with two sinks ----------------------------------------------


def test_decision_level_recurrent_ppo_trains_both_heads_of_example_f():
    agent = _make_agent(
        RecurrentPPO, env=_ComponentMaskVecEnv(), decision_level=True, policy_graph=EXAMPLE_F_GRAPH
    )
    actor = agent.policy.actor
    assert isinstance(actor, SplitDiagGaussianActor)
    assert agent.policy.graph_spec.actor_sinks == ("arm", "gripper")
    before = {name: p.detach().clone() for name, p in agent.policy.named_parameters()}

    _learn_one_window(agent)

    assert agent.rollout_buffer.window_has_components
    after = dict(agent.policy.named_parameters())
    assert all(torch.isfinite(p).all() for p in after.values())
    for prefix in ("actor.heads.0.", "actor.heads.1.", "graph_nodes.gru."):
        assert any(
            not torch.equal(after[name], before[name]) for name in before if name.startswith(prefix)
        ), prefix


# --- 9. CLI preflight ------------------------------------------------------------------------------


def test_preflight_accepts_two_actor_sinks_and_canonicalizes_them():
    command = _preflight(["recurrent_ppo", "--policy-graph", EXAMPLE_F_GRAPH])
    assert command.args.policy_graph == GraphSpec.parse(EXAMPLE_F_GRAPH).to_yaml()


@pytest.mark.parametrize(
    "change,match",
    [
        pytest.param(
            ("[2, 3]", "[1, 3]"),
            r"'arm' \(action_dims \[0, 2\]\) and 'gripper'.*overlap",
            id="overlap",
        ),
        pytest.param(("action_dims: [0, 2]", "net_arch: [8]"), r"missing on \['arm'\]", id="missing"),
    ],
)
def test_preflight_rejects_malformed_slices_before_any_env(change, match):
    with pytest.raises(ConfigError, match=match):
        _preflight(["recurrent_ppo", "--policy-graph", EXAMPLE_F_GRAPH.replace(*change)])
