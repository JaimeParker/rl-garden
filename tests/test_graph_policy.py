"""``GraphPPOPolicy`` and the ``policy_graph`` plumbing: the built-in graphs
are bit-identical to ``PPOPolicy``/``RecurrentPPOPolicy``, per-edge
``detach`` routes gradients, several stateful nodes share one flat hidden
state, and the algorithms, checkpoints and CLI preflight handle graphs."""

from __future__ import annotations

import textwrap
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from _decision_fakes import NUM_ENVS, NUM_STEPS, _ComponentMaskVecEnv, _FixedHoldVecEnv
from gymnasium import spaces

from rl_garden.algorithms import PPO, RecurrentPPO, TransformerPPO
from rl_garden.buffers.recurrent_rollout_buffer import _index_recurrent_state
from rl_garden.common.effective_config import ConfigError
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.factory import build_observation_encoder
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.networks import GTrXLLatentEncoder, RecurrentLatentEncoder
from rl_garden.networks.graph.builtin import mlp_actor_critic, recurrent_actor_critic
from rl_garden.networks.graph.registry import GRAPH_NODE_REGISTRY, register_graph_node
from rl_garden.networks.graph.spec import GraphSpec
from rl_garden.policies import GraphPPOPolicy, PPOPolicy, RecurrentPPOPolicy

SEED = 3
HEADS = {"net_arch": [8]}
ACTION_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
STATE_SPACE = spaces.Dict({"state": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)})
IMAGE_SPACE = spaces.Dict(
    {
        "state": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        "rgb_cam": spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8),
    }
)
GTRXL_PARAMS = {
    "embed_dim": 8,
    "head_dim": 4,
    "num_heads": 2,
    "num_layers": 1,
    "mlp_num": 1,
    "memory_len": 2,
}
STAGE_PARAMS = {"gru": {"hidden_size": 8}, "lstm": {"hidden_size": 8}, "gtrxl": GTRXL_PARAMS}
TRANSFORMER_KWARGS = {
    "embed_dim": 8,
    "head_dim": 4,
    "num_heads": 2,
    "num_transformer_layers": 1,
    "mlp_num": 1,
    "memory_len": 2,
}
IMAGE_ENCODER = EncoderConfig(features_dim=16, plain_conv_pooling="gap")
BATCH = 4
WINDOW = 5


# --- helpers -----------------------------------------------------------------------


def _seeded(build):
    torch.manual_seed(SEED)
    return build()


def _renamed(tensors: dict, stage: str | None = None) -> dict:
    """Graph-policy names -> reference-policy names."""
    renames = {
        "graph_nodes.encoder.": "actor_extractor.",
        "graph_nodes.actor_encoder.": "actor_extractor.",
        "graph_nodes.critic_encoder.": "critic_extractor.",
    }
    if stage is not None:
        renames[f"graph_nodes.{stage}."] = "recurrent_encoder."
    out = {}
    for key, value in tensors.items():
        for old, new in renames.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        out[key] = value
    return out


def _assert_tensor_dicts_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        if tensor is None:
            assert actual[name] is None, name
        else:
            assert torch.equal(actual[name], tensor), name


def _assert_same(actual, expected) -> None:
    """Recursive ``torch.equal``; a graph's 1-tuple hidden state matches the
    reference's bare GRU tensor."""
    if isinstance(expected, torch.Tensor):
        if isinstance(actual, tuple) and len(actual) == 1:
            actual = actual[0]
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_part, expected_part in zip(actual, expected):
            _assert_same(actual_part, expected_part)
    else:
        assert actual == expected


def _as_tuple(state):
    return state if isinstance(state, tuple) else (state,)


def _obs(space, leading: tuple[int, ...], seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    obs = {"state": torch.randn(*leading, 4, generator=generator)}
    if "rgb_cam" in space.spaces:
        obs["rgb_cam"] = torch.randint(
            0, 256, (*leading, 16, 16, 3), generator=generator, dtype=torch.uint8
        )
    return obs


def _actions(leading: tuple[int, ...], seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(*leading, 2, generator=generator) * 1.8 - 0.9


def _parameter_grads(policy: nn.Module) -> dict:
    return {
        name: None if param.grad is None else param.grad.clone()
        for name, param in policy.named_parameters()
    }


def _make_agent(agent_cls, env=None, **kwargs):
    defaults = dict(
        device="cpu",
        num_steps=NUM_STEPS,
        num_minibatches=2,
        update_epochs=2,
        eval_freq=0,
        log_freq=0,
        std_log=False,
        target_kl=None,
        net_arch=[8],
        seed=0,
    )
    defaults.update(kwargs)
    return agent_cls(env if env is not None else _FixedHoldVecEnv(emit_decisions=False), **defaults)


def _learn_one_window(agent) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent.learn(total_timesteps=agent.num_envs * agent.num_steps)


# --- P2: flat built-in graph == PPOPolicy ------------------------------------------------


def _reference_mlp(space, encoder_config, sharing) -> PPOPolicy:
    actor_extractor = build_observation_encoder(space, encoder_config)
    critic_extractor = (
        build_observation_encoder(space, encoder_config) if sharing == "separate" else None
    )
    return PPOPolicy(
        space,
        ACTION_SPACE,
        actor_extractor,
        critic_extractor=critic_extractor,
        encoder_sharing=sharing,
        **HEADS,
    )


def _graph_mlp(space, encoder_config, sharing) -> GraphPPOPolicy:
    return GraphPPOPolicy(
        space, ACTION_SPACE, graph=mlp_actor_critic(encoder_config, sharing), **HEADS
    )


def _flat_calls(policy, obs, actions) -> list:
    results = []
    for deterministic in (False, True):
        for sum_dims in (True, False):
            torch.manual_seed(11)
            results.append(policy(obs, deterministic=deterministic, sum_dims=sum_dims))
            torch.manual_seed(11)
            results.append(
                policy.act_with_value_logprob_and_dist_params(
                    obs, deterministic=deterministic, sum_dims=sum_dims
                )
            )
        torch.manual_seed(11)
        results.append(policy.predict(obs, deterministic=deterministic))
    for sum_dims in (True, False):
        results.append(policy.evaluate_actions(obs, actions, sum_dims=sum_dims))
        results.append(policy.evaluate_actions_with_dist_params(obs, actions, sum_dims=sum_dims))
    results.append(policy.predict_values(obs))
    torch.manual_seed(11)
    results.append(policy.act_with_value_and_logprob(obs))
    return results


FLAT_CASES = [
    pytest.param(STATE_SPACE, None, "shared_critic_grad", id="state-shared_critic_grad"),
    pytest.param(
        STATE_SPACE, EncoderConfig(normalize_obs=True), "shared_critic_grad", id="state-normalized"
    ),
    pytest.param(STATE_SPACE, None, "shared", id="state-shared"),
    pytest.param(STATE_SPACE, None, "separate", id="state-separate"),
    pytest.param(IMAGE_SPACE, IMAGE_ENCODER, "shared", id="image-shared"),
    pytest.param(IMAGE_SPACE, IMAGE_ENCODER, "separate", id="image-separate"),
]


@pytest.mark.parametrize("space,encoder_config,sharing", FLAT_CASES)
def test_builtin_mlp_graph_is_bit_identical_to_ppo_policy(space, encoder_config, sharing):
    reference = _seeded(lambda: _reference_mlp(space, encoder_config, sharing))
    graph = _seeded(lambda: _graph_mlp(space, encoder_config, sharing))
    _assert_tensor_dicts_equal(_renamed(graph.state_dict()), reference.state_dict())

    obs, actions = _obs(space, (BATCH,)), _actions((BATCH,))
    for policy in (reference, graph):
        policy.update_normalizer(obs)
    _assert_tensor_dicts_equal(_renamed(graph.state_dict()), reference.state_dict())
    _assert_same(_flat_calls(graph, obs, actions), _flat_calls(reference, obs, actions))

    for policy in (reference, graph):
        values, log_prob, _ = policy.evaluate_actions(obs, actions)
        (log_prob.mean() + values.pow(2).mean()).backward()
    _assert_tensor_dicts_equal(_renamed(_parameter_grads(graph)), _parameter_grads(reference))


# --- P2: recurrent built-in graphs == RecurrentPPOPolicy -------------------------------


def _stage_module(kind: str, input_dim: int) -> nn.Module:
    if kind == "gtrxl":
        return GTrXLLatentEncoder(input_dim, **GTRXL_PARAMS)
    return RecurrentLatentEncoder(input_dim, rnn_type=kind, **STAGE_PARAMS[kind])


def _reference_recurrent(kind, encoder_config) -> RecurrentPPOPolicy:
    extractor = build_observation_encoder(STATE_SPACE, encoder_config)
    return RecurrentPPOPolicy(
        STATE_SPACE,
        ACTION_SPACE,
        extractor,
        _stage_module(kind, extractor.features_dim),
        encoder_sharing="shared_critic_grad",
        **HEADS,
    )


def _graph_recurrent(kind, encoder_config) -> GraphPPOPolicy:
    graph = recurrent_actor_critic(kind, encoder_config, **STAGE_PARAMS[kind])
    return GraphPPOPolicy(STATE_SPACE, ACTION_SPACE, graph=graph, **HEADS)


def _recurrent_calls(policy, hidden, stop_gradient_actor) -> list:
    obs_step = _obs(STATE_SPACE, (BATCH,), seed=5)
    starts = torch.tensor([0.0, 1.0, 0.0, 1.0])
    obs_seq = _obs(STATE_SPACE, (WINDOW, BATCH), seed=6)
    actions_seq = _actions((WINDOW, BATCH), seed=7)
    starts_seq = torch.zeros(WINDOW, BATCH)
    starts_seq[2, 1] = 1.0
    starts_seq[3, 3] = 1.0
    results = []
    for deterministic in (False, True):
        for sum_dims in (True, False):
            torch.manual_seed(11)
            results.append(
                policy.act_recurrent(
                    obs_step,
                    hidden,
                    starts,
                    deterministic,
                    stop_gradient_actor=stop_gradient_actor,
                    sum_dims=sum_dims,
                )
            )
            torch.manual_seed(11)
            results.append(
                policy.act_recurrent_with_dist_params(
                    obs_step,
                    hidden,
                    starts,
                    deterministic,
                    stop_gradient_actor=stop_gradient_actor,
                    sum_dims=sum_dims,
                )
            )
        torch.manual_seed(11)
        results.append(policy.predict_recurrent(obs_step, hidden, starts, deterministic))
    results.append(policy.predict_values_recurrent(obs_step, hidden, starts))
    for sum_dims in (True, False):
        results.append(
            policy.evaluate_actions_sequence(
                obs_seq,
                actions_seq,
                hidden,
                starts_seq,
                stop_gradient_actor=stop_gradient_actor,
                sum_dims=sum_dims,
            )
        )
        results.append(
            policy.evaluate_actions_sequence_with_dist_params(
                obs_seq,
                actions_seq,
                hidden,
                starts_seq,
                stop_gradient_actor=stop_gradient_actor,
                sum_dims=sum_dims,
            )
        )
    values, log_prob, _ = policy.evaluate_actions_sequence(
        obs_seq, actions_seq, hidden, starts_seq, stop_gradient_actor=stop_gradient_actor
    )
    (log_prob.mean() + values.pow(2).mean()).backward()
    return results


RECURRENT_CASES = [
    pytest.param("gru", None, id="gru"),
    pytest.param("lstm", None, id="lstm"),
    pytest.param("gtrxl", None, id="gtrxl"),
    pytest.param("gru", EncoderConfig(normalize_obs=True), id="gru-normalized"),
]


@pytest.mark.parametrize("kind,encoder_config", RECURRENT_CASES)
def test_builtin_recurrent_graph_is_bit_identical_to_recurrent_ppo_policy(kind, encoder_config):
    reference = _seeded(lambda: _reference_recurrent(kind, encoder_config))
    graph = _seeded(lambda: _graph_recurrent(kind, encoder_config))
    _assert_tensor_dicts_equal(_renamed(graph.state_dict(), kind), reference.state_dict())

    obs = _obs(STATE_SPACE, (BATCH,), seed=4)
    for policy in (reference, graph):
        policy.update_normalizer(obs)
    _assert_same(
        graph.initial_hidden_state(BATCH, torch.device("cpu")),
        reference.initial_hidden_state(BATCH, torch.device("cpu")),
    )
    hidden = reference.initial_hidden_state(BATCH, torch.device("cpu"))
    with torch.no_grad():
        for _ in range(2):
            *_, hidden = reference.act_recurrent(obs, hidden, torch.zeros(BATCH), True)

    expected = _recurrent_calls(reference, hidden, stop_gradient_actor=True)
    actual = _recurrent_calls(graph, _as_tuple(hidden), stop_gradient_actor=False)

    _assert_same(actual, expected)
    _assert_tensor_dicts_equal(
        _renamed(_parameter_grads(graph), kind), _parameter_grads(reference)
    )


# --- P2: one learn() window with and without the built-in graph ------------------------


LEARN_CASES = [
    pytest.param(PPO, {}, "builtin:mlp", None, id="ppo"),
    pytest.param(
        PPO, {"encoder_config": EncoderConfig(normalize_obs=True)}, "builtin:mlp", None,
        id="ppo-normalized",
    ),
    pytest.param(
        RecurrentPPO, {"rnn_type": "gru", "rnn_hidden_size": 8}, "builtin:recurrent", "gru",
        id="gru-recurrent",
    ),
    pytest.param(
        RecurrentPPO, {"rnn_type": "lstm", "rnn_hidden_size": 8}, "builtin:lstm", "lstm",
        id="lstm",
    ),
    pytest.param(TransformerPPO, TRANSFORMER_KWARGS, "builtin:gtrxl", "gtrxl", id="gtrxl"),
]


@pytest.mark.parametrize("agent_cls,extra_kwargs,policy_graph,stage", LEARN_CASES)
def test_learn_window_with_builtin_graph_is_bit_identical(
    agent_cls, extra_kwargs, policy_graph, stage
):
    def trained(**graph_kwargs) -> dict:
        agent = _make_agent(agent_cls, **extra_kwargs, **graph_kwargs)
        initial = {name: t.clone() for name, t in agent.policy.state_dict().items()}
        _learn_one_window(agent)
        final = agent.policy.state_dict()
        assert any(not torch.equal(final[name], t) for name, t in initial.items())
        return final

    reference = trained()
    with_graph = trained(policy_graph=policy_graph)

    _assert_tensor_dicts_equal(_renamed(with_graph, stage), reference)


# --- return_outputs: the training path's evaluate calls -------------------------------------


def _evaluation_calls(policy, *, return_outputs: bool) -> list:
    """Every evaluate method of ``policy``'s family, then the parameter grads
    of a loss over all of their outputs."""
    flag = {"return_outputs": True} if return_outputs else {}
    if policy.is_stateful:
        obs, actions = _obs(STATE_SPACE, (WINDOW, BATCH), seed=6), _actions((WINDOW, BATCH))
        sequence = (
            obs, actions, policy.initial_hidden_state(BATCH, torch.device("cpu")),
            torch.zeros(WINDOW, BATCH),
        )
        results = [
            policy.evaluate_actions_sequence(*sequence, sum_dims=sum_dims, **flag)
            for sum_dims in (True, False)
        ] + [
            policy.evaluate_actions_sequence_with_dist_params(*sequence, sum_dims=sum_dims, **flag)
            for sum_dims in (True, False)
        ]
    else:
        obs, actions = _obs(IMAGE_SPACE, (BATCH,)), _actions((BATCH,))
        results = [
            policy.evaluate_actions(obs, actions, sum_dims=sum_dims, **flag)
            for sum_dims in (True, False)
        ] + [
            policy.evaluate_actions_with_dist_params(obs, actions, sum_dims=sum_dims, **flag)
            for sum_dims in (True, False)
        ]
    policy.zero_grad(set_to_none=True)
    tensors = [part for result in results for part in result if isinstance(part, torch.Tensor)]
    sum(tensor.sum() for tensor in tensors).backward()
    return results, _parameter_grads(policy)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _graph_mlp(IMAGE_SPACE, IMAGE_ENCODER, "shared"), id="flat-image"),
        pytest.param(lambda: _graph_recurrent("gru", None), id="gru"),
    ],
)
def test_return_outputs_appends_the_graph_outputs_and_changes_nothing_else(build):
    policy = _seeded(build)
    plain, plain_grads = _evaluation_calls(policy, return_outputs=False)
    flagged, flagged_grads = _evaluation_calls(policy, return_outputs=True)

    spec = policy.graph_spec
    for plain_result, flagged_result in zip(plain, flagged):
        *evaluated, outputs = flagged_result
        _assert_same(tuple(evaluated), plain_result)
        assert isinstance(outputs, dict)
        assert {*spec.actor_sinks, spec.value_sink} <= outputs.keys()
        rows = WINDOW * BATCH if policy.is_stateful else BATCH
        assert outputs[spec.value_sink].shape[0] == rows
    _assert_tensor_dicts_equal(flagged_grads, plain_grads)


@pytest.mark.parametrize(
    "agent_cls,extra_kwargs,policy_graph",
    [
        pytest.param(PPO, {}, "builtin:mlp", id="ppo"),
        pytest.param(RecurrentPPO, {"rnn_hidden_size": 8}, "builtin:recurrent", id="gru"),
    ],
)
def test_minibatch_eval_carries_outputs_only_for_a_graph_policy(
    agent_cls, extra_kwargs, policy_graph
):
    def skip_update() -> dict:
        return {}

    def first_evaluation(**graph_kwargs):
        agent = _make_agent(agent_cls, **extra_kwargs, **graph_kwargs)
        agent.train = skip_update
        _learn_one_window(agent)
        return agent._evaluate_minibatch(next(agent._iter_minibatches()))

    assert first_evaluation().outputs is None
    outputs = first_evaluation(policy_graph=policy_graph).outputs
    assert {"actor", "value"} <= outputs.keys()
    assert outputs["actor"].shape[0] == NUM_ENVS * NUM_STEPS // 2


# --- P1: default path ----------------------------------------------------------------------


def test_default_agent_builds_no_graph():
    agent = _make_agent(PPO)
    assert type(agent.policy) is PPOPolicy
    assert agent.policy_graph is None
    assert agent._checkpoint_metadata()["policy_graph"] is None


# --- P3: gradient routing --------------------------------------------------------------------


def _routing_graph(*, actor_detach: bool, stage_input_detach: bool = False, stateful=True) -> dict:
    trunk_consumer = "gru" if stateful else "head_input"
    nodes = {
        "obs": {"type": "input"},
        "encoder": {"type": "encoder"},
        "trunk": {"type": "mlp", "net_arch": [8]},
        trunk_consumer: (
            {"type": "gru", "hidden_size": 8}
            if stateful
            else {"type": "mlp", "net_arch": [8]}
        ),
        "actor": {"type": "actor"},
        "value": {"type": "value"},
    }
    edges = [
        {"from": "obs", "to": "encoder"},
        {"from": "encoder", "to": "trunk"},
        {"from": "trunk", "to": trunk_consumer, "detach": stage_input_detach},
        {"from": trunk_consumer, "to": "actor", "detach": actor_detach},
        {"from": trunk_consumer, "to": "value"},
    ]
    return {"nodes": nodes, "edges": edges}


def _touched(policy, prefix: str) -> bool:
    return any(
        param.grad is not None and bool(param.grad.abs().sum() > 0)
        for name, param in policy.named_parameters()
        if name.startswith(prefix)
    )


def _routing(graph: dict, loss: str, *, stateful: bool = True) -> dict[str, bool]:
    policy = _seeded(lambda: GraphPPOPolicy(STATE_SPACE, ACTION_SPACE, graph=graph, **HEADS))
    if stateful:
        state = torch.randn(WINDOW, BATCH, 4).requires_grad_()
        values, log_prob, _ = policy.evaluate_actions_sequence(
            {"state": state},
            _actions((WINDOW, BATCH)),
            policy.initial_hidden_state(BATCH, torch.device("cpu")),
            torch.zeros(WINDOW, BATCH),
        )
    else:
        state = torch.randn(BATCH, 4).requires_grad_()
        values, log_prob, _ = policy.evaluate_actions({"state": state}, _actions((BATCH,)))
    (log_prob.sum() if loss == "actor" else values.sum()).backward()
    stage = "graph_nodes.gru." if stateful else "graph_nodes.head_input."
    return {
        "stage": _touched(policy, stage),
        "trunk": _touched(policy, "graph_nodes.trunk."),
        "obs": state.grad is not None and bool(state.grad.abs().sum() > 0),
        "actor": _touched(policy, "actor."),
    }


@pytest.mark.parametrize("stateful", [True, False], ids=["sequence", "flat"])
def test_detached_actor_edge_keeps_the_actor_loss_out_of_the_trunk(stateful):
    detached = _routing_graph(actor_detach=True, stateful=stateful)
    assert _routing(detached, "actor", stateful=stateful) == {
        "stage": False, "trunk": False, "obs": False, "actor": True,
    }
    live = _routing_graph(actor_detach=False, stateful=stateful)
    assert _routing(live, "actor", stateful=stateful) == {
        "stage": True, "trunk": True, "obs": True, "actor": True,
    }
    # The value consumer of the same producer keeps its gradient.
    assert _routing(detached, "value", stateful=stateful) == {
        "stage": True, "trunk": True, "obs": True, "actor": False,
    }


@pytest.mark.parametrize("loss", ["actor", "value"])
def test_detach_into_the_stage_cuts_everything_upstream_for_both_heads(loss):
    graph = _routing_graph(actor_detach=False, stage_input_detach=True)
    routed = _routing(graph, loss)
    assert routed["stage"] is True
    assert routed["trunk"] is False
    assert routed["obs"] is False


def test_stop_gradient_actor_argument_is_ignored():
    graph = _routing_graph(actor_detach=False)
    policy = _seeded(lambda: GraphPPOPolicy(STATE_SPACE, ACTION_SPACE, graph=graph, **HEADS))
    assert policy.actor_features_detached is False
    hidden = policy.initial_hidden_state(BATCH, torch.device("cpu"))
    obs, actions = _obs(STATE_SPACE, (WINDOW, BATCH)), _actions((WINDOW, BATCH))
    starts = torch.zeros(WINDOW, BATCH)
    _, log_prob, _ = policy.evaluate_actions_sequence(
        obs, actions, hidden, starts, stop_gradient_actor=True
    )
    log_prob.sum().backward()
    assert _touched(policy, "graph_nodes.gru.")


# --- P4: normalizers ---------------------------------------------------------------------------


def test_update_normalizer_reaches_every_encoder_node():
    space = spaces.Dict(
        {
            "state": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            "state_extra": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        }
    )
    normalized = {"normalize_obs": True}
    graph = {
        "nodes": {
            "obs": {"type": "input"},
            "actor_encoder": {"type": "encoder", "keys": ["state"], "config": normalized},
            "critic_encoder": {
                "type": "encoder", "keys": ["state", "state_extra"], "config": normalized,
            },
            "actor": {"type": "actor"},
            "value": {"type": "value"},
        },
        "edges": [
            {"from": "obs", "to": "actor_encoder"},
            {"from": "obs", "to": "critic_encoder"},
            {"from": "actor_encoder", "to": "actor"},
            {"from": "critic_encoder", "to": "value"},
        ],
    }
    policy = GraphPPOPolicy(space, ACTION_SPACE, graph=graph, **HEADS)
    obs = {"state": torch.randn(6, 4) + 2.0, "state_extra": torch.randn(6, 3) - 1.0}

    policy.update_normalizer(obs)

    for name in ("actor_encoder", "critic_encoder"):
        normalizer = policy.graph_nodes[name].normalizer
        assert int(normalizer.count) == 6, name
        assert bool(normalizer._mean.abs().sum() > 0), name
    assert policy.critic_features_dim == 7
    assert policy.actor_features_dim == 4


# --- P5: two stateful nodes ----------------------------------------------------------------------


TWO_STAGE_GRAPH = textwrap.dedent(
    """
    nodes:
      obs:     {type: input}
      encoder: {type: encoder}
      gru:     {type: gru, hidden_size: 8}
      gtrxl:   {type: gtrxl, embed_dim: 8, head_dim: 4, num_heads: 2, num_layers: 1, mlp_num: 1, memory_len: 2}
      actor:   {type: actor}
      value:   {type: value}
    edges:
      - {from: obs, to: encoder}
      - {from: encoder, to: gru}
      - {from: gru, to: gtrxl}
      - {from: gtrxl, to: actor}
      - {from: gtrxl, to: value}
    """
)


def test_two_stateful_nodes_share_one_flat_state_through_the_rollout_buffer():
    agent = _make_agent(RecurrentPPO, policy_graph=TWO_STAGE_GRAPH)
    policy = agent.policy
    assert policy.graph_spec.stateful_nodes == ("gru", "gtrxl")

    initial = policy.initial_hidden_state(NUM_ENVS, torch.device("cpu"))
    assert isinstance(initial, tuple) and len(initial) == 3  # gru h, gtrxl memory + validity
    assert all(part.shape[1] == NUM_ENVS for part in initial)

    keep = torch.tensor([1.0, 0.0, 1.0, 0.0])
    gru = policy.graph_nodes["gru"]
    gtrxl = policy.graph_nodes["gtrxl"]
    state = tuple(torch.randn(part.shape) for part in initial)
    masked = policy.mask_hidden_state(state, keep)
    assert torch.equal(masked[0], gru.mask_state(state[0], keep))
    _assert_same(masked[1:], gtrxl.mask_state(state[1:], keep))

    _learn_one_window(agent)
    assert all(torch.isfinite(p).all() for p in policy.parameters())

    window_hidden = agent._rollout_initial_hidden
    data = next(agent.rollout_buffer.get_sequences(2, window_hidden, shuffle=False))
    assert isinstance(data.initial_hidden, tuple) and len(data.initial_hidden) == 3
    assert all(part.shape[1] == NUM_ENVS // 2 for part in data.initial_hidden)
    indices = torch.arange(NUM_ENVS // 2)
    _assert_same(data.initial_hidden, policy.index_hidden_state(window_hidden, indices))
    _assert_same(data.initial_hidden, _index_recurrent_state(window_hidden, indices))
    values, log_prob, _ = policy.evaluate_actions_sequence(
        data.obs, data.actions, data.initial_hidden, data.episode_starts
    )
    assert values.shape == log_prob.shape == (NUM_STEPS, NUM_ENVS // 2, 1)


# --- P6: different actor / value widths via concat ---------------------------------------------


CONCAT_VALUE_GRAPH = textwrap.dedent(
    """
    nodes:
      obs:     {type: input}
      encoder: {type: encoder}
      gru:     {type: gru, hidden_size: 8}
      fused:   {type: concat}
      actor:   {type: actor}
      value:   {type: value}
    edges:
      - {from: obs, to: encoder}
      - {from: encoder, to: gru}
      - {from: gru, to: actor}
      - {from: encoder, to: fused}
      - {from: gru, to: fused}
      - {from: fused, to: value}
    """
)


def test_value_sink_fed_by_a_concat_gets_its_own_width():
    agent = _make_agent(RecurrentPPO, policy_graph=CONCAT_VALUE_GRAPH)
    policy = agent.policy
    first_value_layer = next(m for m in policy.value_net.modules() if isinstance(m, nn.Linear))
    first_actor_layer = next(m for m in policy.actor.modules() if isinstance(m, nn.Linear))

    assert policy.critic_features_dim == 4 + 8
    assert first_value_layer.in_features == 4 + 8
    assert first_actor_layer.in_features == 8
    assert policy.critic_extractor is None
    assert not any(key.startswith("critic_extractor") for key in policy.state_dict())

    _learn_one_window(agent)
    assert all(torch.isfinite(p).all() for p in policy.parameters())


# --- P7: separate encoders over key subsets -------------------------------------------------------


class _ImageVecEnv(_FixedHoldVecEnv):
    """``_FixedHoldVecEnv`` plus an ``rgb_cam`` image derived from the state."""

    single_observation_space = IMAGE_SPACE

    def _with_image(self, obs: dict) -> dict:
        state = obs["state"]
        level = (state[:, :1, None, None].abs() * 100).clamp(0, 255).to(torch.uint8)
        return {"state": state, "rgb_cam": level.expand(-1, 16, 16, 3).contiguous()}

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._with_image(obs), info

    def step(self, action):
        obs, rewards, terminated, truncated, infos = super().step(action)
        infos["final_observation"] = self._with_image(infos["final_observation"])
        return self._with_image(obs), rewards, terminated, truncated, infos


SPLIT_ENCODER_GRAPH = textwrap.dedent(
    """
    nodes:
      obs:            {type: input}
      actor_encoder:  {type: encoder, keys: [state]}
      critic_encoder: {type: encoder, keys: [state, rgb_cam], config: {features_dim: 16, plain_conv_pooling: gap}}
      actor:          {type: actor}
      value:          {type: value}
    edges:
      - {from: obs, to: actor_encoder}
      - {from: obs, to: critic_encoder}
      - {from: actor_encoder, to: actor}
      - {from: critic_encoder, to: value}
    """
)


def test_separate_encoders_over_key_subsets():
    policy = _seeded(
        lambda: GraphPPOPolicy(IMAGE_SPACE, ACTION_SPACE, graph=SPLIT_ENCODER_GRAPH, **HEADS)
    )
    actor_encoder = policy.graph_nodes["actor_encoder"]
    assert policy.actor_extractor is actor_encoder
    assert isinstance(actor_encoder, FlattenExtractor)
    assert actor_encoder._state_keys == ("state",)

    obs = _obs(IMAGE_SPACE, (BATCH,))
    other_image = dict(obs, rgb_cam=255 - obs["rgb_cam"])
    assert torch.equal(
        policy.extract_actor_features(obs), policy.extract_actor_features(other_image)
    )
    assert not torch.equal(
        policy.extract_critic_features(obs), policy.extract_critic_features(other_image)
    )

    agent = _make_agent(PPO, env=_ImageVecEnv(emit_decisions=False), policy_graph=SPLIT_ENCODER_GRAPH)
    _learn_one_window(agent)
    assert all(torch.isfinite(p).all() for p in agent.policy.parameters())


def test_builtin_mlp_on_images_warns_that_the_objective_differs():
    with pytest.warns(UserWarning, match="'builtin:mlp' on image observations is not PPOPolicy"):
        _make_agent(
            PPO,
            env=_ImageVecEnv(emit_decisions=False),
            policy_graph="builtin:mlp",
            encoder_config=IMAGE_ENCODER,
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _make_agent(PPO, policy_graph="builtin:mlp")
    assert not any("builtin:mlp" in str(warning.message) for warning in caught)


# --- P12: decision-level PPO with a graph -----------------------------------------------------------


def test_decision_level_recurrent_ppo_runs_with_a_builtin_graph():
    agent = _make_agent(
        RecurrentPPO,
        env=_ComponentMaskVecEnv(),
        decision_level=True,
        policy_graph="builtin:recurrent",
        rnn_type="gru",
        rnn_hidden_size=8,
    )
    before = {name: p.detach().clone() for name, p in agent.policy.named_parameters()}

    _learn_one_window(agent)

    assert agent.rollout_buffer.window_has_components
    after = dict(agent.policy.named_parameters())
    assert all(torch.isfinite(p).all() for p in after.values())
    assert any(not torch.equal(after[name], before[name]) for name in before)


# --- P13: a registered node type inside a policy ---------------------------------------------------


class _Double(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.features_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x


@pytest.fixture
def toy_double():
    register_graph_node("toy_double", lambda spec, input_dim, obs_space, act_space: _Double(input_dim))
    yield
    GRAPH_NODE_REGISTRY.pop("toy_double", None)


def test_registered_node_type_runs_inside_a_policy(toy_double):
    graph = {
        "nodes": {
            "obs": {"type": "input"},
            "encoder": {"type": "encoder"},
            "twice": {"type": "toy_double"},
            "actor": {"type": "actor"},
            "value": {"type": "value"},
        },
        "edges": [
            {"from": "obs", "to": "encoder"},
            {"from": "encoder", "to": "twice"},
            {"from": "twice", "to": "actor"},
            {"from": "encoder", "to": "value"},
        ],
    }
    policy = GraphPPOPolicy(STATE_SPACE, ACTION_SPACE, graph=graph, **HEADS)
    obs = _obs(STATE_SPACE, (BATCH,))
    assert torch.equal(policy.extract_actor_features(obs), 2.0 * obs["state"])
    assert torch.equal(policy.extract_critic_features(obs), obs["state"])


# --- method families and algorithm checks ------------------------------------------------------------


def test_flat_and_recurrent_methods_require_the_matching_graph():
    stateful = GraphPPOPolicy(
        STATE_SPACE, ACTION_SPACE, graph=recurrent_actor_critic("gru", hidden_size=8), **HEADS
    )
    stateless = GraphPPOPolicy(STATE_SPACE, ACTION_SPACE, graph=mlp_actor_critic(), **HEADS)
    obs = _obs(STATE_SPACE, (BATCH,))
    assert stateful.is_stateful and not stateless.is_stateful

    with pytest.raises(TypeError, match=r"stateful node\(s\) \['gru'\]"):
        stateful(obs)
    with pytest.raises(TypeError, match="needs a stateful graph"):
        stateless.initial_hidden_state(BATCH, torch.device("cpu"))
    with pytest.raises(TypeError, match="needs a stateful graph"):
        stateless.act_recurrent(obs, (), torch.zeros(BATCH))


def test_structured_encoder_features_cannot_feed_a_stateful_node(monkeypatch):
    """``SequencePPO``'s flat-latent guard applies on the graph path too."""
    token_layout = {"layout": "token_and_prop", "num_patches": 1, "patch_dim": 2, "prop_dim": 2}
    monkeypatch.setattr(FlattenExtractor, "structured_feature_config", lambda self: token_layout)

    with pytest.raises(NotImplementedError, match="encoder node 'encoder' feeds stateful node 'gru'"):
        GraphPPOPolicy(
            STATE_SPACE, ACTION_SPACE, graph=recurrent_actor_critic("gru", hidden_size=8), **HEADS
        )
    GraphPPOPolicy(STATE_SPACE, ACTION_SPACE, graph=mlp_actor_critic(), **HEADS)


@pytest.mark.parametrize(
    "agent_cls,kwargs,match",
    [
        (PPO, {"policy_graph": "builtin:gru"}, "PPO needs a stateless policy graph"),
        (PPO, {"policy_graph": "builtin:recurrent"}, "but PPO has none"),
        (RecurrentPPO, {"policy_graph": "builtin:mlp"}, "RecurrentPPO needs a stateful"),
        (
            RecurrentPPO,
            {"policy_graph": "builtin:gru", "rnn_type": "lstm"},
            r"'builtin:gru' builds a gru stage, but RecurrentPPO is configured for a lstm "
            r"stage \(rnn_type='lstm'\)",
        ),
        (
            TransformerPPO,
            {"policy_graph": "builtin:lstm", **TRANSFORMER_KWARGS},
            r"'builtin:lstm' builds a lstm stage, but TransformerPPO is configured for a "
            r"gtrxl stage",
        ),
        (
            PPO,
            {"policy_graph": "builtin:mlp", "encoder_sharing": "shared"},
            r"\['encoder_sharing'\] must not be set",
        ),
        (
            PPO,
            {"policy_graph": mlp_actor_critic().to_yaml(), "encoder_config": EncoderConfig()},
            "encoder_config is only read by a built-in",
        ),
    ],
)
def test_algorithm_rejects_mismatched_graph_options(agent_cls, kwargs, match):
    with pytest.raises(ConfigError, match=match):
        _make_agent(agent_cls, **kwargs)


def test_metadata_records_canonical_text_for_builtin_and_inline_graphs():
    builtin = _make_agent(
        RecurrentPPO, policy_graph="builtin:recurrent", rnn_type="gru", rnn_hidden_size=8
    )
    expected = recurrent_actor_critic("gru", hidden_size=8).to_yaml()
    metadata = builtin._checkpoint_metadata()
    assert metadata["policy_graph"] == expected
    assert metadata["encoder_sharing"] is None
    assert metadata["encoder_sharing_origin"] == "policy_graph"

    inline = _make_agent(RecurrentPPO, policy_graph=TWO_STAGE_GRAPH)
    assert inline._checkpoint_metadata()["policy_graph"] == GraphSpec.parse(TWO_STAGE_GRAPH).to_yaml()


def test_loading_across_graph_and_non_graph_checkpoints_is_rejected(tmp_path):
    graph_agent = _make_agent(PPO, policy_graph="builtin:mlp")
    plain_agent = _make_agent(PPO)
    graph_path = graph_agent.save(tmp_path / "graph.pt")
    plain_path = plain_agent.save(tmp_path / "plain.pt")

    with pytest.raises(ValueError, match="built from a policy graph but the current agent has none"):
        plain_agent.load(graph_path, load_replay_buffer=False)
    with pytest.raises(ValueError, match="built from a policy graph but the checkpoint's is not"):
        graph_agent.load(plain_path, load_replay_buffer=False)

    reloaded = _make_agent(PPO, policy_graph="builtin:mlp", seed=5)
    reloaded.load(graph_path, load_replay_buffer=False)
    _assert_tensor_dicts_equal(reloaded.policy.state_dict(), graph_agent.policy.state_dict())


def test_loading_a_checkpoint_of_a_different_graph_is_rejected(tmp_path):
    """Same parameter shapes, different graph (the actor edge's ``detach``
    flipped): the canonical texts must match exactly."""
    detached = _make_agent(PPO, policy_graph="builtin:mlp")
    live = _make_agent(PPO, policy_graph=mlp_actor_critic(encoder_sharing="shared").to_yaml())
    assert live.policy.state_dict().keys() == detached.policy.state_dict().keys()
    path = detached.save(tmp_path / "detached.pt")

    with pytest.raises(
        ValueError,
        match=r"policy graph differs from the current agent's; first difference at line 2: "
        r"checkpoint '- detach: true', agent '- detach: false'",
    ):
        live.load(path, load_replay_buffer=False)


# --- P9: CLI preflight ----------------------------------------------------------------------------


def _preflight(argv: list[str]):
    from rl_garden.training.online import registry

    command = registry.parse_command(argv)
    registry._validate_config(command)
    return command


def test_preflight_accepts_a_stateful_builtin_only_for_a_sequence_algorithm():
    command = _preflight(["recurrent_ppo", "--policy-graph", "builtin:recurrent"])
    assert command.args.policy_graph == "builtin:recurrent"
    _preflight(["recurrent_ppo", "--policy-graph", "builtin:gru", "--rnn-type", "gru"])
    _preflight(["transformer_ppo", "--policy-graph", "builtin:recurrent"])
    _preflight(["transformer_ppo", "--policy-graph", "builtin:gtrxl"])
    _preflight(["ppo", "--policy-graph", "builtin:mlp"])
    for stateful in ("builtin:gru", "builtin:recurrent"):
        with pytest.raises(ConfigError, match="ppo needs a stateless policy graph"):
            _preflight(["ppo", "--policy-graph", stateful])
    with pytest.raises(ConfigError, match="recurrent_ppo needs a stateful policy graph"):
        _preflight(["recurrent_ppo", "--policy-graph", "builtin:mlp"])


def test_preflight_rejects_a_builtin_stage_that_disagrees_with_the_algorithm():
    with pytest.raises(
        ConfigError,
        match=r"'builtin:gru' builds a gru stage, but recurrent_ppo is configured for a "
        r"lstm stage \(rnn_type='lstm'\)",
    ):
        _preflight(["recurrent_ppo", "--policy-graph", "builtin:gru"])
    with pytest.raises(ConfigError, match="transformer_ppo is configured for a gtrxl stage"):
        _preflight(["transformer_ppo", "--policy-graph", "builtin:lstm"])


@pytest.mark.parametrize(
    "extra,flag",
    [
        (["--encoder-sharing", "shared"], "--encoder-sharing"),
        (["--detach-encoder-on-actor", "True"], "--detach-encoder-on-actor"),
        (["--obs-groups.actor", "state"], "--obs-groups"),
        (["--critic-encoder.features-dim", "17"], "--critic-encoder"),
    ],
)
def test_preflight_rejects_encoder_options_alongside_a_graph(extra, flag):
    with pytest.raises(ConfigError, match=flag):
        _preflight(["ppo", "--policy-graph", "builtin:mlp", *extra])


def test_preflight_reads_a_path_and_canonicalizes_it(tmp_path):
    text = mlp_actor_critic(encoder_sharing="shared").to_yaml()
    path = tmp_path / "graph.yaml"
    path.write_text(text.replace("detach: false", "detach: false  # trained by both"), "utf-8")

    command = _preflight(["ppo", "--policy-graph", str(path)])

    assert command.args.policy_graph == text
    assert command.derived["policy_graph"]["reason"] == "canonical policy graph"
    assert command.sources["policy_graph"].kind == "runtime-derived"


def test_preflight_accepts_inline_yaml():
    inline = textwrap.dedent(
        """
        nodes:
          obs: {type: input}
          encoder: {type: encoder}
          actor: {type: actor}
          value: {type: value}
        edges:
          - {from: obs, to: encoder}
          - {from: encoder, to: actor}
          - {from: encoder, to: value}
        """
    )
    command = _preflight(["ppo", "--policy-graph", inline])
    assert command.args.policy_graph == GraphSpec.parse(inline).to_yaml()


def test_preflight_rejects_cli_encoder_settings_and_invalid_graphs():
    with pytest.raises(ConfigError, match=r"--encoder\.\* is only read by a built-in"):
        _preflight(
            ["ppo", "--policy-graph", mlp_actor_critic().to_yaml(), "--encoder.features-dim", "64"]
        )
    dead = mlp_actor_critic().to_dict()
    dead["nodes"]["watcher"] = {"type": "mlp", "net_arch": [4]}
    dead["edges"].append({"from": "encoder", "to": "watcher"})

    with pytest.raises(ConfigError, match="node 'watcher' has no outgoing edge"):
        _preflight(["ppo", "--policy-graph", yaml.safe_dump(dead)])
