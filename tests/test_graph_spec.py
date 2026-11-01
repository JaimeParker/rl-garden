"""``GraphSpec`` parsing, validation and canonical form, the node-type
registry, and graph sources (path / inline / built-in)."""

from __future__ import annotations

import re
import textwrap

import pytest
import torch
import torch.nn as nn
import yaml

from rl_garden.common.effective_config import ConfigError
from rl_garden.networks.graph import registry as graph_registry
from rl_garden.networks.graph.builtin import (
    BUILTIN_GRAPHS,
    builtin_graph,
    builtin_graph_name,
    mlp_actor_critic,
    recurrent_actor_critic,
)
from rl_garden.networks.graph.registry import (
    GRAPH_NODE_REGISTRY,
    find_graph_node_type,
    register_graph_node,
)
from rl_garden.networks.graph.spec import GraphSpec, read_graph_source

RECURRENT_GRAPH = textwrap.dedent(
    """
    version: 1
    nodes:
      obs:     {type: input}
      encoder: {type: encoder, config: {normalize_obs: true}}
      gru:     {type: gru, hidden_size: 8}
      actor:   {type: actor}
      value:   {type: value}
    edges:
      - {from: obs, to: encoder}
      - {from: encoder, to: gru}
      - {from: gru, to: actor, detach: true}
      - {from: gru, to: value}
    """
)

# Same graph: other node/edge order, defaults omitted or spelled out.
REORDERED_GRAPH = textwrap.dedent(
    """
    edges:
      - {from: gru, to: value, detach: false}
      - {from: encoder, to: gru}
      - {from: obs, to: encoder}
      - {from: gru, to: actor, detach: true}
    losses: []
    nodes:
      value:   {type: value}
      actor:   {type: actor}
      gru:     {type: gru, hidden_size: 8, num_layers: 1}
      encoder: {type: encoder, keys: null, config: {normalize_obs: true}}
      obs:     {type: input}
    """
)


def _mlp_graph(**overrides) -> dict:
    graph = {
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
    graph.update(overrides)
    return graph


# --- parse / canonical form ----------------------------------------------------


def test_parse_canonical_dump_round_trip():
    spec = GraphSpec.parse(RECURRENT_GRAPH)
    text = spec.to_yaml()

    again = GraphSpec.parse(text)

    assert again == spec
    assert again.to_yaml() == text


def test_canonical_text_ignores_order_and_explicit_defaults():
    assert GraphSpec.parse(REORDERED_GRAPH).to_yaml() == GraphSpec.parse(RECURRENT_GRAPH).to_yaml()


def test_canonical_form_fills_defaults_and_sorts():
    spec = GraphSpec.parse(RECURRENT_GRAPH)
    canonical = spec.to_dict()

    assert list(canonical["nodes"]) == sorted(canonical["nodes"])
    assert canonical["nodes"]["gru"] == {"type": "gru", "hidden_size": 8, "num_layers": 1}
    assert canonical["nodes"]["encoder"]["config"]["normalize_obs"] is True
    assert canonical["nodes"]["encoder"]["config"]["backbone"] == "plain_conv"
    assert canonical["nodes"]["encoder"]["keys"] is None
    assert [(e["from"], e["to"]) for e in canonical["edges"]] == sorted(
        (e["from"], e["to"]) for e in canonical["edges"]
    )
    assert all("detach" in edge for edge in canonical["edges"])
    assert canonical["losses"] == []


def test_mapping_and_text_sources_parse_the_same():
    from_dict = GraphSpec.parse(_mlp_graph())
    from_text = GraphSpec.parse(from_dict.to_yaml())
    assert from_dict == from_text
    assert GraphSpec.parse(from_dict) is from_dict


def test_topological_order_breaks_ties_by_name():
    graph = _mlp_graph(
        nodes={
            "obs": {"type": "input"},
            "zeta": {"type": "encoder"},
            "alpha": {"type": "encoder"},
            "fused": {"type": "concat"},
            "actor": {"type": "actor"},
            "value": {"type": "value"},
        },
        edges=[
            {"from": "obs", "to": "zeta"},
            {"from": "obs", "to": "alpha"},
            {"from": "zeta", "to": "fused"},
            {"from": "alpha", "to": "fused"},
            {"from": "fused", "to": "actor"},
            {"from": "fused", "to": "value"},
        ],
    )
    order = GraphSpec.parse(graph).topological_order()
    assert order == ("obs", "alpha", "zeta", "fused", "actor", "value")


def test_statefulness():
    assert not GraphSpec.parse(_mlp_graph()).is_stateful
    spec = GraphSpec.parse(RECURRENT_GRAPH)
    assert spec.is_stateful
    assert spec.stateful_nodes == ("gru",)


@pytest.mark.parametrize("name", BUILTIN_GRAPHS)
def test_builtin_graphs_round_trip(name):
    spec = builtin_graph(name)
    assert GraphSpec.parse(spec.to_yaml()) == spec
    assert spec.is_stateful is (name != "mlp")


def test_builtin_graph_detach_follows_encoder_sharing():
    detached = {(e.source, e.target): e.detach for e in mlp_actor_critic().edges}
    assert detached[("encoder", "actor")] is True
    shared = {(e.source, e.target): e.detach for e in mlp_actor_critic(encoder_sharing="shared").edges}
    assert shared[("encoder", "actor")] is False
    separate = mlp_actor_critic(encoder_sharing="separate")
    assert {e.target for e in separate.outgoing("actor_encoder")} == {"actor"}
    assert {e.target for e in separate.outgoing("critic_encoder")} == {"value"}
    gtrxl = recurrent_actor_critic("gtrxl", embed_dim=8, memory_len=2)
    assert gtrxl.node("gtrxl").params["embed_dim"] == 8
    with pytest.raises(ConfigError, match="separate"):
        recurrent_actor_critic("gru", encoder_sharing="separate")


def test_unknown_builtin_name_lists_the_choices():
    assert builtin_graph_name("builtin:recurrent") == "recurrent"
    with pytest.raises(ConfigError, match="builtin:recurrent"):
        builtin_graph_name("builtin:transformer")


# --- validation errors (each names the node) ---------------------------------------


def _with_nodes(**changes) -> dict:
    graph = _mlp_graph()
    graph["nodes"] = {**graph["nodes"], **changes}
    return graph


def _with_edges(*extra) -> dict:
    graph = _mlp_graph()
    graph["edges"] = graph["edges"] + list(extra)
    return graph


@pytest.mark.parametrize(
    "graph,match",
    [
        pytest.param(
            _with_nodes(trunk={"type": "transformer"}),
            r"node 'trunk' has unknown type 'transformer'; known types: \[.*'concat'.*'gru'",
            id="unknown-type",
        ),
        pytest.param(
            _mlp_graph(
                nodes={
                    "obs": {"type": "input"},
                    "encoder": {"type": "encoder"},
                    "left": {"type": "concat"},
                    "right": {"type": "concat"},
                    "actor": {"type": "actor"},
                    "value": {"type": "value"},
                },
                edges=[
                    {"from": "obs", "to": "encoder"},
                    {"from": "encoder", "to": "left"},
                    {"from": "right", "to": "left"},
                    {"from": "left", "to": "right"},
                    {"from": "left", "to": "actor"},
                    {"from": "right", "to": "value"},
                ],
            ),
            r"cycle through node\(s\) \['left', 'right'\]",
            id="cycle",
        ),
        pytest.param(
            _with_edges({"from": "trunk", "to": "critic"}),
            r"'to' names unknown node 'critic' \(dangling edge\)",
            id="dangling-edge",
        ),
        pytest.param(
            _with_edges({"from": "trunk", "to": "pi"}) | {
                "nodes": {**_mlp_graph()["nodes"], "pi": {"type": "actor"}}
            },
            r"2 actor sinks \['actor', 'pi'\]: each needs 'action_dims: \[start, stop\]'.*"
            r"missing on \['actor', 'pi'\]",
            id="two-actor-sinks-without-action-dims",
        ),
        pytest.param(
            _mlp_graph(
                nodes={k: v for k, v in _mlp_graph()["nodes"].items() if k != "value"},
                edges=[e for e in _mlp_graph()["edges"] if e["to"] != "value"],
            ),
            r"exactly one 'value' node, found no node",
            id="no-value-sink",
        ),
        pytest.param(
            _with_nodes(extra={"type": "mlp", "net_arch": [4]})
            | {"edges": _mlp_graph()["edges"] + [{"from": "actor", "to": "extra"}]},
            r"sink node 'actor' has an outgoing edge",
            id="sink-with-outgoing-edge",
        ),
        pytest.param(
            _with_nodes(watcher={"type": "mlp", "net_arch": [4]})
            | {"edges": _mlp_graph()["edges"] + [{"from": "trunk", "to": "watcher"}]},
            r"node 'watcher' has no outgoing edge and no loss \(dead node\)",
            id="dead-node",
        ),
        pytest.param(
            _with_nodes(trunk={"type": "mlp", "net_arch": [8], "width": 3}),
            r"node 'trunk' \(type 'mlp'\) has unknown field\(s\) \['width'\]",
            id="unknown-field",
        ),
        pytest.param(
            _with_nodes(encoder={"type": "encoder", "config": {"backbone_name": "x"}}),
            r"node 'encoder': 'config' has a field EncoderConfig does not know",
            id="unknown-encoder-config-field",
        ),
        pytest.param(
            _with_nodes(trunk={"type": "mlp"}),
            r"node 'trunk' \(type 'mlp'\) is missing required field 'net_arch'",
            id="missing-required-field",
        ),
        pytest.param(
            _with_edges({"from": "obs", "to": "trunk"}),
            r"node 'trunk' \(type 'mlp'\) cannot read the input node 'obs' directly",
            id="mlp-reads-observation",
        ),
        pytest.param(
            _mlp_graph(
                nodes={**_mlp_graph()["nodes"], "late_encoder": {"type": "encoder"}},
                edges=[
                    {"from": "obs", "to": "encoder"},
                    {"from": "encoder", "to": "trunk"},
                    {"from": "trunk", "to": "late_encoder"},
                    {"from": "late_encoder", "to": "actor"},
                    {"from": "trunk", "to": "value"},
                ],
            ),
            r"node 'late_encoder' \(type 'encoder'\) reads the observation",
            id="encoder-reads-latent",
        ),
        pytest.param(
            _mlp_graph(
                edges=[{"from": "obs", "to": "encoder", "detach": True}]
                + _mlp_graph()["edges"][1:]
            ),
            r"edge 'obs' -> 'encoder': an edge out of the input node",
            id="detach-on-input-edge",
        ),
        pytest.param(
            _with_edges({"from": "encoder", "to": "trunk"}),
            r"edge 'encoder' -> 'trunk' is declared twice",
            id="duplicate-edge",
        ),
        pytest.param(
            _with_nodes(**{"bad.name": {"type": "concat"}}),
            r"node name 'bad.name' is not a Python identifier",
            id="bad-node-name",
        ),
        pytest.param(
            _mlp_graph(
                nodes={**_mlp_graph()["nodes"], "values": {"type": "mlp", "net_arch": [4]}},
                edges=_mlp_graph()["edges"] + [{"from": "trunk", "to": "values"}],
            ),
            r"node name 'values' is reserved: it is an attribute of nn.ModuleDict",
            id="reserved-node-name",
        ),
        pytest.param(
            _mlp_graph(losses=[{"node": "trunk", "type": "huber", "target": "batch.rewards"}]),
            r"loss on node 'trunk' has type 'huber'",
            id="unknown-loss-type",
        ),
        pytest.param(
            _mlp_graph(losses=[{"node": "trunk", "type": "mse", "target": "rewards"}]),
            r"loss on node 'trunk' has target 'rewards'",
            id="bad-loss-target",
        ),
        pytest.param(
            _mlp_graph(losses=[{"node": "actor", "type": "mse", "target": "batch.rewards"}]),
            r"loss on node 'actor'",
            id="loss-on-sink",
        ),
        pytest.param(
            _mlp_graph(losses=[{"node": "trunk", "type": "cross_entropy", "target": "batch.dones"}]),
            r"loss on node 'trunk': 'cross_entropy' needs integer class targets",
            id="class-loss-on-batch-target",
        ),
        pytest.param(
            _mlp_graph(
                losses=[{"node": "trunk", "type": "mse", "target": "batch.rewards", "weight": -1}]
            ),
            r"loss on node 'trunk': 'weight' must be a finite number >= 0",
            id="negative-loss-weight",
        ),
        pytest.param(
            _mlp_graph(
                losses=[{"node": "trunk", "type": "mse", "target": "batch.rewards", "mask": "all"}]
            ),
            r"loss on node 'trunk' has mask 'all'",
            id="unknown-loss-mask",
        ),
        pytest.param(
            _mlp_graph(
                losses=[
                    {"node": "trunk", "type": "mse", "target": "batch.rewards"},
                    {"node": "trunk", "type": "bce_with_logits", "target": "batch.dones"},
                ]
            ),
            r"node 'trunk' carries more than one loss",
            id="two-losses-on-one-node",
        ),
        pytest.param(
            _mlp_graph(version=2),
            r"version 2 is not supported",
            id="version",
        ),
    ],
)
def test_validation_errors_name_the_node(graph, match):
    with pytest.raises(ConfigError, match=match):
        GraphSpec.parse(graph)


def test_duplicate_yaml_key_is_rejected():
    text = textwrap.dedent(
        """
        nodes:
          obs: {type: input}
          trunk: {type: mlp, net_arch: [8]}
          trunk: {type: mlp, net_arch: [4]}
        edges: []
        """
    )
    with pytest.raises(ConfigError, match="found duplicate key 'trunk'"):
        GraphSpec.parse(text)


def test_a_loss_makes_a_leaf_node_valid_and_round_trips():
    """A loss makes an otherwise dead node valid; omitted fields get their
    defaults (weight 1.0, mask decision) and the canonical text keeps them."""
    graph = _with_nodes(watcher={"type": "mlp", "net_arch": [4], "output_dim": 1})
    graph["edges"] = graph["edges"] + [{"from": "trunk", "to": "watcher"}]
    graph["losses"] = [{"node": "watcher", "type": "mse", "target": "batch.rewards"}]
    spec = GraphSpec.parse(graph)

    (loss,) = spec.losses
    assert (loss.node, loss.type, loss.target, loss.weight, loss.mask) == (
        "watcher", "mse", "batch.rewards", 1.0, "decision",
    )
    assert spec.loss_nodes == ("watcher",)
    assert GraphSpec.parse(spec.to_yaml()) == spec
    assert yaml.safe_load(spec.to_yaml())["losses"] == [
        {"node": "watcher", "type": "mse", "target": "batch.rewards", "weight": 1.0,
         "mask": "decision"}
    ]
    without_loss = {key: value for key, value in graph.items() if key != "losses"}
    with pytest.raises(ConfigError, match=r"node 'watcher' has no outgoing edge and no loss"):
        GraphSpec.parse(without_loss)


def test_zero_weight_and_mask_none_parse():
    graph = _with_nodes(watcher={"type": "mlp", "net_arch": [4], "output_dim": 1})
    graph["edges"] = graph["edges"] + [{"from": "trunk", "to": "watcher"}]
    graph["losses"] = [
        {"node": "watcher", "type": "bce_with_logits", "target": "info.flag", "weight": 0,
         "mask": "none"}
    ]
    (loss,) = GraphSpec.parse(graph).losses
    assert (loss.weight, loss.mask, loss.target) == (0.0, "none", "info.flag")


def test_concat_inputs_are_ordered_by_producer_name():
    graph = _mlp_graph(
        nodes={
            **_mlp_graph()["nodes"],
            "fused": {"type": "concat"},
        },
        edges=[
            {"from": "obs", "to": "encoder"},
            {"from": "encoder", "to": "trunk"},
            {"from": "trunk", "to": "fused"},
            {"from": "encoder", "to": "fused"},
            {"from": "fused", "to": "actor"},
            {"from": "trunk", "to": "value"},
        ],
    )
    spec = GraphSpec.parse(graph)
    assert [edge.source for edge in spec.incoming("fused")] == ["encoder", "trunk"]


# --- registry ------------------------------------------------------------------------


class _Double(nn.Module):
    def __init__(self, input_dim: int, scale: float) -> None:
        super().__init__()
        self.features_dim = input_dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


def _double_factory(spec, input_dim, observation_space, action_space):
    del observation_space, action_space
    return _Double(input_dim, spec.params["scale"])


@pytest.fixture
def toy_double():
    register_graph_node("toy_double", _double_factory, params={"scale": 2.0})
    yield
    GRAPH_NODE_REGISTRY.pop("toy_double", None)


def _toy_graph() -> dict:
    graph = _mlp_graph()
    graph["nodes"] = {**graph["nodes"], "twice": {"type": "toy_double"}}
    graph["edges"] = [
        {"from": "obs", "to": "encoder"},
        {"from": "encoder", "to": "twice"},
        {"from": "twice", "to": "trunk"},
        {"from": "trunk", "to": "actor"},
        {"from": "trunk", "to": "value"},
    ]
    return graph


def test_registered_toy_node_is_usable_from_a_spec(toy_double):
    spec = GraphSpec.parse(_toy_graph())
    assert spec.node("twice").params == {"scale": 2.0}
    assert spec.node_type("twice").factory is _double_factory


def test_duplicate_registration_is_rejected(toy_double):
    with pytest.raises(ValueError, match="already registered"):
        register_graph_node("toy_double", _double_factory)


def test_entry_point_registers_a_node_type(monkeypatch):
    class _EntryPoint:
        def load(self):
            register_graph_node("toy_entry_point", _double_factory, params={"scale": 3.0})

    def entry_points(*, group):
        assert group == "rlgarden.graph_nodes"
        return [_EntryPoint()]

    monkeypatch.setattr(graph_registry, "_DISCOVERED", False)
    monkeypatch.setattr(graph_registry.importlib.metadata, "entry_points", entry_points)
    try:
        assert find_graph_node_type("toy_entry_point").params == {"scale": 3.0}
    finally:
        GRAPH_NODE_REGISTRY.pop("toy_entry_point", None)


# --- graph sources ---------------------------------------------------------------------


def test_graph_source_reads_a_path_and_keeps_inline_text_and_builtins(tmp_path):
    path = tmp_path / "graph.yaml"
    path.write_text(RECURRENT_GRAPH, encoding="utf-8")

    assert read_graph_source(str(path)) == RECURRENT_GRAPH
    assert read_graph_source(RECURRENT_GRAPH) == RECURRENT_GRAPH
    assert read_graph_source("builtin:gru") == "builtin:gru"
    for missing in (tmp_path / "missing.yaml", tmp_path / "graph_without_suffix"):
        with pytest.raises(ConfigError, match=re.escape(f"policy graph file not found: {missing}")):
            read_graph_source(str(missing))
    with pytest.raises(ConfigError, match="cannot be read"):
        read_graph_source(str(tmp_path))  # a directory


def test_long_single_line_inline_yaml_is_text_not_a_path():
    canonical = GraphSpec.parse(_mlp_graph()).to_dict()
    one_line = yaml.safe_dump(canonical, default_flow_style=True, width=10**6).strip()
    assert "\n" not in one_line and len(one_line) > 300

    assert read_graph_source(one_line) == one_line
    assert GraphSpec.parse(read_graph_source(one_line)) == GraphSpec.parse(_mlp_graph())
