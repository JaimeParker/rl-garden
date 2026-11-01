"""Policy graph spec: a YAML description of a PPO policy's trunk.

A graph names the networks between the observation and the two PPO heads
and wires them with edges. ``GraphPPOPolicy``
(``rl_garden.policies.graph_ppo_policy``) builds the modules and runs the
graph; the Gaussian actor head and the scalar value head stay the PPO
contract and read the inputs of the ``actor`` and ``value`` sinks.

Example (the built-in recurrent graph)::

    version: 1
    nodes:
      obs:     {type: input}
      encoder: {type: encoder, keys: null, config: {backbone: plain_conv, features_dim: 256}}
      gru:     {type: gru, hidden_size: 256, num_layers: 1}
      actor:   {type: actor}
      value:   {type: value}
    edges:
      - {from: obs, to: encoder}
      - {from: encoder, to: gru}
      - {from: gru, to: actor, detach: true}
      - {from: gru, to: value}

Top-level fields: ``version`` (``1``), ``nodes`` (name -> node), ``edges``
(list), ``losses`` (list, optional). Node names are Python identifiers and
must not shadow an ``nn.ModuleDict`` attribute (``keys``, ``values``,
``train``, ``type``, ...).

Node types (in-tree; more can be registered, see ``registry``):

==========  ======================================================  ==========
type        parameters (default)                                    inputs
==========  ======================================================  ==========
input       --                                                      none
encoder     ``keys`` (null: every key), ``config`` (EncoderConfig   the input
            fields, ``{}``: defaults)
mlp         ``net_arch`` (required), ``output_dim`` (null: no       one
            output layer), ``activation`` (relu|gelu|mish, relu)
concat      --                                                      one or more
gru, lstm   ``hidden_size`` (256), ``num_layers`` (1)               one
gtrxl       ``embed_dim`` (256), ``head_dim`` (64), ``num_heads``   one
            (4), ``num_layers`` (3), ``mlp_num`` (2),
            ``memory_len`` (64), ``dropout_rate`` (0.0),
            ``gru_bias`` (2.0)
actor       -- (sink: the actor head reads this input)              one
value       -- (sink: the value head reads this input)              one
==========  ======================================================  ==========

``gru``/``lstm``/``gtrxl`` are stateful: a graph with one or more of them
needs a ``SequencePPO`` algorithm (``recurrent_ppo``/``transformer_ppo``), a
graph without one needs ``ppo``. ``concat`` concatenates its inputs along
the feature axis in the order of the producer names.

Edge fields: ``from``, ``to``, ``detach`` (false). ``detach: true`` cuts the
gradient from that consumer into the producer only; other consumers of the
same producer keep it. Edges out of the ``input`` node cannot detach.

Loss fields (auxiliary losses on extra nodes, trained with the PPO loss):

* ``node``: a non-sink, non-input node; at most one loss per node (add a
  node to put a second loss on the same features). A node with a loss and
  no outgoing edge is valid.
* ``type``: ``mse``, ``bce_with_logits`` (the node outputs logits) or
  ``cross_entropy`` (class logits, integer class targets); more can be
  registered, see ``rl_garden.networks.graph.losses``.
* ``target``: ``info.<key>`` (a tensor the env reports in ``infos[<key>]``
  on every rollout step, stored next to the observation), ``batch.rewards``
  or ``batch.dones`` (the step's reward / after-step done flag). A
  ``batch.*`` target needs a float-target loss on a node of width 1;
  ``cross_entropy`` needs an ``info.<key>`` target. ``info.<key>`` targets
  cannot be checked before the env runs: a missing key or a wrong shape or
  dtype raises ``ValueError`` at the first rollout step that shows it.
* ``weight`` (1.0, ``>= 0``): the loss adds ``weight * mean`` to the PPO
  loss; ``0`` evaluates and logs it without training on it.
* ``mask`` (``decision``|``none``, ``decision``): under decision-level PPO,
  ``decision`` averages over decision steps like the policy loss; ``none``
  averages over every step. Without decision-level PPO both are the plain
  mean.

A loss node runs only in the update (never during rollouts), on every
minibatch: on a recurrent graph over the ``(T, B)`` window, flattened to
``T * B`` rows T-major exactly like its target. Its gradient reaches every
upstream node unless an edge on its path has ``detach: true``. It is logged
as ``losses/graph_loss/<node>`` (the unweighted mean over the window).

Rules: exactly one ``input``, one ``actor`` and one ``value`` node; sinks
have no outgoing edge; every other non-input node has an outgoing edge or a
loss; no cycle; ``encoder`` reads the ``input`` node and only it. Every
violation raises ``ConfigError`` naming the node. ``GraphSpec.to_yaml()``
writes the canonical form (sorted nodes and edges, every default explicit)
that checkpoints record.
"""

from __future__ import annotations

import heapq
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Union

import torch.nn as nn
import yaml

from rl_garden.common.effective_config import ConfigError, UniqueKeyLoader
from rl_garden.networks.graph.losses import find_graph_loss, known_graph_losses
from rl_garden.networks.graph.registry import (
    REQUIRED,
    GraphNodeType,
    find_graph_node_type,
    known_graph_node_types,
)

GRAPH_SPEC_VERSION = 1
BUILTIN_GRAPH_PREFIX = "builtin:"
BATCH_LOSS_TARGETS = ("batch.rewards", "batch.dones")
LOSS_MASKS = ("decision", "none")

_TOP_LEVEL_FIELDS = ("version", "nodes", "edges", "losses")
_EDGE_FIELDS = ("from", "to", "detach")
_LOSS_FIELDS = ("node", "type", "target", "weight", "mask")
_NODE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INFO_TARGET = re.compile(r"^info\.[A-Za-z_][A-Za-z0-9_]*$")
# Node modules live in an nn.ModuleDict keyed by node name, which refuses
# keys that shadow its own attributes.
_RESERVED_NODE_NAMES = frozenset(dir(nn.ModuleDict()))


@dataclass(frozen=True)
class NodeSpec:
    name: str
    type: str
    #: Type-specific parameters, every default filled in.
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    detach: bool = False


@dataclass(frozen=True)
class LossSpec:
    node: str
    type: str
    target: str
    weight: float = 1.0
    mask: str = "decision"


GraphSource = Union["GraphSpec", str, Mapping[str, Any]]


@dataclass(frozen=True)
class GraphSpec:
    """A validated policy graph. Build one with :meth:`parse`."""

    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    losses: tuple[LossSpec, ...] = ()
    version: int = GRAPH_SPEC_VERSION

    # --- construction -----------------------------------------------------

    @classmethod
    def parse(cls, source: GraphSource) -> "GraphSpec":
        """Parse and validate YAML text, a mapping, or an existing spec.
        Needs no observation space; raises ``ConfigError``."""
        if isinstance(source, GraphSpec):
            return source
        raw = _load_mapping(source)
        unknown = sorted(set(raw) - set(_TOP_LEVEL_FIELDS))
        if unknown:
            raise ConfigError(
                f"policy graph has unknown top-level field(s) {unknown}; "
                f"allowed: {list(_TOP_LEVEL_FIELDS)}."
            )
        version = raw.get("version", GRAPH_SPEC_VERSION)
        if version != GRAPH_SPEC_VERSION:
            raise ConfigError(
                f"policy graph version {version!r} is not supported "
                f"(expected {GRAPH_SPEC_VERSION})."
            )
        nodes = _parse_nodes(raw.get("nodes"))
        edges = _parse_edges(raw.get("edges"), {node.name for node in nodes})
        losses = _parse_losses(raw.get("losses", []))
        spec = cls(
            nodes=tuple(sorted(nodes, key=lambda node: node.name)),
            edges=tuple(sorted(edges, key=lambda edge: (edge.source, edge.target))),
            losses=tuple(sorted(losses, key=lambda loss: loss.node)),
        )
        spec._validate_structure()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "nodes": {node.name: {"type": node.type, **node.params} for node in self.nodes},
            "edges": [
                {"from": edge.source, "to": edge.target, "detach": edge.detach}
                for edge in self.edges
            ],
            "losses": [
                {
                    "node": loss.node,
                    "type": loss.type,
                    "target": loss.target,
                    "weight": loss.weight,
                    "mask": loss.mask,
                }
                for loss in self.losses
            ],
        }

    def to_yaml(self) -> str:
        """Canonical text: equal specs give equal text."""
        return yaml.safe_dump(self.to_dict(), sort_keys=True, default_flow_style=False)

    # --- queries ----------------------------------------------------------

    def node(self, name: str) -> NodeSpec:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)

    def node_type(self, name: str) -> GraphNodeType:
        node_type = find_graph_node_type(self.node(name).type)
        assert node_type is not None, name  # checked by parse()
        return node_type

    def incoming(self, name: str) -> tuple[EdgeSpec, ...]:
        """Edges into ``name``, ordered by producer name."""
        return tuple(edge for edge in self.edges if edge.target == name)

    def outgoing(self, name: str) -> tuple[EdgeSpec, ...]:
        return tuple(edge for edge in self.edges if edge.source == name)

    def nodes_with_role(self, role: str) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes if self.node_type(node.name).role == role)

    @property
    def input_node(self) -> str:
        return self.nodes_with_role("input")[0]

    @property
    def actor_sink(self) -> str:
        return self.nodes_with_role("actor")[0]

    @property
    def value_sink(self) -> str:
        return self.nodes_with_role("value")[0]

    @property
    def stateful_nodes(self) -> tuple[str, ...]:
        """Stateful nodes in topological order."""
        return tuple(name for name in self.topological_order() if self.node_type(name).stateful)

    @property
    def is_stateful(self) -> bool:
        return bool(self.stateful_nodes)

    @property
    def loss_nodes(self) -> tuple[str, ...]:
        """Nodes that carry a loss, sorted by name."""
        return tuple(loss.node for loss in self.losses)

    def topological_order(self) -> tuple[str, ...]:
        """Every node, producers first, ties broken by name. This is also
        the module construction order (it decides RNG consumption)."""
        order, remaining = self._kahn_result
        if remaining:
            raise ConfigError(f"policy graph has a cycle through node(s) {remaining}.")
        return order

    def ancestors(self, name: str) -> frozenset[str]:
        return self._ancestor_sets[name]

    @cached_property
    def _kahn_result(self) -> tuple[tuple[str, ...], list[str]]:
        return _kahn_order(self)

    @cached_property
    def _ancestor_sets(self) -> dict[str, frozenset[str]]:
        found: dict[str, frozenset[str]] = {}
        for node in self.nodes:
            seen: set[str] = set()
            pending = [edge.source for edge in self.incoming(node.name)]
            while pending:
                current = pending.pop()
                if current not in seen:
                    seen.add(current)
                    pending.extend(edge.source for edge in self.incoming(current))
            found[node.name] = frozenset(seen)
        return found

    # --- validation -------------------------------------------------------

    def _validate_structure(self) -> None:
        for role in ("input", "actor", "value"):
            named = self.nodes_with_role(role)
            if len(named) != 1:
                described = f"node(s) {list(named)}" if named else "no node"
                raise ConfigError(
                    f"policy graph needs exactly one {role!r} node, found {described}."
                )
        for sink in (self.actor_sink, self.value_sink):
            if self.outgoing(sink):
                raise ConfigError(
                    f"sink node {sink!r} has an outgoing edge; actor/value sinks "
                    "only consume."
                )
        _, remaining = self._kahn_result
        if remaining:
            raise ConfigError(f"policy graph has a cycle through node(s) {remaining}.")
        loss_nodes = {loss.node for loss in self.losses}
        for node in self.nodes:
            self._validate_node_edges(node, loss_nodes)
        for loss in self.losses:
            if loss.node not in {node.name for node in self.nodes}:
                raise ConfigError(f"loss on unknown node {loss.node!r}.")
            if self.node_type(loss.node).role is not None:
                raise ConfigError(
                    f"loss on node {loss.node!r}: input and sink nodes produce no "
                    "output of their own to put a loss on."
                )

    def _validate_node_edges(self, node: NodeSpec, loss_nodes: set[str]) -> None:
        node_type = self.node_type(node.name)
        incoming = self.incoming(node.name)
        input_name = self.input_node
        if node_type.role == "input":
            if incoming:
                raise ConfigError(f"input node {node.name!r} cannot have incoming edges.")
        elif node_type.reads_observation:
            if len(incoming) != 1 or incoming[0].source != input_name:
                raise ConfigError(
                    f"node {node.name!r} (type {node.type!r}) reads the observation: "
                    f"it needs exactly one incoming edge, from the input node {input_name!r}."
                )
        else:
            if any(edge.source == input_name for edge in incoming):
                raise ConfigError(
                    f"node {node.name!r} (type {node.type!r}) cannot read the input node "
                    f"{input_name!r} directly; put an encoder node in between."
                )
            if node_type.multi_input:
                if not incoming:
                    raise ConfigError(f"node {node.name!r} has no incoming edge.")
            elif len(incoming) != 1:
                raise ConfigError(
                    f"node {node.name!r} (type {node.type!r}) takes exactly one incoming "
                    f"edge, found {len(incoming)}."
                )
        for edge in self.outgoing(node.name):
            if node_type.role == "input" and edge.detach:
                raise ConfigError(
                    f"edge {node.name!r} -> {edge.target!r}: an edge out of the input "
                    "node carries the observation and cannot detach."
                )
        if (
            node_type.role not in ("actor", "value")
            and not self.outgoing(node.name)
            and node.name not in loss_nodes
        ):
            raise ConfigError(
                f"node {node.name!r} has no outgoing edge and no loss (dead node)."
            )


def _kahn_order(spec: GraphSpec) -> tuple[tuple[str, ...], list[str]]:
    in_degree = {node.name: 0 for node in spec.nodes}
    for edge in spec.edges:
        in_degree[edge.target] += 1
    ready = [name for name, degree in in_degree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        for edge in spec.outgoing(name):
            in_degree[edge.target] -= 1
            if in_degree[edge.target] == 0:
                heapq.heappush(ready, edge.target)
    # Unsorted nodes are on a cycle or downstream of one; drop the latter.
    remaining = {name for name, degree in in_degree.items() if degree > 0}
    pruned = True
    while pruned:
        downstream = {
            name
            for name in remaining
            if not any(edge.target in remaining for edge in spec.outgoing(name))
        }
        remaining -= downstream
        pruned = bool(downstream)
    return tuple(order), sorted(remaining)


# --- parsing helpers ---------------------------------------------------------


def _to_plain(value: Any) -> Any:
    """Plain dict/list copies of nested mappings/sequences (e.g. OmegaConf
    containers from an RLinf config)."""
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_plain(item) for item in value]
    return value


def _load_mapping(source: Union[str, Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(source, str):
        try:
            loaded = yaml.load(source, Loader=UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise ConfigError(f"policy graph is not valid YAML: {exc}") from exc
    else:
        loaded = source
    if not isinstance(loaded, Mapping):
        raise ConfigError(
            f"policy graph must be a mapping with 'nodes' and 'edges', got "
            f"{type(loaded).__name__}."
        )
    return _to_plain(loaded)


def _parse_nodes(raw: Any) -> list[NodeSpec]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("policy graph needs a non-empty 'nodes' mapping.")
    nodes = []
    for name, body in raw.items():
        if not isinstance(name, str) or not _NODE_NAME.match(name):
            raise ConfigError(f"node name {name!r} is not a Python identifier.")
        if name in _RESERVED_NODE_NAMES:
            raise ConfigError(
                f"node name {name!r} is reserved: it is an attribute of nn.ModuleDict, "
                "which holds the graph's node modules; rename the node."
            )
        if not isinstance(body, dict) or "type" not in body:
            raise ConfigError(f"node {name!r} must be a mapping with a 'type' field.")
        type_name = body["type"]
        node_type = find_graph_node_type(type_name) if isinstance(type_name, str) else None
        if node_type is None:
            raise ConfigError(
                f"node {name!r} has unknown type {type_name!r}; known types: "
                f"{known_graph_node_types()}."
            )
        params = {key: value for key, value in body.items() if key != "type"}
        nodes.append(NodeSpec(name, type_name, _node_params(name, node_type, params)))
    return nodes


def _node_params(name: str, node_type: GraphNodeType, given: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(given) - set(node_type.params))
    if unknown:
        raise ConfigError(
            f"node {name!r} (type {node_type.name!r}) has unknown field(s) {unknown}; "
            f"allowed: {sorted(node_type.params)}."
        )
    params = {}
    for key, default in node_type.params.items():
        if key in given:
            params[key] = given[key]
        elif default is REQUIRED:
            raise ConfigError(
                f"node {name!r} (type {node_type.name!r}) is missing required field {key!r}."
            )
        else:
            params[key] = _to_plain(default)
    if node_type.normalize_params is not None:
        params = node_type.normalize_params(name, params)
    return params


def _parse_edges(raw: Any, node_names: set[str]) -> list[EdgeSpec]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("policy graph needs a non-empty 'edges' list.")
    edges = []
    seen = set()
    for body in raw:
        if not isinstance(body, dict):
            raise ConfigError(f"edge {body!r} must be a mapping with 'from' and 'to'.")
        unknown = sorted(set(body) - set(_EDGE_FIELDS))
        if unknown:
            raise ConfigError(
                f"edge {body!r} has unknown field(s) {unknown}; allowed: {list(_EDGE_FIELDS)}."
            )
        source, target = body.get("from"), body.get("to")
        for end, name in (("from", source), ("to", target)):
            if name not in node_names:
                raise ConfigError(
                    f"edge {source!r} -> {target!r}: '{end}' names unknown node "
                    f"{name!r} (dangling edge)."
                )
        detach = body.get("detach", False)
        if not isinstance(detach, bool):
            raise ConfigError(f"edge {source!r} -> {target!r}: 'detach' must be true or false.")
        if (source, target) in seen:
            raise ConfigError(f"edge {source!r} -> {target!r} is declared twice.")
        seen.add((source, target))
        edges.append(EdgeSpec(source, target, detach))
    return edges


def _parse_losses(raw: Any) -> list[LossSpec]:
    if not isinstance(raw, list):
        raise ConfigError("policy graph 'losses' must be a list.")
    losses = []
    for body in raw:
        if not isinstance(body, dict):
            raise ConfigError(f"loss {body!r} must be a mapping.")
        unknown = sorted(set(body) - set(_LOSS_FIELDS))
        if unknown:
            raise ConfigError(
                f"loss {body!r} has unknown field(s) {unknown}; allowed: {list(_LOSS_FIELDS)}."
            )
        node = body.get("node")
        if not isinstance(node, str):
            raise ConfigError(f"loss {body!r} needs a 'node' name.")
        loss_type = body.get("type")
        registered = find_graph_loss(loss_type) if isinstance(loss_type, str) else None
        if registered is None:
            raise ConfigError(
                f"loss on node {node!r} has type {loss_type!r}; known: "
                f"{known_graph_losses()}."
            )
        target = body.get("target")
        if not isinstance(target, str) or not (
            target in BATCH_LOSS_TARGETS or _INFO_TARGET.match(target)
        ):
            raise ConfigError(
                f"loss on node {node!r} has target {target!r}; expected 'info.<key>' "
                f"or one of {list(BATCH_LOSS_TARGETS)}."
            )
        if registered.class_targets and target in BATCH_LOSS_TARGETS:
            raise ConfigError(
                f"loss on node {node!r}: {loss_type!r} needs integer class targets, "
                f"which {target!r} is not; use an 'info.<key>' target."
            )
        weight = body.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not (
            math.isfinite(weight) and weight >= 0
        ):
            raise ConfigError(
                f"loss on node {node!r}: 'weight' must be a finite number >= 0."
            )
        mask = body.get("mask", "decision")
        if mask not in LOSS_MASKS:
            raise ConfigError(
                f"loss on node {node!r} has mask {mask!r}; expected one of {list(LOSS_MASKS)}."
            )
        if node in {loss.node for loss in losses}:
            raise ConfigError(
                f"node {node!r} carries more than one loss; put each loss on its own node."
            )
        losses.append(LossSpec(node, loss_type, target, float(weight), mask))
    return losses


# --- CLI sources -------------------------------------------------------------


def is_builtin_graph_source(source: str) -> bool:
    return source.startswith(BUILTIN_GRAPH_PREFIX)


def _is_inline_yaml(source: str) -> bool:
    return "\n" in source or "{" in source or ": " in source


def read_graph_source(source: str) -> str:
    """Resolve a ``--policy-graph`` value: ``builtin:<name>`` is returned
    unchanged; text with a newline, ``{`` or ``": "`` is inline YAML;
    anything else is a path, read from disk (a missing or unreadable file
    raises ``ConfigError``)."""
    if is_builtin_graph_source(source) or _is_inline_yaml(source):
        return source
    try:
        return Path(source).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"policy graph file not found: {source}") from exc
    except OSError as exc:
        raise ConfigError(f"policy graph file {source!r} cannot be read: {exc}") from exc


def optional_graph_source(source: Optional[str]) -> Optional[str]:
    return None if source is None else read_graph_source(source)
