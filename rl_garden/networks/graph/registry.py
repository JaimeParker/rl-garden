"""Policy-graph node-type registry.

Every ``type:`` a graph spec may use is a :class:`GraphNodeType` in
``GRAPH_NODE_REGISTRY``. The in-tree types live in
``rl_garden.networks.graph.nodes``; an external package adds its own by
calling :func:`register_graph_node` at import time and advertising that
module under the ``rlgarden.graph_nodes`` entry-point group in its own
``pyproject.toml``::

    [project.entry-points."rlgarden.graph_nodes"]
    my_node = "my_package.graph_nodes"

Discovery is one-shot, locked, and lazy (first lookup), mirroring
``rl_garden.envs.backend_registry.discover_env_backends``: importing
``rl_garden`` never imports a node module.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional

_GRAPH_NODE_ENTRY_POINT_GROUP = "rlgarden.graph_nodes"
_IN_TREE_NODE_MODULE = "rl_garden.networks.graph.nodes"

NodeRole = Literal["input", "actor", "value"]


class _Required:
    """Marker default for a node parameter the spec must give."""

    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()

#: ``factory(spec, input_dim, observation_space, action_space) -> nn.Module``.
#: ``spec`` is the node's ``NodeSpec`` (parameters already defaulted and
#: normalized); ``input_dim`` is the summed width of the node's inputs, or
#: ``None`` for a node that reads the observation. The returned module exposes
#: ``features_dim`` (its output width) and ``forward(x)``; a stateful node
#: implements ``SequenceLatentEncoder`` instead of a plain ``forward``.
GraphNodeFactory = Callable[[Any, Optional[int], Any, Any], Any]


@dataclass(frozen=True)
class GraphNodeType:
    """What the spec validator and the policy builder need to know about one
    node type."""

    name: str
    factory: Optional[GraphNodeFactory]
    #: Parameter name -> default (``REQUIRED`` when the spec must give it).
    params: Mapping[str, Any] = field(default_factory=dict)
    #: Carries hidden state across steps (``SequenceLatentEncoder``).
    stateful: bool = False
    #: Consumes the observation itself (only from the ``input`` node).
    reads_observation: bool = False
    #: Takes one or more incoming edges instead of exactly one.
    multi_input: bool = False
    #: ``normalize_params(node_name, params) -> params``: deeper validation
    #: and canonicalization of a node's parameters; raises ``ConfigError``.
    normalize_params: Optional[Callable[[str, dict], dict]] = None
    #: Structural role of the built-in ``input``/``actor``/``value`` types.
    role: Optional[NodeRole] = None


GRAPH_NODE_REGISTRY: dict[str, GraphNodeType] = {}
_DISCOVERED = False
_DISCOVERY_LOCK = threading.Lock()


def add_graph_node_type(node_type: GraphNodeType) -> None:
    if node_type.name in GRAPH_NODE_REGISTRY:
        raise ValueError(f"Graph node type {node_type.name!r} already registered")
    GRAPH_NODE_REGISTRY[node_type.name] = node_type


def register_graph_node(
    name: str,
    factory: GraphNodeFactory,
    *,
    params: Optional[Mapping[str, Any]] = None,
    stateful: bool = False,
    reads_observation: bool = False,
    multi_input: bool = False,
    normalize_params: Optional[Callable[[str, dict], dict]] = None,
) -> None:
    """Register a graph node type usable as ``type: <name>`` in a spec.

    ``params`` maps each accepted parameter to its default (``REQUIRED`` for
    a mandatory one); a spec field outside it is rejected.
    """
    add_graph_node_type(
        GraphNodeType(
            name=name,
            factory=factory,
            params=dict(params or {}),
            stateful=stateful,
            reads_observation=reads_observation,
            multi_input=multi_input,
            normalize_params=normalize_params,
        )
    )


def discover_graph_nodes() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    with _DISCOVERY_LOCK:
        if _DISCOVERED:
            return
        importlib.import_module(_IN_TREE_NODE_MODULE)
        for entry_point in importlib.metadata.entry_points(
            group=_GRAPH_NODE_ENTRY_POINT_GROUP
        ):
            entry_point.load()
        _DISCOVERED = True


def known_graph_node_types() -> list[str]:
    discover_graph_nodes()
    return sorted(GRAPH_NODE_REGISTRY)


def find_graph_node_type(name: str) -> Optional[GraphNodeType]:
    """The registered type ``name``, or ``None`` when no type has that name."""
    discover_graph_nodes()
    return GRAPH_NODE_REGISTRY.get(name)
