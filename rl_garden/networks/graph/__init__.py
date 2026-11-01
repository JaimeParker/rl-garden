"""YAML-declared policy graphs for the PPO family.

``spec`` holds the YAML reference and validation, ``registry`` the node
types, ``nodes`` the in-tree node modules, ``executor`` the three run modes,
``builtin`` the graphs that reproduce ``PPOPolicy``/``RecurrentPPOPolicy``,
``losses`` the auxiliary loss types.
``rl_garden.policies.graph_ppo_policy.GraphPPOPolicy`` turns a spec into a
policy. The user guide is ``docs/guides/policy-graphs.md``.
"""

from rl_garden.networks.graph.losses import GRAPH_LOSS_REGISTRY, register_graph_loss
from rl_garden.networks.graph.registry import (
    GRAPH_NODE_REGISTRY,
    REQUIRED,
    GraphNodeType,
    discover_graph_nodes,
    register_graph_node,
)
from rl_garden.networks.graph.spec import EdgeSpec, GraphSpec, LossSpec, NodeSpec

__all__ = [
    "GRAPH_LOSS_REGISTRY",
    "GRAPH_NODE_REGISTRY",
    "REQUIRED",
    "EdgeSpec",
    "GraphNodeType",
    "GraphSpec",
    "LossSpec",
    "NodeSpec",
    "discover_graph_nodes",
    "register_graph_loss",
    "register_graph_node",
]
