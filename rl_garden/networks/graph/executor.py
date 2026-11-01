"""Runs a policy graph in the three modes PPO needs.

* ``run_flat(graph, nodes, obs, sinks)``: one batch, no state (``PPO``).
* ``run_step(graph, nodes, obs, state, episode_starts)``: one rollout step of
  a stateful graph; every stateful node calls ``step``.
* ``run_sequence(graph, nodes, obs, state, episode_starts)``: a ``(T, B)``
  BPTT window. Every tensor travels flattened to ``(T * B, ...)``; a
  stateful node reshapes its input to ``(T, B, -1)``, unrolls
  ``forward_sequence`` and flattens its output back, exactly like
  ``RecurrentPPOPolicy.evaluate_actions_sequence``. ``episode_starts[0]`` is
  honored as given (see ``RecurrentLatentEncoder.forward_sequence``).

Pure functions of the ``GraphSpec``, the node modules and the inputs: the
returned ``outputs`` dict (node name -> output; a sink maps to the input its
head reads, after any ``detach``) is built per call. Each edge's ``detach``
applies to that consumer's view only.

Policy hidden state is one flat tuple of tensors, batch at dim 1 (what
``RecurrentRolloutBuffer`` indexes): the stateful nodes' own states
concatenated in topological order. ``StateLayout`` maps it to and from the
per-node states.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

from rl_garden.common.obs_utils import flatten_leading_dims
from rl_garden.networks.graph.spec import GraphSpec

GraphState = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class StateLayout:
    """Where each stateful node's state sits in the flat policy state."""

    #: (node name, number of tensors, whether the node's own state is a tuple)
    parts: tuple[tuple[str, int, bool], ...]

    @classmethod
    def from_nodes(cls, spec: GraphSpec, nodes: Mapping[str, nn.Module]) -> "StateLayout":
        parts = []
        for name in spec.stateful_nodes:
            probe = nodes[name].get_initial_state(1, torch.device("cpu"))
            is_tuple = isinstance(probe, tuple)
            parts.append((name, len(probe) if is_tuple else 1, is_tuple))
        return cls(tuple(parts))

    def split(self, state: GraphState) -> dict[str, Any]:
        expected = sum(count for _, count, _ in self.parts)
        if not isinstance(state, tuple) or len(state) != expected:
            raise ValueError(
                f"graph hidden state must be a tuple of {expected} tensors, got "
                f"{type(state).__name__} of length {len(state) if isinstance(state, tuple) else 1}."
            )
        per_node: dict[str, Any] = {}
        offset = 0
        for name, count, is_tuple in self.parts:
            chunk = state[offset : offset + count]
            per_node[name] = chunk if is_tuple else chunk[0]
            offset += count
        return per_node

    def join(self, per_node: Mapping[str, Any]) -> GraphState:
        flat: list[torch.Tensor] = []
        for name, _, is_tuple in self.parts:
            flat.extend(per_node[name] if is_tuple else (per_node[name],))
        return tuple(flat)

    def initial(self, nodes: Mapping[str, nn.Module], batch_size: int, device) -> GraphState:
        return self.join(
            {name: nodes[name].get_initial_state(batch_size, device) for name, _, _ in self.parts}
        )

    def mask(self, nodes: Mapping[str, nn.Module], state: GraphState, keep_mask: torch.Tensor) -> GraphState:
        return self.join(
            {
                name: nodes[name].mask_state(node_state, keep_mask)
                for name, node_state in self.split(state).items()
            }
        )

    def index(self, nodes: Mapping[str, nn.Module], state: GraphState, indices: torch.Tensor) -> GraphState:
        return self.join(
            {
                name: nodes[name].index_state(node_state, indices)
                for name, node_state in self.split(state).items()
            }
        )


def execution_order(spec: GraphSpec, targets: Iterable[str], *, with_stateful: bool) -> tuple[str, ...]:
    """Topologically ordered nodes needed for ``targets`` (plus, with
    ``with_stateful``, every stateful node so the whole state advances),
    excluding the input node."""
    needed: set[str] = set()
    roots = list(targets) + (list(spec.stateful_nodes) if with_stateful else [])
    for name in roots:
        needed.add(name)
        needed |= spec.ancestors(name)
    needed.discard(spec.input_node)
    return tuple(name for name in spec.topological_order() if name in needed)


def _node_input(spec: GraphSpec, outputs: Mapping[str, Any], name: str) -> Any:
    values = [
        outputs[edge.source].detach() if edge.detach else outputs[edge.source]
        for edge in spec.incoming(name)
    ]
    return values if spec.node_type(name).multi_input else values[0]


def run_flat(
    spec: GraphSpec,
    nodes: Mapping[str, nn.Module],
    obs: Any,
    targets: Iterable[str],
) -> dict[str, Any]:
    """Outputs of every node ``targets`` needs, for a flat batch. The graph
    must be stateless."""
    outputs: dict[str, Any] = {spec.input_node: obs}
    for name in execution_order(spec, targets, with_stateful=False):
        node_input = _node_input(spec, outputs, name)
        outputs[name] = node_input if name not in nodes else nodes[name](node_input)
    return outputs


def run_step(
    spec: GraphSpec,
    nodes: Mapping[str, nn.Module],
    obs: Any,
    state: Mapping[str, Any],
    episode_starts: Optional[torch.Tensor],
    targets: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One rollout step. ``state``/the returned new state are per-node
    (see ``StateLayout``)."""
    outputs: dict[str, Any] = {spec.input_node: obs}
    new_state: dict[str, Any] = {}
    for name in execution_order(spec, targets, with_stateful=True):
        node_input = _node_input(spec, outputs, name)
        if name not in nodes:
            outputs[name] = node_input
        elif name in state:
            outputs[name], new_state[name] = nodes[name].step(
                node_input, state[name], episode_starts
            )
        else:
            outputs[name] = nodes[name](node_input)
    return outputs, new_state


def run_sequence(
    spec: GraphSpec,
    nodes: Mapping[str, nn.Module],
    obs: Any,
    state: Mapping[str, Any],
    episode_starts: torch.Tensor,
    targets: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A ``(T, B)`` window: ``obs`` is ``(T, B, ...)``, ``episode_starts``
    ``(T, B)``; every output is flattened to ``(T * B, ...)`` (T-major)."""
    num_steps, num_envs = episode_starts.shape[0], episode_starts.shape[1]
    outputs: dict[str, Any] = {spec.input_node: flatten_leading_dims(obs)}
    final_state: dict[str, Any] = {}
    for name in execution_order(spec, targets, with_stateful=True):
        node_input = _node_input(spec, outputs, name)
        if name not in nodes:
            outputs[name] = node_input
        elif name in state:
            sequence, final_state[name] = nodes[name].forward_sequence(
                node_input.reshape(num_steps, num_envs, -1), state[name], episode_starts
            )
            outputs[name] = sequence.reshape(num_steps * num_envs, -1)
        else:
            outputs[name] = nodes[name](node_input)
    return outputs, final_state
