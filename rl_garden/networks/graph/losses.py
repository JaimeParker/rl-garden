"""Auxiliary losses of a policy graph (the ``losses:`` section of the spec).

A loss function maps a node's output ``pred`` (rows, D) and its target to an
unreduced per-row loss (rows,). The algorithm reduces it with the same row
mean as its policy loss and adds ``weight *`` that mean to the PPO loss.

Built-in types: ``mse`` and ``bce_with_logits`` (float target with the
node's shape, ``(rows, D)``; ``bce_with_logits`` reads ``pred`` as logits),
``cross_entropy`` (``pred`` holds class logits, the target is an integer
class index per row). Register more with ``register_graph_loss`` before the
graph is parsed.

Per-env target layout of an ``info.<key>`` target (what the env reports on
every step, with a leading ``num_envs`` axis): ``(D,)`` floating point (or
bool) for a float-target loss, where ``D`` is the node's output width
(``(N,)`` is accepted when ``D == 1``); ``()`` integer for a class-index
loss. ``batch.rewards``/``batch.dones`` are one float per row, so they need
a float-target loss on a node of width 1.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from rl_garden.common.effective_config import ConfigError

if TYPE_CHECKING:
    from rl_garden.networks.graph.spec import GraphSpec

GraphLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
#: ``info.<key>`` -> (per-env target shape, stored dtype).
TargetSpecs = dict[str, tuple[tuple[int, ...], torch.dtype]]

INFO_TARGET_PREFIX = "info."


@dataclass(frozen=True)
class GraphLossType:
    name: str
    fn: GraphLossFn
    #: True: the target is one integer class index per row; False: a float
    #: target with the prediction's shape.
    class_targets: bool = False


GRAPH_LOSS_REGISTRY: dict[str, GraphLossType] = {}


def register_graph_loss(name: str, fn: GraphLossFn, *, class_targets: bool = False) -> None:
    """Register ``fn(pred, target) -> (rows,)`` under ``name``."""
    if name in GRAPH_LOSS_REGISTRY:
        raise ValueError(f"graph loss type {name!r} is already registered.")
    GRAPH_LOSS_REGISTRY[name] = GraphLossType(name, fn, class_targets)


def find_graph_loss(name: str) -> GraphLossType | None:
    return GRAPH_LOSS_REGISTRY.get(name)


def known_graph_losses() -> list[str]:
    return sorted(GRAPH_LOSS_REGISTRY)


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(-1)


def _bce_with_logits(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean(-1)


def _cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(pred, target, reduction="none")


register_graph_loss("mse", _mse)
register_graph_loss("bce_with_logits", _bce_with_logits)
register_graph_loss("cross_entropy", _cross_entropy, class_targets=True)


def target_rows(
    loss_type: GraphLossType, pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``pred`` as ``(rows, D)`` and ``target`` laid out for ``loss_type``.
    Both are flattened with ``reshape`` in their own row order, which is
    the same for every term of a minibatch (T-major for sequences)."""
    pred = pred.reshape(-1, pred.shape[-1])
    if loss_type.class_targets:
        return pred, target.reshape(-1).long()
    return pred, target.reshape(pred.shape).to(pred.dtype)


def info_target_specs(spec: GraphSpec, node_dims: Mapping[str, int]) -> TargetSpecs:
    """Per-env layout of every ``info.<key>`` target of ``spec``'s losses,
    keyed by ``<key>``. Raises ``ConfigError`` for a ``batch.*`` target on a
    node wider than 1 and for one key used with two different layouts."""
    specs: TargetSpecs = {}
    for loss in spec.losses:
        loss_type = find_graph_loss(loss.type)
        assert loss_type is not None, loss.type  # checked by GraphSpec.parse()
        width = node_dims[loss.node]
        if not loss.target.startswith(INFO_TARGET_PREFIX):
            if width != 1:
                raise ConfigError(
                    f"loss on node {loss.node!r}: target {loss.target!r} is one value per "
                    f"step, but the node outputs {width}; give it output_dim 1."
                )
            continue
        layout = ((), torch.int64) if loss_type.class_targets else ((width,), torch.float32)
        key = loss.target[len(INFO_TARGET_PREFIX):]
        if specs.setdefault(key, layout) != layout:
            raise ConfigError(
                f"target {loss.target!r} is used by losses that need different layouts "
                f"({specs[key]} and {layout})."
            )
    return specs


def read_info_targets(
    infos: Mapping[str, Any], specs: TargetSpecs, num_envs: int, step: int
) -> dict[str, torch.Tensor]:
    """The ``info.<key>`` targets of one rollout step, checked against
    ``specs`` with tensor metadata and host-side dict lookups only (no
    device sync). Raises ``ValueError`` naming the key on a missing key,
    a non-tensor, or a wrong shape or dtype."""
    targets = {}
    for key, (shape, dtype) in specs.items():
        if key not in infos:
            raise ValueError(
                f"graph loss target 'info.{key}' is missing from infos at rollout step "
                f"{step}; the env must report infos[{key!r}] on every step."
            )
        value = infos[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(  # noqa: TRY004
                f"infos[{key!r}] (graph loss target) must be a torch.Tensor, got "
                f"{type(value).__name__}."
            )
        expected = (num_envs, *shape)
        accepted = {expected, (num_envs,)} if shape == (1,) else {expected}
        if tuple(value.shape) not in accepted:
            raise ValueError(
                f"infos[{key!r}] (graph loss target) must have shape {expected}, got "
                f"{tuple(value.shape)}."
            )
        if dtype == torch.int64:
            dtype_ok = not value.dtype.is_floating_point and value.dtype != torch.bool
        else:
            dtype_ok = value.dtype.is_floating_point or value.dtype == torch.bool
        if not dtype_ok:
            wanted = "an integer (class index)" if dtype == torch.int64 else "a float or bool"
            raise ValueError(
                f"infos[{key!r}] (graph loss target) must have {wanted} dtype, got "
                f"{value.dtype}."
            )
        targets[key] = value
    return targets
