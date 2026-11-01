"""In-tree policy-graph node types and the graph module builder.

Every non-structural node is an ``nn.Module`` with a ``features_dim``
(output width). Stateful nodes are the existing sequence encoders
themselves (``RecurrentLatentEncoder``, ``GTrXLLatentEncoder``), so their
parameters keep the same names as in ``RecurrentPPOPolicy.recurrent_encoder``;
``encoder`` nodes are the extractor ``build_observation_encoder`` returns.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

from rl_garden.common.effective_config import ConfigError
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.factory import build_observation_encoder
from rl_garden.networks.graph.registry import (
    REQUIRED,
    GraphNodeType,
    add_graph_node_type,
    register_graph_node,
)
from rl_garden.networks.graph.spec import GraphSpec, NodeSpec
from rl_garden.networks.gtrxl import GTrXLLatentEncoder
from rl_garden.networks.mlp import create_mlp, resolve_activation
from rl_garden.networks.recurrent import RecurrentLatentEncoder


@runtime_checkable
class GraphNode(Protocol):
    """A non-stateful graph node. Stateful nodes implement
    ``rl_garden.networks.sequence_encoder.SequenceLatentEncoder`` instead of
    ``forward``."""

    @property
    def features_dim(self) -> int: ...

    def forward(self, x: Any) -> torch.Tensor: ...


class MLPNode(nn.Module):
    def __init__(
        self,
        input_dim: int,
        net_arch: list[int],
        output_dim: Optional[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.net = create_mlp(
            input_dim,
            output_dim if output_dim is not None else -1,
            net_arch,
            activation_fn=resolve_activation(activation, nn.ReLU),
        )
        self._features_dim = output_dim if output_dim is not None else net_arch[-1]

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConcatNode(nn.Module):
    """Concatenates its inputs along the feature axis (producer-name order)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._features_dim = input_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        return inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)


# --- parameter checks ----------------------------------------------------------


def _positive_int(name: str, field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"node {name!r}: {field!r} must be a positive integer, got {value!r}.")
    return value


def _number(name: str, field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"node {name!r}: {field!r} must be a number, got {value!r}.")
    return float(value)


def _normalize_encoder(name: str, params: dict) -> dict:
    keys = params["keys"]
    if keys is not None:
        if (
            not isinstance(keys, list)
            or not keys
            or not all(isinstance(key, str) for key in keys)
            or len(set(keys)) != len(keys)
        ):
            raise ConfigError(
                f"node {name!r}: 'keys' must be null or a non-empty list of distinct "
                f"observation keys, got {keys!r}."
            )
    config = params["config"]
    if not isinstance(config, dict):
        raise ConfigError(f"node {name!r}: 'config' must be a mapping of EncoderConfig fields.")
    try:
        encoder_config = EncoderConfig(**config)
    except TypeError as exc:
        known = sorted(field.name for field in dataclasses.fields(EncoderConfig))
        raise ConfigError(
            f"node {name!r}: 'config' has a field EncoderConfig does not know ({exc}); "
            f"known: {known}."
        ) from exc
    return {"keys": keys, "config": dataclasses.asdict(encoder_config)}


def _normalize_mlp(name: str, params: dict) -> dict:
    net_arch = params["net_arch"]
    if not isinstance(net_arch, list):
        raise ConfigError(f"node {name!r}: 'net_arch' must be a list of layer widths.")
    net_arch = [_positive_int(name, "net_arch", width) for width in net_arch]
    output_dim = params["output_dim"]
    if output_dim is not None:
        output_dim = _positive_int(name, "output_dim", output_dim)
    elif not net_arch:
        raise ConfigError(f"node {name!r}: an empty 'net_arch' needs an 'output_dim'.")
    if params["activation"] not in ("relu", "gelu", "mish"):
        raise ConfigError(
            f"node {name!r}: 'activation' must be relu, gelu or mish, got "
            f"{params['activation']!r}."
        )
    return {"net_arch": net_arch, "output_dim": output_dim, "activation": params["activation"]}


def _normalize_rnn(name: str, params: dict) -> dict:
    return {key: _positive_int(name, key, value) for key, value in params.items()}


_GTRXL_FLOAT_PARAMS = ("dropout_rate", "gru_bias")


def _normalize_gtrxl(name: str, params: dict) -> dict:
    normalized = {
        key: (_number(name, key, value) if key in _GTRXL_FLOAT_PARAMS else _positive_int(name, key, value))
        for key, value in params.items()
    }
    if normalized["embed_dim"] % 2 != 0:
        raise ConfigError(f"node {name!r}: 'embed_dim' must be even.")
    if not 0.0 <= normalized["dropout_rate"] < 1.0:
        raise ConfigError(f"node {name!r}: 'dropout_rate' must be in [0, 1).")
    return normalized


# --- factories -------------------------------------------------------------------


def _encoder_factory(spec: NodeSpec, input_dim, observation_space, action_space) -> nn.Module:
    del input_dim, action_space
    return build_observation_encoder(
        observation_space,
        EncoderConfig(**spec.params["config"]),
        keys=spec.params["keys"],
    )


def _mlp_factory(spec: NodeSpec, input_dim, observation_space, action_space) -> nn.Module:
    del observation_space, action_space
    return MLPNode(input_dim, **spec.params)


def _concat_factory(spec: NodeSpec, input_dim, observation_space, action_space) -> nn.Module:
    del spec, observation_space, action_space
    return ConcatNode(input_dim)


def _rnn_factory(rnn_type: str):
    def factory(spec: NodeSpec, input_dim, observation_space, action_space) -> nn.Module:
        del observation_space, action_space
        return RecurrentLatentEncoder(
            input_dim=input_dim,
            hidden_size=spec.params["hidden_size"],
            rnn_type=rnn_type,
            num_layers=spec.params["num_layers"],
        )

    return factory


def _gtrxl_factory(spec: NodeSpec, input_dim, observation_space, action_space) -> nn.Module:
    del observation_space, action_space
    return GTrXLLatentEncoder(input_dim=input_dim, **spec.params)


for _role in ("input", "actor", "value"):
    add_graph_node_type(GraphNodeType(name=_role, factory=None, role=_role))
register_graph_node(
    "encoder",
    _encoder_factory,
    params={"keys": None, "config": {}},
    reads_observation=True,
    normalize_params=_normalize_encoder,
)
register_graph_node(
    "mlp",
    _mlp_factory,
    params={"net_arch": REQUIRED, "output_dim": None, "activation": "relu"},
    normalize_params=_normalize_mlp,
)
register_graph_node("concat", _concat_factory, multi_input=True)
for _rnn_type in ("gru", "lstm"):
    register_graph_node(
        _rnn_type,
        _rnn_factory(_rnn_type),
        params={"hidden_size": 256, "num_layers": 1},
        stateful=True,
        normalize_params=_normalize_rnn,
    )
register_graph_node(
    "gtrxl",
    _gtrxl_factory,
    params={
        "embed_dim": 256,
        "head_dim": 64,
        "num_heads": 4,
        "num_layers": 3,
        "mlp_num": 2,
        "memory_len": 64,
        "dropout_rate": 0.0,
        "gru_bias": 2.0,
    },
    stateful=True,
    normalize_params=_normalize_gtrxl,
)


# --- builder ---------------------------------------------------------------------


def build_graph_nodes(
    spec: GraphSpec, observation_space, action_space
) -> tuple[dict[str, nn.Module], dict[str, int]]:
    """Construct every node module in ``spec.topological_order()`` (the
    RNG-relevant order). Returns the modules (structural nodes have none)
    and every node's output width; a sink's width is its input's width."""
    modules: dict[str, nn.Module] = {}
    dims: dict[str, int] = {}
    for name in spec.topological_order():
        node_type = spec.node_type(name)
        if node_type.role == "input":
            continue
        incoming = spec.incoming(name)
        input_dim = (
            None
            if node_type.reads_observation
            else sum(dims[edge.source] for edge in incoming)
        )
        if node_type.role is not None:
            dims[name] = input_dim
            continue
        module = node_type.factory(spec.node(name), input_dim, observation_space, action_space)
        features_dim = getattr(module, "features_dim", None)
        if not isinstance(module, nn.Module) or not isinstance(features_dim, int):
            raise ConfigError(
                f"node {name!r}: the {node_type.name!r} factory must return an nn.Module "
                "with an integer features_dim."
            )
        modules[name] = module
        dims[name] = features_dim
    return modules, dims
