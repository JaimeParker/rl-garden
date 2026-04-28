from __future__ import annotations

from typing import Sequence

import torch.nn as nn


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: Sequence[int],
    *,
    activation_fn: type[nn.Module] = nn.ReLU,
    use_layer_norm: bool = False,
    with_bias: bool = True,
    squash_output: bool = False,
) -> nn.Sequential:
    """Build an MLP from an SB3-style architecture spec.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension. Set to -1 to omit the output layer.
        net_arch: Hidden layer sizes.
        activation_fn: Activation after hidden layers.
        use_layer_norm: Add LayerNorm after hidden linear layers.
        with_bias: Whether linear layers use bias.
        squash_output: Append Tanh at the end.
    """
    layers: list[nn.Module] = []
    c = input_dim
    for h in net_arch:
        layers.append(nn.Linear(c, h, bias=with_bias))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation_fn())
        c = h

    if output_dim > 0:
        layers.append(nn.Linear(c, output_dim, bias=with_bias))
    if squash_output:
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)
