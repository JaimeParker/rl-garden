from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn

KernelInit = Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]


def _apply_kernel_init(module: nn.Module, kernel_init: Optional[KernelInit]) -> None:
    if kernel_init is None:
        return
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if kernel_init == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif kernel_init == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif kernel_init == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif kernel_init == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown kernel_init: {kernel_init!r}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _make_norm(
    num_features: int,
    *,
    use_layer_norm: bool,
    use_group_norm: bool,
    num_groups: int,
) -> Optional[nn.Module]:
    if use_layer_norm and use_group_norm:
        raise ValueError("use_layer_norm and use_group_norm cannot both be True.")
    if use_layer_norm:
        return nn.LayerNorm(num_features)
    if use_group_norm:
        if num_features % num_groups != 0:
            # fall back to a divisor of num_features to avoid hard crash on odd dims
            for g in (num_groups, 16, 8, 4, 2, 1):
                if num_features % g == 0:
                    num_groups = g
                    break
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    return None


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: Sequence[int],
    *,
    activation_fn: type[nn.Module] = nn.ReLU,
    use_layer_norm: bool = False,
    use_group_norm: bool = False,
    num_groups: int = 32,
    dropout_rate: Optional[float] = None,
    kernel_init: Optional[KernelInit] = None,
    with_bias: bool = True,
    squash_output: bool = False,
) -> nn.Sequential:
    """Build an MLP from an SB3-style architecture spec.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output dimension. Set to -1 to omit the output layer.
        net_arch: Hidden layer sizes.
        activation_fn: Activation after hidden layers.
        use_layer_norm: Add LayerNorm after hidden linear layers (mutually exclusive with use_group_norm).
        use_group_norm: Add GroupNorm after hidden linear layers (mutually exclusive with use_layer_norm).
        num_groups: GroupNorm group count (clipped to a divisor of layer width if needed).
        dropout_rate: If not None, append Dropout after each hidden activation.
        kernel_init: Optional initialization for nn.Linear weights.
        with_bias: Whether linear layers use bias.
        squash_output: Append Tanh at the end.
    """
    layers: list[nn.Module] = []
    c = input_dim
    for h in net_arch:
        layers.append(nn.Linear(c, h, bias=with_bias))
        norm = _make_norm(
            h,
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
        )
        if norm is not None:
            layers.append(norm)
        layers.append(activation_fn())
        if dropout_rate is not None and dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        c = h

    if output_dim > 0:
        layers.append(nn.Linear(c, output_dim, bias=with_bias))
    if squash_output:
        layers.append(nn.Tanh())

    mlp = nn.Sequential(*layers)
    _apply_kernel_init(mlp, kernel_init)
    return mlp


class _MLPResNetBlock(nn.Module):
    """WSRL residual MLP block: dropout/norm -> Linear(4H) -> act -> Linear(H) -> +x."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        activation_fn: type[nn.Module] = nn.SiLU,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 32,
        dropout_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        norm = _make_norm(
            hidden_dim,
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
        )
        self.norm: nn.Module = norm if norm is not None else nn.Identity()
        self.dropout: nn.Module = (
            nn.Dropout(p=dropout_rate)
            if dropout_rate is not None and dropout_rate > 0
            else nn.Identity()
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act = activation_fn()
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.dropout(x)
        h = self.norm(h)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return residual + h


class MLPResNet(nn.Module):
    """Residual MLP matching the JAX WSRL reference.

    Structure: input projection -> N residual MLP blocks -> Swish -> output
    projection. Each block expands to ``4 * hidden_dim`` then projects back.
    Output projection is omitted when ``output_dim < 0`` (use as a trunk).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        activation_fn: type[nn.Module] = nn.SiLU,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 32,
        dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                _MLPResNetBlock(
                    hidden_dim,
                    activation_fn=activation_fn,
                    use_layer_norm=use_layer_norm,
                    use_group_norm=use_group_norm,
                    num_groups=num_groups,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_act = activation_fn()

        if output_dim > 0:
            self.output_proj: nn.Module = nn.Linear(hidden_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.output_proj = nn.Identity()
            self.output_dim = hidden_dim

        _apply_kernel_init(self, kernel_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_act(h)
        return self.output_proj(h)
