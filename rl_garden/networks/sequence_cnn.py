"""Stateless whole-window sequence <-> latent modules for A2A (``a2a_bc.py``).

Ports ``3rd_party/A2A_Flow_Matching``'s ``CNNActionEncoder``/``SimpleActionDecoder``
(``roboverse_learn/il/policies/a2a/action_ae.py``): a small 1D-CNN encoder that
maps a fixed-length window ``(B, seq_len, input_dim)`` to a single latent
vector, and a plain MLP decoder that maps a latent vector back to
``(B, horizon, output_dim)``. Used twice by ``A2APolicy`` with disjoint
parameters -- once over a state-history window (the flow's source), once over
an action-chunk window (the flow's target) -- hence ``CNNSequenceEncoder``
rather than a name tied to either role.

Unrelated to ``rl_garden.networks.sequence_encoder.SequenceLatentEncoder``: that
is a *stateful*, step-wise ``(latent, state) -> (latent, state)`` Protocol for
online recurrent rollouts (``RecurrentLatentEncoder``/``GTrXLLatentEncoder``).
``CNNSequenceEncoder`` is stateless and consumes a whole window in one forward
pass, only ever used against a pre-materialized offline dataset
(``load_h5_dataset_as_chunks``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from rl_garden.networks.mlp import Activation, KernelInit, create_mlp, resolve_activation


class CNNSequenceEncoder(nn.Module):
    """``(B, seq_len, input_dim) -> (B, latent_dim)`` via a stride-2 Conv1d stack.

    ``x`` must be in chronological order (oldest first, newest last) --
    matches ``load_h5_dataset_as_chunks``'s ``obs_history`` ordering directly,
    so no reordering is needed at the call site.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        latent_dim: int,
        *,
        num_layers: int = 3,
        hidden_channels: int = 512,
        kernel_size: int = 5,
        activation_fn: Optional[Activation] = "relu",
    ) -> None:
        super().__init__()
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}.")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}.")

        # padding=kernel_size//2 gives exactly ceil(L/2) per stride-2 layer
        # for any L>=1, and a length-1 sequence stays at 1 (never shrinks
        # below it) -- so final_len>=1 is guaranteed once seq_len>=1.
        final_len = seq_len
        for _ in range(num_layers):
            final_len = -(-final_len // 2)  # ceil(final_len / 2)
        self.input_dim = input_dim
        self.seq_len = seq_len
        self._final_len = final_len

        activation = resolve_activation(activation_fn, default=nn.ReLU)
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_channels = input_dim
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(in_channels, hidden_channels, kernel_size, stride=2, padding=padding)
            )
            layers.append(activation())
            in_channels = hidden_channels
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_channels * final_len, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (self.seq_len, self.input_dim), (
            f"expected (B, {self.seq_len}, {self.input_dim}), got {tuple(x.shape)}"
        )
        x = x.permute(0, 2, 1)  # (B, input_dim, seq_len)
        x = self.conv(x)  # (B, hidden_channels, final_len)
        x = x.flatten(1)
        return self.proj(x)


class ActionChunkDecoder(nn.Module):
    """``(B, latent_dim) -> (B, horizon, output_dim)`` via a plain stacked MLP.

    Ports the reference ``SimpleActionDecoder``'s ``input_proj -> [Mlp, ...] ->
    output_proj`` shape (no residual skip connections).
    """

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        output_dim: int,
        *,
        net_arch: Sequence[int] = (512, 512, 512, 512),
        activation_fn: Optional[Activation] = None,
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.net = create_mlp(
            input_dim=latent_dim,
            output_dim=horizon * output_dim,
            net_arch=list(net_arch),
            activation_fn=resolve_activation(activation_fn, default=nn.GELU),
            kernel_init=kernel_init,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(z.shape[0], self.horizon, self.output_dim)
