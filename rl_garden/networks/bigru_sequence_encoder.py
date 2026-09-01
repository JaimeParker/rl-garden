"""Stateless whole-window bidirectional-GRU sequence encoder.

Ports SUPE's OPAL hand-rolled ``SimpleBiGRU`` stack
(``SUPE/supe/pretraining/opal.py:32-68,71-102``) onto PyTorch's
built-in ``nn.GRU(bidirectional=True)`` -- confirmed equivalent: each of the
two stacked layers concatenates (not sums) forward/backward hidden state at
every timestep, and the second layer's input width is the first layer's
full ``2*hidden_size`` output, exactly matching ``nn.GRU``'s own multi-layer
bidirectional semantics.

Same ``(B, seq_len, input_dim) -> (B, latent_dim)`` contract as
``CNNSequenceEncoder`` (``sequence_cnn.py``) -- a drop-in sibling in the
sequence-encoder family, not an OPAL-only class -- but pools by flattening
the *entire* per-timestep output through one projection (matching OPAL's
``recur_output="concat"`` pooling, the only value the reference ever uses),
not by taking the last hidden state.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiGRUSequenceEncoder(nn.Module):
    """``(B, seq_len, input_dim) -> (B, latent_dim)`` via a stacked
    bidirectional GRU with whole-window-flatten pooling."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        latent_dim: int,
        *,
        hidden_size: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}.")
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Linear(seq_len * 2 * hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1:] == (self.seq_len, self.input_dim), (
            f"expected (B, {self.seq_len}, {self.input_dim}), got {tuple(x.shape)}"
        )
        out, _ = self.gru(x)  # (B, seq_len, 2*hidden_size)
        out = out.reshape(out.shape[0], -1)  # flatten whole window, not last-hidden-state
        return self.proj(out)
