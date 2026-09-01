from __future__ import annotations

import pytest
import torch

from rl_garden.networks import BiGRUSequenceEncoder


def test_output_shape():
    enc = BiGRUSequenceEncoder(input_dim=5, seq_len=4, latent_dim=16, hidden_size=8, num_layers=2)
    x = torch.randn(3, 4, 5)
    out = enc(x)
    assert out.shape == (3, 16)


def test_wrong_seq_len_raises():
    enc = BiGRUSequenceEncoder(input_dim=5, seq_len=4, latent_dim=16, hidden_size=8)
    x = torch.randn(3, 5, 5)
    with pytest.raises(AssertionError):
        enc(x)


def test_wrong_input_dim_raises():
    enc = BiGRUSequenceEncoder(input_dim=5, seq_len=4, latent_dim=16, hidden_size=8)
    x = torch.randn(3, 4, 6)
    with pytest.raises(AssertionError):
        enc(x)
