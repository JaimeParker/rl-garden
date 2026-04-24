"""Tests for the PyTorch ResNet encoder."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.encoders import (
    CombinedExtractor,
    ResNetEncoder,
    SpatialLearnedEmbeddings,
    resnet_encoder_factory,
)


def test_spatial_learned_embeddings_shape():
    m = SpatialLearnedEmbeddings(channels=8, height=4, width=4, num_features=3)
    y = m(torch.randn(2, 8, 4, 4))
    assert y.shape == (2, 8 * 3)


def test_resnet10_forward_and_grad():
    space = spaces.Box(0.0, 1.0, (3, 64, 64), np.float32)
    enc = ResNetEncoder(space, stage_sizes=(1, 1, 1, 1))
    assert enc.features_dim == 256
    x = torch.rand(2, 3, 64, 64, requires_grad=False)
    y = enc(x)
    assert y.shape == (2, 256)
    # Grad flows through the encoder.
    loss = y.pow(2).sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert has_grad


def test_resnet18_forward_shape():
    space = spaces.Box(0.0, 1.0, (3, 128, 128), np.float32)
    enc = ResNetEncoder(space, stage_sizes=(2, 2, 2, 2))
    y = enc(torch.rand(1, 3, 128, 128))
    assert y.shape == (1, 256)


def test_combined_extractor_with_resnet_factory():
    dict_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (64, 64, 3), np.uint8),
            "state": spaces.Box(-1.0, 1.0, (5,), np.float32),
        }
    )
    factory = resnet_encoder_factory("resnet10", features_dim=256)
    ce = CombinedExtractor(
        dict_space, image_keys=("rgb",), state_key="state", image_encoder_factory=factory
    )
    assert ce.features_dim == 256 + 64
    obs = {
        "rgb": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "state": torch.randn(2, 5),
    }
    out = ce(obs)
    assert out.shape == (2, 320)


def test_default_pooling_is_spatial_softmax():
    space = spaces.Box(0.0, 1.0, (3, 64, 64), np.float32)
    enc = ResNetEncoder(space)
    # spatial_softmax emits (2 * C) pooled dim which bottleneck maps to 256.
    assert enc.features_dim == 256
    from rl_garden.encoders.pooling import SpatialSoftmax

    assert isinstance(enc.pool, SpatialSoftmax)


def test_pretrained_weights_load(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_GARDEN_PRETRAINED_DIR", str(tmp_path))
    space = spaces.Box(0.0, 1.0, (3, 64, 64), np.float32)
    # Train a throwaway encoder, snapshot its state, load it into a fresh one.
    source = ResNetEncoder(space)
    torch.save(source.state_dict(), tmp_path / "my-weights.pt")

    target = ResNetEncoder(space, pretrained_weights="my-weights")
    for (k, v_s), (_k, v_t) in zip(
        source.state_dict().items(), target.state_dict().items()
    ):
        assert torch.allclose(v_s, v_t), f"mismatch at {k}"


def test_pretrained_weights_can_freeze_full_encoder(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_GARDEN_PRETRAINED_DIR", str(tmp_path))
    space = spaces.Box(0.0, 1.0, (3, 64, 64), np.float32)
    source = ResNetEncoder(space)
    torch.save(source.state_dict(), tmp_path / "frozen-weights.pt")

    target = ResNetEncoder(
        space,
        pretrained_weights="frozen-weights",
        freeze_resnet_encoder=True,
    )
    assert all(not p.requires_grad for p in target.parameters())


def test_pretrained_weights_can_freeze_only_backbone(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_GARDEN_PRETRAINED_DIR", str(tmp_path))
    space = spaces.Box(0.0, 1.0, (3, 64, 64), np.float32)
    source = ResNetEncoder(space)
    torch.save(source.state_dict(), tmp_path / "backbone-weights.pt")

    target = ResNetEncoder(
        space,
        pretrained_weights="backbone-weights",
        freeze_resnet_backbone=True,
    )
    backbone_modules = (target.stem_conv, target.stem_norm, target.blocks)
    assert all(not p.requires_grad for module in backbone_modules for p in module.parameters())
    assert all(p.requires_grad for p in target.pool.parameters())
    assert all(p.requires_grad for p in target.bottleneck.parameters())


def test_full_encoder_freeze_takes_precedence_over_backbone_freeze(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_GARDEN_PRETRAINED_DIR", str(tmp_path))
    space = spaces.Box(0.0, 1.0, (3, 64, 64), np.float32)
    source = ResNetEncoder(space)
    torch.save(source.state_dict(), tmp_path / "precedence-weights.pt")

    target = ResNetEncoder(
        space,
        pretrained_weights="precedence-weights",
        freeze_resnet_encoder=True,
        freeze_resnet_backbone=True,
    )
    assert all(not p.requires_grad for p in target.parameters())


def test_resnet_factory_propagates_freeze_options(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_GARDEN_PRETRAINED_DIR", str(tmp_path))
    img_space = spaces.Box(0.0, 1.0, (3, 64, 64), np.float32)
    source = ResNetEncoder(img_space)
    torch.save(source.state_dict(), tmp_path / "factory-weights.pt")

    factory = resnet_encoder_factory(
        "resnet10",
        features_dim=256,
        pretrained_weights="factory-weights",
        freeze_resnet_backbone=True,
    )
    encoder = factory(img_space)
    assert all(not p.requires_grad for p in encoder.blocks.parameters())
    assert all(p.requires_grad for p in encoder.bottleneck.parameters())


def test_pretrained_weights_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_GARDEN_PRETRAINED_DIR", str(tmp_path))
    space = spaces.Box(0.0, 1.0, (3, 64, 64), np.float32)
    with pytest.raises(FileNotFoundError):
        ResNetEncoder(space, pretrained_weights="does-not-exist")
