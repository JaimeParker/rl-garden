from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.common.cli_args import VisionArgs, image_encoder_factory_from_args
from rl_garden.encoders import CNN3DEncoder, DrQv2Encoder


def test_cnn3d_encoder_forward_shape() -> None:
    enc = CNN3DEncoder(
        spaces.Box(0.0, 1.0, (9, 64, 64), dtype=np.float32), num_frames=3, features_dim=32
    )
    out = enc(torch.rand(4, 9, 64, 64))

    assert out.shape == (4, 32)


def test_cnn3d_encoder_requires_at_least_two_frames() -> None:
    with pytest.raises(ValueError, match="frame_stack >= 2"):
        CNN3DEncoder(spaces.Box(0.0, 1.0, (3, 64, 64), dtype=np.float32), num_frames=1)


def test_cnn3d_encoder_requires_channels_divisible_by_num_frames() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CNN3DEncoder(spaces.Box(0.0, 1.0, (5, 64, 64), dtype=np.float32), num_frames=2)


def test_cnn3d_encoder_reshapes_folded_channels_into_time_axis() -> None:
    """CombinedExtractor folds (B,T,H,W,C) -> (B,H,W,T*C) via reshape after a
    (0,2,3,1,4) permute, i.e. channel index = t*C + c. This must round-trip
    exactly back to (B,C,T,H,W) for the 3D conv to see the right frame/channel
    at each position.
    """
    num_frames, channels_per_frame, h, w, batch = 3, 2, 8, 8, 2
    enc = CNN3DEncoder(
        spaces.Box(0.0, 1.0, (num_frames * channels_per_frame, h, w), dtype=np.float32),
        num_frames=num_frames,
        features_dim=4,
    )
    captured: dict[str, torch.Tensor] = {}
    enc.conv.register_forward_hook(
        lambda module, args, output: captured.__setitem__("input", args[0].detach().clone())
    )

    x = torch.empty(batch, num_frames * channels_per_frame, h, w)
    for t in range(num_frames):
        for c in range(channels_per_frame):
            x[:, t * channels_per_frame + c] = t * 10 + c

    enc(x)

    reshaped = captured["input"]
    assert reshaped.shape == (batch, channels_per_frame, num_frames, h, w)
    for t in range(num_frames):
        for c in range(channels_per_frame):
            assert torch.all(reshaped[:, c, t] == t * 10 + c)


def test_cnn3d_factory_from_args_builds_encoder_with_frame_stack() -> None:
    factory = image_encoder_factory_from_args(
        VisionArgs(encoder="cnn3d", frame_stack=3, encoder_features_dim=16)
    )
    enc = factory(spaces.Box(0.0, 1.0, (9, 64, 64), dtype=np.float32))

    assert isinstance(enc, CNN3DEncoder)
    assert enc.num_frames == 3
    assert enc.features_dim == 16


def test_cnn3d_factory_from_args_defers_frame_stack_validation_to_build() -> None:
    # image_encoder_factory_from_args itself must not raise for frame_stack=1:
    # test_image_encoder_factory_returns_callable_for_each_encoder (in
    # test_cli_args.py) calls it with default VisionArgs() (frame_stack=1) for
    # every registered encoder. Only constructing the encoder should raise.
    factory = image_encoder_factory_from_args(VisionArgs(encoder="cnn3d", frame_stack=1))

    with pytest.raises(ValueError, match="frame_stack >= 2"):
        factory(spaces.Box(0.0, 1.0, (3, 64, 64), dtype=np.float32))


def test_drqv2_conv_factory_from_args_builds_drqv2_encoder() -> None:
    factory = image_encoder_factory_from_args(VisionArgs(encoder="drqv2_conv"))
    enc = factory(spaces.Box(0, 255, (3, 84, 84), dtype=np.uint8))

    assert isinstance(enc, DrQv2Encoder)
