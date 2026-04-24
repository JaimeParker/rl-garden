"""Spatial pooling heads for image encoders.

Ported from ``3rd_party/hil-serl/.../vision/resnet_v1.py`` (Flax) and
``vision/spatial.py``. The PyTorch versions are numerically equivalent to
the Flax originals up to channel-order and tensor layout (NCHW vs NHWC).

Implemented:
  - ``SpatialLearnedEmbeddings`` — per-spatial-location learned weights,
    summed over (H, W) to yield ``(B, C * num_features)``. Followed here
    by an optional bottleneck (Linear -> LayerNorm -> tanh) matching the
    ``bottleneck_dim`` branch of ``ResNetEncoder``.
  - ``SpatialSoftmax`` — soft-argmax that returns ``(B, 2 * C)`` expected
    (x, y) coordinates per channel (kept simple; FiLM-style temperature).
  - ``AvgPool`` — thin wrapper around ``x.mean((-2,-1))`` for parity.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AvgPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NCHW -> NC
        return x.mean(dim=(-2, -1))


class SpatialLearnedEmbeddings(nn.Module):
    """Port of hil-serl ``SpatialLearnedEmbeddings`` (resnet_v1.py L81).

    Flax (NHWC): ``kernel[h, w, c, n]``, output ``sum_hw(x[h,w,c] * kernel[h,w,c,n])``.
    PyTorch (NCHW): we store ``kernel[c, n, h, w]`` and contract over (h, w).
    Output shape: ``(B, C * num_features)``.
    """

    def __init__(self, channels: int, height: int, width: int, num_features: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.num_features = num_features
        # lecun_normal in Flax ~ kaiming with fan_in and scale 1.
        kernel = torch.empty(channels, num_features, height, width)
        nn.init.kaiming_normal_(kernel, a=0.0, mode="fan_in", nonlinearity="linear")
        self.kernel = nn.Parameter(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W). Broadcast multiply against kernel (C, N, H, W), sum over H,W.
        # einsum keeps it explicit.
        b = x.shape[0]
        y = torch.einsum("bchw,cnhw->bcn", x, self.kernel)
        return y.reshape(b, self.channels * self.num_features)


class SpatialSoftmax(nn.Module):
    """Soft-argmax pooling: ``(B, C, H, W) -> (B, 2*C)`` expected (x, y)."""

    def __init__(
        self, channels: int, height: int, width: int, temperature: float = 1.0
    ) -> None:
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))  # (H*W,)
        self.register_buffer("pos_y", pos_y.reshape(-1))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w) / self.temperature
        attn = torch.softmax(flat, dim=-1)
        ex = (attn * self.pos_x).sum(-1)  # (B, C)
        ey = (attn * self.pos_y).sum(-1)
        return torch.cat([ex, ey], dim=-1)
