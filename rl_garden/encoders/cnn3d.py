"""3D CNN (spatiotemporal) image encoder.

Runs ``Conv3d`` directly over the frame-stacked time axis instead of folding
stacked frames into the channel axis and running a 2D CNN over them (the
``plain_conv``/``resnet``/``vit`` approach).

Downsampling schedule ported from FAIR's X3D design (not the X3D code
itself -- X3D's own stage/channel widths target Kinetics-scale clips):
``3rd_party/pytorchvideo/pytorchvideo/models/x3d.py``,
``create_x3d_stem()``/``create_x3d_bottleneck_block()`` -- spatial-only
stride ``(1, 2, 2)`` with same-padding on every axis (``padding=[k // 2 for
k in kernel_size]``), so the temporal dimension is left untouched by every
conv and only collapsed by the final pool. Channel widths and layer count
are hand-picked for RL's short frame-stack windows (``frame_stack`` ~2-4)
and small resolutions (64-128px) instead of X3D's Kinetics-scale clips
(224px, 8-16+ frames).

Input: the same ``(B, T*C, H, W)`` tensor ``CombinedExtractor`` already
produces for every other registry encoder -- frame stacking folded into the
channel axis, already ``[0, 1]``-normalized. This encoder reshapes it back
into ``(B, C, T, H, W)`` before convolving, using ``num_frames`` supplied at
construction time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import create_mlp


class CNN3DEncoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        num_frames: int,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)
        if num_frames < 2:
            raise ValueError(
                "cnn3d encoder requires frame_stack >= 2 to have a time axis "
                f"to convolve over (got frame_stack={num_frames}); use a 2D "
                "encoder for single-frame observations."
            )
        total_channels = int(observation_space.shape[0])
        if total_channels % num_frames != 0:
            raise ValueError(
                f"Observation channel count {total_channels} is not "
                f"divisible by num_frames={num_frames}."
            )
        self.num_frames = num_frames
        self.channels_per_frame = total_channels // num_frames

        # Stride/padding schedule ported from X3D's create_x3d_stem() /
        # create_x3d_bottleneck_block() (see module docstring): stride only
        # on (H, W), same-padding on all three axes so T is never reduced by
        # a conv -- only by self.pool at the end.
        self.conv = nn.Sequential(
            nn.Conv3d(self.channels_per_frame, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        fc = create_mlp(input_dim=64, output_dim=features_dim, net_arch=[], squash_output=False)
        self.fc = nn.Sequential(fc, nn.ReLU())

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        x = image.view(b, self.num_frames, self.channels_per_frame, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T*C, H, W) -> (B, C, T, H, W)
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def cnn3d_encoder_factory(num_frames: int, features_dim: int = 256):
    """Return a factory that creates ``CNN3DEncoder`` instances.

    Usable as ``image_encoder_factory`` in ``CombinedExtractor`` so the
    entire image pathway uses spatiotemporal 3D convolution:

        CombinedExtractor(..., image_encoder_factory=cnn3d_encoder_factory(num_frames=3), ...)
    """

    def _factory(img_space: spaces.Box) -> CNN3DEncoder:
        return CNN3DEncoder(img_space, num_frames=num_frames, features_dim=features_dim)

    return _factory
