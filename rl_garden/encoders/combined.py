"""Combined extractor for Dict observations ({rgb[/depth], state, ...}).

Design
------
For each observation key we plug in one of:
  - image encoder (``PlainConv`` or a ResNet) when the key is an image,
  - proprio branch (Dense -> LayerNorm -> tanh) when the key is ``state``,
  - plain flatten for any other vector key.

By default, image keys are combined by channel-concatenation BEFORE the
encoder (matches ``EncoderObsWrapper`` in ManiSkill's sac_rgbd.py), so a
single encoder sees all image modalities stacked along channel dim.
Alternatively, ``fusion_mode="per_key"`` mirrors hil-serl's
``EncodingWrapper`` by encoding each image key independently and concatenating
the encoded features. The proprio branch follows hil-serl's design
(common/encoding.py L65-L69).

Inputs from ManiSkill's ``FlattenRGBDObservationWrapper`` are HWC uint8
tensors for images; we permute to NCHW and normalize to [0,1] here.
"""
from __future__ import annotations

from typing import Callable, Iterable, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.plain_conv import PlainConv

# A factory takes the stacked image ``spaces.Box`` (channels-first) and returns
# a ``BaseFeaturesExtractor``. This lets the caller swap PlainConv for a ResNet
# without changing the CombinedExtractor.
ImageEncoderFactory = Callable[[spaces.Box], BaseFeaturesExtractor]
ImageFusionMode = Literal["stack_channels", "per_key"]


def default_image_encoder_factory(features_dim: int = 256) -> ImageEncoderFactory:
    def _factory(img_space: spaces.Box) -> BaseFeaturesExtractor:
        _, h, w = img_space.shape
        return PlainConv(img_space, features_dim=features_dim, image_size=(h, w))

    return _factory


class ProprioEncoder(BaseFeaturesExtractor):
    """Proprio branch: Linear -> LayerNorm -> tanh. Port of hil-serl's block."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh(),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        image_keys: Iterable[str] = ("rgb", "depth"),
        state_key: str = "state",
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        proprio_latent_dim: int = 64,
        use_proprio: bool = True,
        fusion_mode: ImageFusionMode = "stack_channels",
        enable_stacking: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Dict)
        if fusion_mode not in ("stack_channels", "per_key"):
            raise ValueError(
                "fusion_mode must be either 'stack_channels' or 'per_key', "
                f"got {fusion_mode!r}"
            )

        # Determine which image keys are actually present.
        present_image_keys = [k for k in image_keys if k in observation_space.spaces]
        has_state = use_proprio and state_key in observation_space.spaces

        image_specs: dict[str, tuple[int, int, int]] = {}
        for k in present_image_keys:
            sp = observation_space.spaces[k]
            assert isinstance(sp, spaces.Box), f"image key {k!r} must be a Box"
            image_specs[k] = self._image_space_to_hwc(
                sp, image_key=k, enable_stacking=enable_stacking
            )

        features_dim = 0
        self._has_images = bool(image_specs)
        factory = image_encoder_factory or default_image_encoder_factory()
        image_encoder: Optional[BaseFeaturesExtractor] = None
        image_encoders: nn.ModuleDict = nn.ModuleDict()
        if self._has_images:
            if fusion_mode == "stack_channels":
                total_channels = 0
                image_hw: Optional[tuple[int, int]] = None
                for h, w, c in image_specs.values():
                    total_channels += c
                    image_hw = (h, w) if image_hw is None else image_hw
                    assert image_hw == (h, w), "all image keys must share the same H, W"
                assert image_hw is not None
                image_encoder = factory(
                    spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(total_channels, image_hw[0], image_hw[1]),
                        dtype=np.float32,
                    )
                )
                features_dim += image_encoder.features_dim
            else:
                for k, (h, w, c) in image_specs.items():
                    encoder = factory(
                        spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(c, h, w),
                            dtype=np.float32,
                        )
                    )
                    image_encoders[k] = encoder
                    features_dim += encoder.features_dim

        proprio: Optional[ProprioEncoder] = None
        if has_state:
            proprio = ProprioEncoder(
                observation_space.spaces[state_key], features_dim=proprio_latent_dim
            )
            features_dim += proprio.features_dim

        vector_extractors: nn.ModuleDict = nn.ModuleDict()
        image_key_set = set(present_image_keys)
        for key, subspace in observation_space.spaces.items():
            if key in image_key_set or key == state_key:
                continue
            assert isinstance(subspace, spaces.Box), f"vector key {key!r} must be a Box"
            vector_extractors[key] = nn.Flatten()
            features_dim += int(np.prod(subspace.shape))

        assert features_dim > 0, "CombinedExtractor produced 0-dim output."
        super().__init__(observation_space, features_dim)

        self.image_keys: tuple[str, ...] = tuple(present_image_keys)
        self.state_key = state_key
        self.has_state = has_state
        self.fusion_mode = fusion_mode
        self.enable_stacking = enable_stacking
        self.image_encoder = image_encoder
        self.image_encoders = image_encoders
        self.proprio = proprio
        self.vector_extractors = vector_extractors

    @staticmethod
    def _image_space_to_hwc(
        space: spaces.Box, image_key: str, enable_stacking: bool
    ) -> tuple[int, int, int]:
        if len(space.shape) == 3:
            h, w, c = space.shape
            return int(h), int(w), int(c)
        if enable_stacking and len(space.shape) == 4:
            t, h, w, c = space.shape
            return int(h), int(w), int(t * c)
        raise AssertionError(
            f"image key {image_key!r} must be a 3D Box (H, W, C)"
            + (
                " or 4D Box (T, H, W, C) when enable_stacking=True"
                if enable_stacking
                else ""
            )
            + f"; got {space.shape}"
        )

    def _prepare_image(self, key: str, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8 or key == "rgb":
            # uint8 HWC -> float32, normalized to [0, 1]
            x = x.float() / 255.0
        else:
            x = x.float()
        if self.enable_stacking and x.ndim == 5:
            # B,T,H,W,C -> B,H,W,(T*C), matching hil-serl's stacking behavior.
            b, t, h, w, c = x.shape
            x = x.permute(0, 2, 3, 1, 4).reshape(b, h, w, t * c)
        return x

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def _stack_images(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = [self._prepare_image(k, obs[k]) for k in self.image_keys]
        x = torch.cat(tensors, dim=-1)  # (B, H, W, Ctotal)
        return self._to_nchw(x)

    def _encode_images(
        self, obs: dict[str, torch.Tensor], stop_gradient: bool
    ) -> list[torch.Tensor]:
        if not self._has_images:
            return []
        if self.fusion_mode == "stack_channels":
            assert self.image_encoder is not None
            encoded = self.image_encoder(self._stack_images(obs))
            return [encoded.detach() if stop_gradient else encoded]

        encoded = []
        for key in self.image_keys:
            image = self._to_nchw(self._prepare_image(key, obs[key]))
            y = self.image_encoders[key](image)
            encoded.append(y.detach() if stop_gradient else y)
        return encoded

    def _encode_proprio(self, state: torch.Tensor) -> torch.Tensor:
        if self.enable_stacking and state.ndim > 2:
            state = state.flatten(1)
        assert self.proprio is not None
        return self.proprio(state)

    def extract(
        self, obs: dict[str, torch.Tensor], stop_gradient: bool = False
    ) -> torch.Tensor:
        # TODO: add an is_encoded fast path if replay buffers store image features.
        out = []
        out.extend(self._encode_images(obs, stop_gradient=stop_gradient))
        if self.has_state:
            out.append(self._encode_proprio(obs[self.state_key]))
        for key, extractor in self.vector_extractors.items():
            out.append(extractor(obs[key]))
        return torch.cat(out, dim=-1)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.extract(obs, stop_gradient=False)


def discover_image_keys(observation_space: spaces.Dict) -> tuple[str, ...]:
    """Return all keys in ``observation_space`` whose names start with ``rgb``
    or ``depth``.

    RGB keys come before depth keys; within each group, keys are sorted
    alphabetically for determinism. Used by per-camera training entrypoints
    (e.g. peg) to discover ``rgb_<cam>`` / ``depth_<cam>`` keys produced by
    :class:`PerCameraRGBDWrapper`.
    """
    if not isinstance(observation_space, spaces.Dict):
        raise TypeError(
            "discover_image_keys expects a Dict observation space, got "
            f"{type(observation_space).__name__}"
        )
    rgb_keys = sorted(k for k in observation_space.spaces if k.startswith("rgb"))
    depth_keys = sorted(k for k in observation_space.spaces if k.startswith("depth"))
    return tuple(rgb_keys + depth_keys)
