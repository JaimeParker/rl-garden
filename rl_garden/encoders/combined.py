"""Combined extractor for Dict observations
({state, state_<name>, rgb_<cam>, depth_<cam>}).

Design
------
For each observation key we plug in one of:
  - image encoder (``PlainConv`` or a ResNet) when the key is an image,
  - proprio branch (Dense -> LayerNorm -> tanh) when the key is a state key
    (``"state"`` and/or any ``state_<name>`` keys, concatenated into one
    vector -- in ``schema.state_keys`` order -- before the branch),

By default, image keys are combined by channel-concatenation BEFORE the
encoder (matches ``EncoderObsWrapper`` in ManiSkill's sac_rgbd.py), so a
single encoder sees all image modalities stacked along channel dim.
Alternatively, ``fusion_mode="per_key"`` mirrors hil-serl's
``EncodingWrapper`` by encoding each image key independently and concatenating
the encoded features. The proprio branch follows hil-serl's design
(common/encoding.py L65-L69).

Inputs from ManiSkill's ``FlattenRGBDObservationWrapper`` are HWC uint8
tensors for images; we permute to NCHW and normalize to [0,1] here.

Which keys get an image/proprio branch is driven entirely by ``schema``
(an ``rl_garden.observations.ObservationSchema``): its ``image_keys`` and
``has_state``. Any key present in ``observation_space`` that ``schema``
does not include (e.g. dropped by an asymmetric ``ObsGroups`` actor/critic
split) is never looked up and so never contributes to the output, whatever
its name or shape.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.obs_normalization import RunningObsNormalizer
from rl_garden.encoders.base import BaseFeaturesExtractor, image_needs_normalization
from rl_garden.encoders.augment import RandomShiftsAug
from rl_garden.encoders.plain_conv import PlainConv
from rl_garden.observations import ObservationSchema

if TYPE_CHECKING:
    from rl_garden.encoders.config import EncoderConfig

# A factory takes the stacked image ``spaces.Box`` (channels-first) and returns
# a ``BaseFeaturesExtractor``. This lets the caller swap PlainConv for a ResNet
# without changing the CombinedExtractor.
ImageEncoderFactory = Callable[[spaces.Box], BaseFeaturesExtractor]
ImageFusionMode = Literal["stack_channels", "per_key"]
ImageAugmentationMode = Literal["none", "random_shift"]

_AUG_STACK_KEY = "_rl_garden_aug_stack_image"
_AUG_IMAGES_KEY = "_rl_garden_aug_images"
_AUG_SEED_SALT = 1_000_003


def default_image_encoder_factory(
    features_dim: int = 256,
    plain_conv_last_act: bool = True,
    plain_conv_weight_init: Literal["kaiming_uniform", "orthogonal"] = "kaiming_uniform",
    plain_conv_pooling: Literal["flatten", "gap", "adaptive_max"] = "flatten",
) -> ImageEncoderFactory:
    def _factory(img_space: spaces.Box) -> BaseFeaturesExtractor:
        _, h, w = img_space.shape
        return PlainConv(
            img_space,
            features_dim=features_dim,
            image_size=(h, w),
            pooling=plain_conv_pooling,
            last_act=plain_conv_last_act,
            weight_init=plain_conv_weight_init,
        )

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
    """Dict-observation feature extractor.

    ``schema`` (an ``rl_garden.observations.ObservationSchema``) says *what*
    is observed: image keys, state presence, and per-key stacking. ``encoder_config``
    (an ``rl_garden.encoders.EncoderConfig``) says *how* to encode it: backbone,
    fusion mode, augmentation, proprio dim, ...
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        schema: ObservationSchema,
        encoder_config: "EncoderConfig",
        *,
        augmentation_seed: Optional[int] = None,
    ) -> None:
        assert isinstance(observation_space, spaces.Dict)
        fusion_mode = encoder_config.image_fusion_mode
        image_augmentation = encoder_config.image_augmentation
        if fusion_mode not in ("stack_channels", "per_key"):
            raise ValueError(
                "fusion_mode must be either 'stack_channels' or 'per_key', "
                f"got {fusion_mode!r}"
            )
        if image_augmentation not in ("none", "random_shift"):
            raise ValueError(
                "image_augmentation must be either 'none' or 'random_shift', "
                f"got {image_augmentation!r}"
            )

        state_keys = schema.state_keys
        image_keys = schema.image_keys
        has_state = schema.has_state
        enable_stacking = any(schema.entries[k].stacked for k in image_keys)
        num_frames = 1
        if enable_stacking:
            frame_counts = {
                schema.entries[k].shape[0] for k in image_keys if schema.entries[k].stacked
            }
            assert len(frame_counts) == 1, (
                "all stacked image keys must share one frame count, got "
                f"{frame_counts} across {image_keys!r}"
            )
            num_frames = frame_counts.pop()

        image_specs: dict[str, tuple[int, int, int]] = {}
        for k in image_keys:
            sp = observation_space.spaces[k]
            assert isinstance(sp, spaces.Box), f"image key {k!r} must be a Box"
            image_specs[k] = self._image_space_to_hwc(
                sp, image_key=k, enable_stacking=enable_stacking
            )

        features_dim = 0
        self._has_images = bool(image_specs)
        image_encoder: Optional[BaseFeaturesExtractor] = None
        image_encoders: nn.ModuleDict = nn.ModuleDict()
        if self._has_images:
            if encoder_config.backbone == "cnn3d":
                # num_frames is a schema (Layer A) concern -- derived above
                # from the schema's stacked entries, not an EncoderConfig
                # field -- so cnn3d is built directly here instead of via
                # encoder_config.image_encoder_factory().
                from rl_garden.encoders.cnn3d import cnn3d_encoder_factory

                factory = cnn3d_encoder_factory(
                    num_frames=num_frames, features_dim=encoder_config.features_dim
                )
            else:
                factory = encoder_config.image_encoder_factory()

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
        state_dim = 0
        if has_state:
            state_dim = sum(
                int(np.prod(observation_space.spaces[k].shape)) for k in state_keys
            )
            proprio_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
            )
            proprio = ProprioEncoder(
                proprio_space, features_dim=encoder_config.proprio_latent_dim
            )
            features_dim += proprio.features_dim

        assert features_dim > 0, (
            "CombinedExtractor produced 0-dim output: schema has no image "
            f"keys and no state keys ({schema.keys!r})."
        )
        super().__init__(observation_space, features_dim)

        self.image_keys: tuple[str, ...] = tuple(image_keys)
        self._needs_norm: frozenset[str] = frozenset(
            k for k in self.image_keys
            if image_needs_normalization(observation_space.spaces[k])
        )
        self.state_keys: tuple[str, ...] = state_keys
        self.has_state = has_state
        self.fusion_mode = fusion_mode
        self.enable_stacking = enable_stacking
        self.image_encoder = image_encoder
        self.image_encoders = image_encoders
        self.proprio = proprio
        # No genuine vector key can exist under the strict observation
        # schema (state / rgb_<cam> / depth_<cam> only), so this is always
        # empty; kept so extract()/update_normalizer() stay simple no-op
        # loops and callers checking `key in extractor.vector_extractors`
        # keep working.
        self.vector_extractors: nn.ModuleDict = nn.ModuleDict()
        self._obs_normalizers: nn.ModuleDict = nn.ModuleDict()
        if encoder_config.normalize_obs and has_state:
            self._obs_normalizers["state"] = RunningObsNormalizer(state_dim)
        self.image_augmentation = image_augmentation
        self.random_shift_pad = encoder_config.image_random_shift_pad
        self.random_shift = (
            RandomShiftsAug(self.random_shift_pad)
            if image_augmentation == "random_shift" and self._has_images
            else None
        )
        self._augmentation_seed = None
        if self.random_shift is not None:
            self._augmentation_seed = (
                int(augmentation_seed)
                if augmentation_seed is not None
                else int(torch.initial_seed()) + _AUG_SEED_SALT
            )
        self._augmentation_generators: dict[str, torch.Generator] = {}
        # Instance-scoped (not the bare module constants): two distinct
        # CombinedExtractor instances (e.g. actor's + critic's own, under a
        # heterogeneous-encoder policy) may both call prepare_batch() on the
        # very same obs/next_obs dict -- a shared key would let the second
        # call silently overwrite the first's cached augmented images.
        self._aug_stack_key = f"{_AUG_STACK_KEY}_{id(self)}"
        self._aug_images_key = f"{_AUG_IMAGES_KEY}_{id(self)}"

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
        if key in self._needs_norm:
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

    def _stack_images_uncached(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = [self._prepare_image(k, obs[k]) for k in self.image_keys]
        x = torch.cat(tensors, dim=-1)  # (B, H, W, Ctotal)
        return self._to_nchw(x)

    def _stack_images(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        cached = obs.get(self._aug_stack_key)
        if cached is not None:
            return cached
        return self._stack_images_uncached(obs)

    def _image_for_key(self, obs: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        cached = obs.get(self._aug_images_key)
        if cached is not None and key in cached:
            return cached[key]
        return self._to_nchw(self._prepare_image(key, obs[key]))

    def _augmentation_generator(self, device: torch.device) -> torch.Generator:
        device_key = str(device)
        generator = self._augmentation_generators.get(device_key)
        if generator is None:
            assert self._augmentation_seed is not None
            generator = torch.Generator(device=device)
            generator.manual_seed(self._augmentation_seed)
            self._augmentation_generators[device_key] = generator
        return generator

    def _augment_image(self, image: torch.Tensor) -> torch.Tensor:
        assert self.random_shift is not None
        return self.random_shift(
            image,
            generator=self._augmentation_generator(image.device),
        )

    def prepare_batch(
        self,
        obs: dict,
        next_obs: Optional[dict] = None,
    ) -> None:
        if self.random_shift is None or not self._has_images:
            return
        self._prepare_augmented_images(obs)
        if next_obs is not None:
            with torch.no_grad():
                self._prepare_augmented_images(next_obs)

    def _prepare_augmented_images(self, obs: dict) -> None:
        if self.fusion_mode == "stack_channels":
            obs[self._aug_stack_key] = self._augment_image(self._stack_images_uncached(obs))
            return

        obs[self._aug_images_key] = {
            key: self._augment_image(self._to_nchw(self._prepare_image(key, obs[key])))
            for key in self.image_keys
        }

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
            image = self._image_for_key(obs, key)
            y = self.image_encoders[key](image)
            encoded.append(y.detach() if stop_gradient else y)
        return encoded

    def _concat_state(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for key in self.state_keys:
            part = obs[key]
            if self.enable_stacking and part.ndim > 2:
                part = part.flatten(1)
            parts.append(part)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _encode_proprio(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = self._concat_state(obs)
        if "state" in self._obs_normalizers:
            state = self._obs_normalizers["state"](state)
        assert self.proprio is not None
        return self.proprio(state)

    def extract(
        self, obs: dict[str, torch.Tensor], stop_gradient: bool = False
    ) -> torch.Tensor:
        # TODO: add an is_encoded fast path if replay buffers store image features.
        out = []
        out.extend(self._encode_images(obs, stop_gradient=stop_gradient))
        if self.has_state:
            out.append(self._encode_proprio(obs))
        for key, extractor in self.vector_extractors.items():
            flat = extractor(obs[key])
            if key in self._obs_normalizers:
                flat = self._obs_normalizers[key](flat)
            out.append(flat)
        return torch.cat(out, dim=-1)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.extract(obs, stop_gradient=False)

    def update_normalizer(self, obs: dict[str, torch.Tensor]) -> None:
        if not self._obs_normalizers:
            return
        if self.has_state and "state" in self._obs_normalizers:
            self._obs_normalizers["state"].update(self._concat_state(obs))
        for key, extractor in self.vector_extractors.items():
            if key in self._obs_normalizers:
                self._obs_normalizers[key].update(extractor(obs[key]))
