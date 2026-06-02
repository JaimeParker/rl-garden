"""Small ViT encoders ported from residual-offpolicy-rl.

The residual reference uses one ViT per camera/key, applies random shifts before
encoding training batches, and concatenates camera features along the patch
dimension. This module keeps that behavior available while fitting
rl-garden's ``BaseFeaturesExtractor`` conventions.
"""
from __future__ import annotations

from typing import Iterable, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.nn.init import trunc_normal_

from rl_garden.encoders.augment import RandomShiftsAug
from rl_garden.encoders.base import BaseFeaturesExtractor, TokenAndPropFeatureConfig

ViTFusionMode = Literal["per_key", "stack_channels"]
ViTAugmentationMode = Literal["random_shift", "none"]

_VIT_CACHE_KEY = "_vit_features"


def _init_weights_vit_timm(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class PatchEmbed2(nn.Module):
    """Conv patch embed matching residual-offpolicy-rl's tested embed2 path."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        use_norm: bool,
        image_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=8, stride=4),
            nn.GroupNorm(embed_dim, embed_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size[0], image_size[1])
            out = self.embed(dummy)
        self.grid_size = (int(out.shape[-2]), int(out.shape[-1]))
        self.num_patch = self.grid_size[0] * self.grid_size[1]
        self.patch_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embed(x)
        return y.flatten(2).transpose(1, 2).contiguous()


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, t, c = x.shape
        head_dim = c // self.num_heads
        qkv = self.qkv_proj(x).reshape(b, t, 3, self.num_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn_v = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, attn_mask=attn_mask
        )
        attn_v = attn_v.transpose(1, 2).reshape(b, t, c)
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.dropout(self.mha(self.layer_norm1(x), attn_mask))
        x = x + self.dropout(self.linear2(F.gelu(self.linear1(self.layer_norm2(x)))))
        return x


class MinVit(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: tuple[int, int],
        embed_dim: int = 128,
        embed_norm: bool = False,
        num_heads: int = 4,
        depth: int = 1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2(
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_norm=embed_norm,
            image_size=image_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patch, embed_dim)
        )
        self.net = nn.Sequential(
            *[
                TransformerLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = self.patch_embed.num_patch
        self.patch_dim = embed_dim

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_weights_vit_timm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.net(x)
        return self.norm(x)


class ViTImageEncoder(BaseFeaturesExtractor):
    """Image-only ViT encoder returning a flat feature vector."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        *,
        embed_dim: int = 128,
        depth: int = 1,
        num_heads: int = 4,
        embed_norm: bool = False,
        augmentation: ViTAugmentationMode = "none",
        random_shift_pad: int = 4,
    ) -> None:
        if len(observation_space.shape) != 3:
            raise ValueError(
                "ViTImageEncoder expects channels-first image space (C,H,W), "
                f"got {observation_space.shape}."
            )
        c, h, w = (int(v) for v in observation_space.shape)
        super().__init__(observation_space, features_dim)
        self.vit = MinVit(
            in_channels=c,
            image_size=(h, w),
            embed_dim=embed_dim,
            embed_norm=embed_norm,
            num_heads=num_heads,
            depth=depth,
        )
        self.patch_dim = self.vit.patch_dim
        self.num_patches = self.vit.num_patches
        self.repr_dim = self.patch_dim * self.num_patches
        self.augmentation = augmentation
        self.random_shift = RandomShiftsAug(random_shift_pad)
        self.proj = nn.Linear(self.repr_dim, features_dim)

    def encode_tokens(self, x: torch.Tensor, *, augment: bool = False) -> torch.Tensor:
        if augment and self.augmentation == "random_shift":
            x = self.random_shift(x)
        x = x - 0.5
        return self.vit(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(x, augment=self.training)
        return self.proj(tokens.flatten(1))


def vit_image_encoder_factory(
    *,
    features_dim: int = 256,
    embed_dim: int = 128,
    depth: int = 1,
    num_heads: int = 4,
    embed_norm: bool = False,
    augmentation: ViTAugmentationMode = "none",
    random_shift_pad: int = 4,
):
    def _factory(img_space: spaces.Box) -> BaseFeaturesExtractor:
        return ViTImageEncoder(
            img_space,
            features_dim=features_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            embed_norm=embed_norm,
            augmentation=augmentation,
            random_shift_pad=random_shift_pad,
        )

    return _factory


class ViTTokenAndPropExtractor(BaseFeaturesExtractor):
    """Dict extractor for SAC-family token-aware ViT policies.

    The returned tensor layout is ``[tokens.flatten(1), prop]``.
    ``structured_feature_config()`` exposes this layout so that ``SACPolicy``
    can self-configure a spatial Q-critic and actor adapter without any
    isinstance checks.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        image_keys: Iterable[str] = ("rgb", "depth"),
        state_key: str = "state",
        use_proprio: bool = True,
        fusion_mode: ViTFusionMode = "per_key",
        enable_stacking: bool = False,
        embed_dim: int = 128,
        depth: int = 1,
        num_heads: int = 4,
        embed_norm: bool = False,
        augmentation: ViTAugmentationMode = "random_shift",
        random_shift_pad: int = 4,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("ViTTokenAndPropExtractor requires a Dict observation space.")
        if fusion_mode not in ("per_key", "stack_channels"):
            raise ValueError(
                f"fusion_mode must be 'per_key' or 'stack_channels', got {fusion_mode!r}."
            )

        self.image_keys = tuple(k for k in image_keys if k in observation_space.spaces)
        if not self.image_keys:
            raise ValueError("ViTTokenAndPropExtractor requires at least one image key.")
        self.state_key = state_key
        self.has_state = use_proprio and state_key in observation_space.spaces
        self.fusion_mode = fusion_mode
        self.enable_stacking = enable_stacking
        self.augmentation = augmentation

        specs = {
            k: self._image_space_to_hwc(
                observation_space.spaces[k], image_key=k, enable_stacking=enable_stacking
            )
            for k in self.image_keys
        }

        if fusion_mode == "stack_channels":
            image_hw: Optional[tuple[int, int]] = None
            channels = 0
            for h, w, c in specs.values():
                image_hw = (h, w) if image_hw is None else image_hw
                if image_hw != (h, w):
                    raise ValueError("stack_channels fusion requires matching H,W.")
                channels += c
            assert image_hw is not None
            num_patches = self._num_patches_for_hw(image_hw)
            patch_dim = embed_dim
        else:
            num_patches = 0
            patch_dim = embed_dim
            for h, w, _c in specs.values():
                num_patches += self._num_patches_for_hw((h, w))

        prop_dim = 0
        if self.has_state:
            prop_dim += int(np.prod(observation_space.spaces[state_key].shape))
        image_key_set = set(self.image_keys)
        self.vector_keys: tuple[str, ...] = tuple(
            key
            for key, subspace in observation_space.spaces.items()
            if key not in image_key_set and key != state_key and isinstance(subspace, spaces.Box)
        )
        for key in self.vector_keys:
            prop_dim += int(np.prod(observation_space.spaces[key].shape))

        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.token_dim = num_patches * patch_dim
        self.prop_dim = prop_dim
        super().__init__(observation_space, features_dim=self.token_dim + self.prop_dim)

        self.random_shift = RandomShiftsAug(random_shift_pad)
        self.image_encoder: Optional[MinVit] = None
        self.image_encoders = nn.ModuleDict()
        if fusion_mode == "stack_channels":
            assert image_hw is not None
            self.image_encoder = MinVit(
                in_channels=channels,
                image_size=image_hw,
                embed_dim=embed_dim,
                embed_norm=embed_norm,
                num_heads=num_heads,
                depth=depth,
            )
        else:
            for key, (h, w, c) in specs.items():
                self.image_encoders[key] = MinVit(
                    in_channels=c,
                    image_size=(h, w),
                    embed_dim=embed_dim,
                    embed_norm=embed_norm,
                    num_heads=num_heads,
                    depth=depth,
                )

    # --- BaseFeaturesExtractor overrides ---

    def structured_feature_config(self) -> TokenAndPropFeatureConfig:
        return TokenAndPropFeatureConfig(
            layout="token_and_prop",
            num_patches=self.num_patches,
            patch_dim=self.patch_dim,
            prop_dim=self.prop_dim,
        )

    def prepare_batch(
        self,
        obs: dict,
        next_obs: Optional[dict] = None,
    ) -> None:
        obs[_VIT_CACHE_KEY] = self.encode(obs, augment=True)
        if next_obs is not None:
            with torch.no_grad():
                # Preserve PR #21 behavior: cache augmented target observations too.
                next_obs[_VIT_CACHE_KEY] = self.encode(next_obs, augment=True)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if _VIT_CACHE_KEY in obs:
            return obs[_VIT_CACHE_KEY]
        return self.encode(obs, augment=False)

    # --- internal helpers ---

    @staticmethod
    def _num_patches_for_hw(image_size: tuple[int, int]) -> int:
        h, w = image_size
        h1 = (h - 8) // 4 + 1
        w1 = (w - 8) // 4 + 1
        h2 = (h1 - 3) // 2 + 1
        w2 = (w1 - 3) // 2 + 1
        if h2 <= 0 or w2 <= 0:
            raise ValueError(f"ViT image size is too small for PatchEmbed2: {image_size}.")
        return h2 * w2

    @staticmethod
    def _image_space_to_hwc(
        space: spaces.Space, image_key: str, enable_stacking: bool
    ) -> tuple[int, int, int]:
        if not isinstance(space, spaces.Box):
            raise TypeError(f"image key {image_key!r} must be a Box.")
        if len(space.shape) == 3:
            h, w, c = space.shape
            return int(h), int(w), int(c)
        if enable_stacking and len(space.shape) == 4:
            t, h, w, c = space.shape
            return int(h), int(w), int(t * c)
        raise ValueError(
            f"image key {image_key!r} must be 3D HWC"
            + (" or 4D THWC with stacking enabled" if enable_stacking else "")
            + f"; got {space.shape}."
        )

    def _prepare_image(self, key: str, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        if self.enable_stacking and x.ndim == 5:
            b, t, h, w, c = x.shape
            x = x.permute(0, 2, 3, 1, 4).reshape(b, h, w, t * c)
        return x.permute(0, 3, 1, 2).contiguous()

    def _stack_images(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = [self._prepare_image(k, obs[k]) for k in self.image_keys]
        return torch.cat(tensors, dim=1)

    def _encode_tokens(
        self, obs: dict[str, torch.Tensor], *, augment: bool
    ) -> torch.Tensor:
        if self.fusion_mode == "stack_channels":
            assert self.image_encoder is not None
            image = self._stack_images(obs)
            if augment and self.augmentation == "random_shift":
                image = self.random_shift(image)
            return self.image_encoder(image - 0.5)

        tokens = []
        for key in self.image_keys:
            image = self._prepare_image(key, obs[key])
            if augment and self.augmentation == "random_shift":
                image = self.random_shift(image)
            tokens.append(self.image_encoders[key](image - 0.5))
        return torch.cat(tokens, dim=1)

    def _encode_prop(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        props = []
        if self.has_state:
            state = obs[self.state_key]
            props.append(state.float().flatten(1))
        for key in self.vector_keys:
            props.append(obs[key].float().flatten(1))
        if not props:
            batch = next(iter(obs.values())).shape[0]
            device = next(iter(obs.values())).device
            return torch.empty(batch, 0, device=device)
        return torch.cat(props, dim=-1)

    def encode(
        self,
        obs: dict[str, torch.Tensor],
        *,
        augment: bool = False,
    ) -> torch.Tensor:
        tokens = self._encode_tokens(obs, augment=augment)
        return torch.cat([tokens.flatten(1), self._encode_prop(obs)], dim=-1)
