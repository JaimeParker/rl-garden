"""``EncoderConfig``: "how observations are encoded", independent of "what is
observed" (``rl_garden.observations.ObservationConfig``/``ObservationSchema``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from rl_garden.encoders.combined import ImageEncoderFactory
from rl_garden.encoders.registry import ENCODER_REGISTRY

if TYPE_CHECKING:
    from rl_garden.observations import ObservationSchema


@dataclass
class EncoderConfig:
    backbone: Literal[
        "plain_conv", "resnet10", "resnet18", "vit", "drqv2_conv", "cnn3d"
    ] = "plain_conv"
    features_dim: int = 256
    image_augmentation: Literal["none", "random_shift"] = "none"
    image_random_shift_pad: int = 4
    image_fusion_mode: Literal["stack_channels", "per_key"] = "stack_channels"
    proprio_latent_dim: int = 64
    normalize_obs: bool = False
    vit_fusion_mode: Literal["per_key", "stack_channels"] = "per_key"
    vit_embed_dim: int = 128
    vit_depth: int = 1
    vit_num_heads: int = 4
    vit_embed_norm: bool = False
    vit_augmentation: Literal["random_shift", "none"] = "random_shift"
    vit_random_shift_pad: int = 4
    vit_actor_feature_dim: int = 128
    vit_critic_spatial_emb_dim: int = 1024
    plain_conv_weight_init: Literal["kaiming_uniform", "orthogonal"] = "kaiming_uniform"
    plain_conv_last_act: bool = True
    plain_conv_pooling: Literal["flatten", "gap", "adaptive_max"] = "flatten"
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False
    # Matches rl_garden.encoders.resnet.PoolingMethod's values -- not imported
    # directly to keep this module's top-level imports light.
    pooling_method: Literal["spatial_learned_embeddings", "spatial_softmax", "avg"] = "spatial_softmax"

    def _resolve_spec(self):
        try:
            return ENCODER_REGISTRY[self.backbone]
        except KeyError:
            raise ValueError(
                f"Unknown encoder {self.backbone!r}. Known: {sorted(ENCODER_REGISTRY)}."
            )

    def image_encoder_factory(self) -> ImageEncoderFactory:
        """Return the flat image-encoder factory for ``self.backbone``.

        Also enforces that resnet-only options (``pretrained_weights`` /
        ``freeze_resnet_*``) are not set for non-resnet encoders, and that
        plain_conv-only options are not set for other encoders.
        """
        spec = self._resolve_spec()
        if not spec.allows_resnet_weights and (
            self.pretrained_weights is not None
            or self.freeze_resnet_encoder
            or self.freeze_resnet_backbone
        ):
            raise ValueError(
                "pretrained_weights, freeze_resnet_encoder, and "
                "freeze_resnet_backbone are only supported for resnet encoders."
            )
        if self.backbone != "plain_conv" and (
            self.plain_conv_weight_init != "kaiming_uniform"
            or self.plain_conv_last_act is not True
            or self.plain_conv_pooling != "flatten"
        ):
            raise ValueError(
                "plain_conv_weight_init, plain_conv_last_act, and "
                "plain_conv_pooling are only supported for the plain_conv "
                "encoder."
            )
        return spec.build_factory(self)

    def sac_kwargs(self, schema: "ObservationSchema") -> dict:
        """Structured-path kwargs for SAC-family constructors (currently only
        the ``vit`` backbone installs a structured extractor); ``{}`` for
        every other backbone.
        """
        return self._resolve_spec().build_sac_kwargs(self, schema)
