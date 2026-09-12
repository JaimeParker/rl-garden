"""Single source of truth for image encoders.

Moved out of ``rl_garden/common/cli_args.py`` so ``rl_garden/encoders`` does not
depend on the CLI-args layer.

Each per-encoder factory helper below takes an ``EncoderConfig`` (kept as
``Any`` to avoid importing it here and creating a cycle with
``rl_garden/encoders/config.py``, which imports this module).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def _plain_conv_factory(config: Any):
    from rl_garden.encoders import default_image_encoder_factory

    return default_image_encoder_factory(
        features_dim=config.features_dim,
        plain_conv_last_act=config.plain_conv_last_act,
        plain_conv_weight_init=config.plain_conv_weight_init,
        plain_conv_pooling=config.plain_conv_pooling,
    )


def _resnet_factory(config: Any):
    from rl_garden.encoders import resnet_encoder_factory

    return resnet_encoder_factory(
        name=config.backbone,
        features_dim=config.features_dim,
        pretrained_weights=config.pretrained_weights,
        freeze_resnet_encoder=config.freeze_resnet_encoder,
        freeze_resnet_backbone=config.freeze_resnet_backbone,
        pooling_method=config.pooling_method,
    )


def _vit_factory(config: Any):
    # Image-only flat ViT factory used by generic CombinedExtractor paths (e.g.
    # PPO). SAC-family structured ViT instead installs ViTTokenAndPropExtractor
    # via EncoderConfig.sac_kwargs(), which overrides the whole extractor.
    from rl_garden.encoders import vit_image_encoder_factory

    return vit_image_encoder_factory(
        features_dim=config.features_dim,
        embed_dim=config.vit_embed_dim,
        depth=config.vit_depth,
        num_heads=config.vit_num_heads,
        embed_norm=config.vit_embed_norm,
        augmentation=config.vit_augmentation,
        random_shift_pad=config.vit_random_shift_pad,
    )


def _drqv2_conv_factory(config: Any):
    from rl_garden.encoders import drq_v2_encoder_factory

    return drq_v2_encoder_factory()


def _cnn3d_factory(config: Any):
    # num_frames is a schema (Layer A) concern, not an EncoderConfig one:
    # CombinedExtractor derives it from the schema's stacked entries and
    # calls cnn3d_encoder_factory(num_frames=...) directly instead of going
    # through EncoderConfig.image_encoder_factory() / this registry entry
    # (see rl_garden/encoders/combined.py). This entry exists only so
    # EncoderConfig(backbone="cnn3d").image_encoder_factory() still returns
    # a callable (e.g. for generic per-backbone checks); num_frames=1 is a
    # placeholder that is never used for real construction.
    from rl_garden.encoders import cnn3d_encoder_factory

    return cnn3d_encoder_factory(num_frames=1, features_dim=config.features_dim)


def _no_sac_kwargs(config: Any, schema: Any) -> dict[str, Any]:
    return {}


def _vit_sac_kwargs(config: Any, schema: Any) -> dict[str, Any]:
    from rl_garden.encoders import ViTTokenAndPropExtractor

    return {
        "policy_kwargs": {
            "features_extractor_class": ViTTokenAndPropExtractor,
            "features_extractor_kwargs": {
                "schema": schema,
                "fusion_mode": config.vit_fusion_mode,
                "embed_dim": config.vit_embed_dim,
                "depth": config.vit_depth,
                "num_heads": config.vit_num_heads,
                "embed_norm": config.vit_embed_norm,
                "augmentation": config.vit_augmentation,
                "random_shift_pad": config.vit_random_shift_pad,
            },
        },
        "actor_feature_dim": config.vit_actor_feature_dim,
        "critic_spatial_emb_dim": config.vit_critic_spatial_emb_dim,
    }


@dataclass(frozen=True)
class EncoderSpec:
    """Declares how one image encoder wires into the CLI/training layer.

    ``build_factory`` returns the flat image-encoder factory used by *all*
    algorithms (including PPO/eval). For the structured ViT/SAC path this factory
    is overridden by ``build_sac_kwargs``' ``policy_kwargs`` features extractor,
    but PPO/eval still consume the flat factory directly.

    ``build_sac_kwargs`` returns the structured-path kwargs for SAC-family
    constructors (``policy_kwargs`` + ``actor_feature_dim`` +
    ``critic_spatial_emb_dim``). It returns ``{}`` for encoders without a
    structured extractor, so callers can splat the result and fall back to the
    algorithm constructor defaults.

    ``build_sac_kwargs`` is intentionally scoped to the SAC family
    (SAC/CQL/CalQL/WSRL): only those constructors expose
    ``actor_feature_dim``/``critic_spatial_emb_dim``, and only ``SACPolicy``'s
    head consumes a ``token_and_prop`` structured extractor (actor token
    compression + spatial critic embedding). PPO/IQL/BC use only
    ``build_factory`` (the flat encoder) and have no structured path today.
    TODO(ppo-vit): to give PPO a structured ViT, add ``token_and_prop`` handling
    to the PPO policy and a parallel structured-kwargs builder here (e.g. a
    ``build_ppo_kwargs`` field, or a per-family mapping replacing
    ``build_sac_kwargs``), then have the PPO entrypoints splat it. The registry
    is the only place that would change — no other entrypoint needs touching.

    ``allows_resnet_weights`` records whether ``--pretrained_weights`` /
    ``--freeze_resnet_*`` apply (resnet only); it centralizes the compatibility
    check that used to be duplicated per-branch.
    """

    build_factory: Callable[[Any], Any]
    build_sac_kwargs: Callable[[Any, Any], dict[str, Any]]
    allows_resnet_weights: bool


# Single source of truth for image encoders. Adding a new encoder = one entry
# here (plus its name in EncoderConfig.backbone); training entrypoints stay
# encoder-agnostic. The ``test_encoder_registry_matches_backbone_literal`` test
# guards that this dict and the ``EncoderConfig.backbone`` Literal stay in sync.
ENCODER_REGISTRY: dict[str, EncoderSpec] = {
    "plain_conv": EncoderSpec(_plain_conv_factory, _no_sac_kwargs, allows_resnet_weights=False),
    "resnet10": EncoderSpec(_resnet_factory, _no_sac_kwargs, allows_resnet_weights=True),
    "resnet18": EncoderSpec(_resnet_factory, _no_sac_kwargs, allows_resnet_weights=True),
    "vit": EncoderSpec(_vit_factory, _vit_sac_kwargs, allows_resnet_weights=False),
    "drqv2_conv": EncoderSpec(_drqv2_conv_factory, _no_sac_kwargs, allows_resnet_weights=False),
    "cnn3d": EncoderSpec(_cnn3d_factory, _no_sac_kwargs, allows_resnet_weights=False),
}
