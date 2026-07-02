"""Shared dataclass CLI arguments for training examples."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

@dataclass
class LoggingArgs:
    log_dir: str = "runs"
    exp_name: Optional[str] = None
    log_freq: int = 1_000
    eval_freq: int = 10_000
    num_eval_steps: int = 50
    std_log: bool = True
    log_type: Literal["tensorboard", "wandb", "none"] = "wandb"
    log_keywords: Optional[str] = None
    wandb_project: str = "rl-garden"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None


@dataclass
class CheckpointArgs:
    checkpoint_dir: Optional[str] = None
    checkpoint_freq: int = 0
    load_checkpoint: Optional[str] = None
    save_replay_buffer: bool = False
    load_replay_buffer: bool = True
    save_final_checkpoint: bool = True


@dataclass
class VisionArgs:
    obs_mode: str = "rgb"
    include_state: bool = True
    frame_stack: int = 1
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64
    encoder: Literal[
        "plain_conv", "resnet10", "resnet18", "vit", "drqv2_conv", "cnn3d"
    ] = "plain_conv"
    encoder_features_dim: int = 256
    image_fusion_mode: Literal["stack_channels", "per_key"] = "stack_channels"
    vit_fusion_mode: Literal["per_key", "stack_channels"] = "per_key"
    vit_embed_dim: int = 128
    vit_depth: int = 1
    vit_num_heads: int = 4
    vit_embed_norm: bool = False
    vit_augmentation: Literal["random_shift", "none"] = "random_shift"
    vit_random_shift_pad: int = 4
    vit_actor_feature_dim: int = 128
    vit_critic_spatial_emb_dim: int = 1024
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False
    plain_conv_weight_init: Literal["kaiming_uniform", "orthogonal"] = "kaiming_uniform"
    plain_conv_last_act: bool = True
    plain_conv_pooling: Literal["flatten", "gap", "adaptive_max"] = "flatten"
    # Optional comma-separated image-key subset, e.g. ``rgb_base_camera``.
    # Leave unset to keep the env/default image-key discovery behavior.
    image_keys: Optional[str] = None
    # Default is intentionally off: random shifts consume augmentation RNG and
    # should be enabled only as an explicit visual-RL ablation.
    image_augmentation: Literal["none", "random_shift"] = "none"
    image_random_shift_pad: int = 4
    # Keep each camera as its own ``rgb_<cam>`` / ``depth_<cam>`` key instead of
    # channel-stacking. Required for multi-camera envs (e.g. peg) when each
    # camera should feed an independent encoder under ``per_key`` fusion.
    per_camera_rgbd: bool = False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_str(name: str, default: Optional[str]) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else None


def apply_log_env_defaults(args: LoggingArgs) -> None:
    """Apply RLG_* values to a default config before explicit CLI parsing."""
    args.std_log = _env_bool("RLG_STD_LOG", args.std_log)
    args.log_type = _env_str("RLG_LOG_TYPE", args.log_type) or args.log_type
    args.log_keywords = _env_str("RLG_LOG_KEYWORDS", args.log_keywords)
    args.wandb_project = (
        _env_str("RLG_WANDB_PROJECT", args.wandb_project) or args.wandb_project
    )
    args.wandb_entity = _env_str("RLG_WANDB_ENTITY", args.wandb_entity)
    args.wandb_group = _env_str("RLG_WANDB_GROUP", args.wandb_group)


apply_log_env_overrides = apply_log_env_defaults


def logging_args_from(args: Any) -> Optional[LoggingArgs]:
    """Return flat or nested logging arguments from a training config."""
    logging_args = getattr(args, "logging", args)
    return logging_args if hasattr(logging_args, "std_log") else None


def resolve_checkpoint_dir(args: Any, run_name: str) -> Optional[str]:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    if not args.save_final_checkpoint and args.checkpoint_freq <= 0:
        return None
    return os.path.join(args.log_dir, run_name, "checkpoints")


def resolve_eval_record_dir(args: Any, run_name: str) -> str:
    if args.eval_output_dir:
        return args.eval_output_dir
    return os.path.join(args.log_dir, run_name, "eval_videos")


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
    build_sac_kwargs: Callable[[Any, tuple[str, ...]], dict[str, Any]]
    allows_resnet_weights: bool


def _plain_conv_factory(args: VisionArgs):
    from rl_garden.encoders import default_image_encoder_factory

    return default_image_encoder_factory(
        features_dim=args.encoder_features_dim,
        plain_conv_last_act=args.plain_conv_last_act,
        plain_conv_weight_init=args.plain_conv_weight_init,
        plain_conv_pooling=args.plain_conv_pooling,
    )


def _resnet_factory(args: VisionArgs):
    from rl_garden.encoders import resnet_encoder_factory

    return resnet_encoder_factory(
        name=args.encoder,
        features_dim=args.encoder_features_dim,
        pretrained_weights=args.pretrained_weights,
        freeze_resnet_encoder=args.freeze_resnet_encoder,
        freeze_resnet_backbone=args.freeze_resnet_backbone,
    )


def _vit_factory(args: VisionArgs):
    # Image-only flat ViT factory used by generic CombinedExtractor paths (e.g.
    # PPO). SAC-family structured ViT instead installs ViTTokenAndPropExtractor
    # via vit_sac_kwargs_from_args(), which overrides the whole extractor.
    from rl_garden.encoders import vit_image_encoder_factory

    return vit_image_encoder_factory(
        features_dim=args.encoder_features_dim,
        embed_dim=args.vit_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_num_heads,
        embed_norm=args.vit_embed_norm,
        augmentation=args.vit_augmentation,
        random_shift_pad=args.vit_random_shift_pad,
    )


def _drqv2_conv_factory(args: VisionArgs):
    from rl_garden.encoders import drq_v2_encoder_factory

    return drq_v2_encoder_factory()


def _cnn3d_factory(args: VisionArgs):
    from rl_garden.encoders import cnn3d_encoder_factory

    return cnn3d_encoder_factory(
        num_frames=args.frame_stack,
        features_dim=args.encoder_features_dim,
    )


def _no_sac_kwargs(args: VisionArgs, image_keys: tuple[str, ...]) -> dict[str, Any]:
    return {}


def _vit_sac_kwargs(args: VisionArgs, image_keys: tuple[str, ...]) -> dict[str, Any]:
    from rl_garden.encoders import ViTTokenAndPropExtractor

    return {
        "policy_kwargs": {
            "features_extractor_class": ViTTokenAndPropExtractor,
            "features_extractor_kwargs": {
                "image_keys": image_keys,
                "state_key": "state",
                "use_proprio": args.include_state,
                "fusion_mode": args.vit_fusion_mode,
                "enable_stacking": False,
                "embed_dim": args.vit_embed_dim,
                "depth": args.vit_depth,
                "num_heads": args.vit_num_heads,
                "embed_norm": args.vit_embed_norm,
                "augmentation": args.vit_augmentation,
                "random_shift_pad": args.vit_random_shift_pad,
            },
        },
        "actor_feature_dim": args.vit_actor_feature_dim,
        "critic_spatial_emb_dim": args.vit_critic_spatial_emb_dim,
    }


# Single source of truth for image encoders. Adding a new encoder = one entry
# here (plus its name in VisionArgs.encoder); training/eval entrypoints stay
# encoder-agnostic. The ``test_encoder_registry_matches_literal`` test guards
# that this dict and the ``VisionArgs.encoder`` Literal stay in sync.
ENCODER_REGISTRY: dict[str, EncoderSpec] = {
    "plain_conv": EncoderSpec(_plain_conv_factory, _no_sac_kwargs, allows_resnet_weights=False),
    "resnet10": EncoderSpec(_resnet_factory, _no_sac_kwargs, allows_resnet_weights=True),
    "resnet18": EncoderSpec(_resnet_factory, _no_sac_kwargs, allows_resnet_weights=True),
    "vit": EncoderSpec(_vit_factory, _vit_sac_kwargs, allows_resnet_weights=False),
    "drqv2_conv": EncoderSpec(_drqv2_conv_factory, _no_sac_kwargs, allows_resnet_weights=False),
    "cnn3d": EncoderSpec(_cnn3d_factory, _no_sac_kwargs, allows_resnet_weights=False),
}


def _resolve_encoder_spec(args: VisionArgs) -> EncoderSpec:
    try:
        return ENCODER_REGISTRY[args.encoder]
    except KeyError:
        raise ValueError(
            f"Unknown encoder {args.encoder!r}. Known: {sorted(ENCODER_REGISTRY)}."
        )


def image_encoder_factory_from_args(args: VisionArgs):
    """Return the flat image-encoder factory for ``args.encoder``.

    Also enforces that resnet-only options (``--pretrained_weights`` /
    ``--freeze_resnet_*``) are not set for non-resnet encoders.
    """
    spec = _resolve_encoder_spec(args)
    if not spec.allows_resnet_weights and (
        args.pretrained_weights is not None
        or args.freeze_resnet_encoder
        or args.freeze_resnet_backbone
    ):
        raise ValueError(
            "--pretrained_weights, --freeze_resnet_encoder, and "
            "--freeze_resnet_backbone are only supported for resnet encoders."
        )
    if args.encoder != "plain_conv" and (
        getattr(args, "plain_conv_weight_init", "kaiming_uniform") != "kaiming_uniform"
        or getattr(args, "plain_conv_last_act", True) is not True
        or getattr(args, "plain_conv_pooling", "flatten") != "flatten"
    ):
        raise ValueError(
            "--plain_conv_weight_init, --plain_conv_last_act, and "
            "--plain_conv_pooling are only "
            "supported for the plain_conv encoder."
        )
    return spec.build_factory(args)


def vit_sac_kwargs_from_args(
    args: VisionArgs, image_keys: tuple[str, ...]
) -> dict[str, Any]:
    """Structured-path kwargs for SAC-family constructors, keyed by encoder.

    For encoders that install a structured features extractor (currently only
    ``vit``) this returns ``policy_kwargs`` plus the policy-head hyperparameters
    (``actor_feature_dim``, ``critic_spatial_emb_dim``). For every other encoder
    it returns ``{}`` so callers can splat the result and fall back to the
    algorithm constructor defaults (``actor_feature_dim=None``,
    ``critic_spatial_emb_dim=1024``, ``policy_kwargs=None``).

    NOTE(ppo-vit): the ``sac`` here is load-bearing, not just a label -- the
    returned dict is keyed to SAC-family constructor params and would raise
    ``TypeError`` if splatted into a PPO/IQL/BC constructor (they have no
    ``actor_feature_dim``/``critic_spatial_emb_dim``). When PPO gains
    ``token_and_prop`` handling, do NOT blindly reuse this bundle and do rename
    this to a per-family form: PPO's head kwargs will likely differ -- notably
    ``critic_spatial_emb_dim`` is a Q-critic concept, whereas PPO has a value
    head. Add a sibling builder (e.g. ``build_ppo_kwargs``) on ``EncoderSpec``
    rather than widening this one.
    """
    return _resolve_encoder_spec(args).build_sac_kwargs(args, image_keys)


def image_keys_from_obs_mode(obs_mode: str) -> tuple[str, ...]:
    return ("rgb",) if obs_mode == "rgb" else ("rgb", "depth")


def _parse_image_key_filter(value: Optional[str]) -> Optional[tuple[str, ...]]:
    if value is None:
        return None
    keys = tuple(k.strip() for k in value.split(",") if k.strip())
    if not keys:
        raise ValueError("image_keys must contain at least one key when provided.")
    return keys


def image_keys_from_env(env: Any, args: VisionArgs) -> tuple[str, ...]:
    """Resolve image keys for ``CombinedExtractor`` from the built env.

    When ``args.per_camera_rgbd`` is set the env emits one ``rgb_<cam>`` (and
    optionally ``depth_<cam>``) key per camera; we discover them from the
    observation space. Otherwise we fall back to the single-key default that
    matches ``FlattenRGBDObservationWrapper``.
    """
    explicit_keys = _parse_image_key_filter(args.image_keys)
    if explicit_keys is not None:
        obs_space = env.single_observation_space
        if hasattr(obs_space, "spaces"):
            missing = [key for key in explicit_keys if key not in obs_space.spaces]
            if missing:
                raise ValueError(
                    "Requested image_keys are not present in the observation space: "
                    + ", ".join(missing)
                )
        return explicit_keys
    if args.per_camera_rgbd:
        from rl_garden.encoders import discover_image_keys

        return discover_image_keys(env.single_observation_space)
    return image_keys_from_obs_mode(args.obs_mode)
