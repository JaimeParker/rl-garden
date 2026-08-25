"""Shared dataclass CLI arguments for training examples."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Optional

@dataclass
class LoggingArgs:
    log_dir: str = "runs"
    exp_name: Optional[str] = None
    log_freq: int = 1_000
    eval_freq: int = 10_000
    num_eval_steps: int = 50
    std_log: bool = True
    log_type: Literal["tensorboard", "wandb", "none"] = "tensorboard"
    log_keywords: Optional[str] = None
    wandb_project: str = "rl-garden"
    wandb_entity: Optional[str] = None
    # Groups runs in wandb's UI and nests TensorBoard's writer directory
    # (log_dir/<log_group>/<run_name>/) -- see Logger.create in
    # rl_garden/common/logger.py.
    log_group: Optional[str] = None


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
    # Matches rl_garden.encoders.resnet.PoolingMethod's values -- not
    # imported directly to keep this module's top-level imports light (see
    # _resnet_factory's own lazy import of the encoder module below).
    pooling_method: Literal["spatial_learned_embeddings", "spatial_softmax", "avg"] = "spatial_softmax"
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
    # Opt-in heterogeneous critic encoder. Unset (default) keeps today's
    # exact behavior: actor and critic share one encoder instance and
    # ``backbone_type``. See critic_features_extractor_kwargs_from_args()
    # below.
    critic_encoder: Optional[
        Literal["plain_conv", "resnet10", "resnet18", "vit", "drqv2_conv", "cnn3d"]
    ] = None
    critic_backbone_type: Optional[Literal["mlp", "mlp_resnet"]] = None
    # Critic-only obs-key overrides (asymmetric/privileged-critic-lite --
    # subsets keys already present in the obs dict, does not synthesize new
    # ones). Unset falls back to the actor's image_keys/include_state.
    critic_image_keys: Optional[str] = None
    critic_include_state: Optional[bool] = None


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


def resolve_num_eval_steps(
    *,
    num_eval_steps: Optional[int],
    num_eval_episodes: Optional[int],
    eval_episode_horizon: Optional[int],
    default: int,
) -> int:
    """Resolve the eval-loop step cap.

    An explicit ``num_eval_steps`` always wins. Otherwise, if both
    ``num_eval_episodes`` and ``eval_episode_horizon`` (expected worst-case
    steps for one eval episode) are set, the budget is derived so an
    episode-count-driven eval loop (e.g. ``run_exact_episode_eval``) has room
    to finish ``num_eval_episodes`` episodes. Falls back to ``default``
    otherwise -- including when only one of the two is set, since a fixed-step
    eval loop (no episode target) can't use a horizon to size anything.
    """
    if num_eval_steps is not None:
        return int(num_eval_steps)
    if eval_episode_horizon is not None and num_eval_episodes is not None:
        return max(int(num_eval_episodes) * int(eval_episode_horizon), 1)
    return default


def warn_if_eval_budget_undersized(
    *,
    num_eval_steps: Optional[int],
    num_eval_episodes: Optional[int],
    eval_episode_horizon: Optional[int],
) -> None:
    """Warn about likely-misconfigured eval budgets.

    Two independent, backend/env-agnostic cases: an explicit step cap too
    small to let ``num_eval_episodes`` finish given ``eval_episode_horizon``,
    or a horizon that was set but has no episode target to size a budget for.
    """
    if (
        num_eval_steps is not None
        and eval_episode_horizon is not None
        and num_eval_episodes is not None
    ):
        derived = int(num_eval_episodes) * int(eval_episode_horizon)
        if int(num_eval_steps) < derived:
            warnings.warn(
                f"num_eval_steps={num_eval_steps} is below "
                f"num_eval_episodes={num_eval_episodes} x "
                f"eval_episode_horizon={eval_episode_horizon}={derived}; "
                "evaluation may stop before "
                f"{num_eval_episodes} episodes finish (watch "
                "eval/episodes_completed). Leave --num_eval_steps unset to "
                "derive the budget automatically.",
                RuntimeWarning,
                stacklevel=2,
            )
    elif eval_episode_horizon is not None and num_eval_episodes is None:
        warnings.warn(
            f"--eval_episode_horizon={eval_episode_horizon} was ignored: this "
            "algorithm evaluates with a fixed step budget (no episode "
            "target), so the horizon cannot size the budget. Set "
            "--num_eval_episodes to enable episode-count evaluation, or "
            "raise --num_eval_steps directly.",
            RuntimeWarning,
            stacklevel=2,
        )


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
        pooling_method=args.pooling_method,
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


def critic_features_extractor_kwargs_from_args(
    args: VisionArgs, image_keys: tuple[str, ...]
) -> dict[str, Any]:
    """``policy_kwargs``-shaped ``{critic_features_extractor_class,
    critic_features_extractor_kwargs}`` for ``args.critic_encoder`` -- covers
    both the flat ``CombinedExtractor`` path and vit's structured
    ``ViTTokenAndPropExtractor`` path (dispatched the same way
    ``vit_sac_kwargs_from_args`` does for the actor/shared path), so the
    caller doesn't need to special-case vit. Returns ``{}`` when
    ``args.critic_encoder`` is unset -- purely additive; the default/shared
    path never calls this.
    """
    if not args.critic_encoder:
        return {}
    critic_args = replace(args, encoder=args.critic_encoder)
    sac_kwargs = vit_sac_kwargs_from_args(critic_args, image_keys)
    if sac_kwargs:
        policy_kwargs = sac_kwargs["policy_kwargs"]
        return {
            "critic_features_extractor_class": policy_kwargs["features_extractor_class"],
            "critic_features_extractor_kwargs": policy_kwargs["features_extractor_kwargs"],
        }
    from rl_garden.encoders import CombinedExtractor

    use_proprio = (
        args.critic_include_state
        if args.critic_include_state is not None
        else args.include_state
    )
    return {
        "critic_features_extractor_class": CombinedExtractor,
        "critic_features_extractor_kwargs": dict(
            image_keys=image_keys,
            image_encoder_factory=image_encoder_factory_from_args(critic_args),
            use_proprio=use_proprio,
            fusion_mode=args.image_fusion_mode,
            enable_stacking=args.frame_stack > 1,
            image_augmentation=args.image_augmentation,
            random_shift_pad=args.image_random_shift_pad,
        ),
    }


def image_keys_from_obs_mode(obs_mode: str) -> tuple[str, ...]:
    return ("rgb",) if obs_mode == "rgb" else ("rgb", "depth")


def _parse_image_key_filter(value: Optional[str]) -> Optional[tuple[str, ...]]:
    if value is None:
        return None
    keys = tuple(k.strip() for k in value.split(",") if k.strip())
    if not keys:
        raise ValueError("image_keys must contain at least one key when provided.")
    return keys


def image_keys_from_env(
    env: Any, args: VisionArgs, image_key_filter: Optional[str] = None
) -> tuple[str, ...]:
    """Resolve image keys for ``CombinedExtractor`` from the built env.

    When ``args.per_camera_rgbd`` is set the env emits one ``rgb_<cam>`` (and
    optionally ``depth_<cam>``) key per camera; we discover them from the
    observation space. Otherwise we fall back to the single-key default that
    matches ``FlattenRGBDObservationWrapper``.

    ``image_key_filter`` overrides ``args.image_keys`` when given (e.g. to
    resolve ``args.critic_image_keys`` through this same logic instead).
    """
    explicit_keys = _parse_image_key_filter(
        image_key_filter if image_key_filter is not None else args.image_keys
    )
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
