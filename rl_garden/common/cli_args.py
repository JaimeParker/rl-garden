"""Shared dataclass CLI arguments for training examples."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObservationConfig, ObsGroups

# Duplicated (not imported) from rl_garden.algorithms._observation.EncoderSharing
# to keep this module's import graph light -- algorithms is a heavy package
# and cli_args is imported very early by every training entrypoint.
EncoderSharing = Literal["shared_critic_grad", "shared", "separate"]


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
    load_replay_buffer: bool = False
    save_final_checkpoint: bool = True


@dataclass
class ObservationArgs:
    """Unified "what is observed / how it is encoded" CLI surface.

    Shared by every offline/online/off2on algorithm's args class (a single
    surface replacing the old, per-family flat vision-args split). ``obs`` decides
    composition (state/rgb/depth/image_size/frame_stack); ``encoder``
    decides how any images are encoded; ``obs_groups`` optionally splits
    actor/critic observation keys (asymmetric/privileged critic);
    ``critic_encoder`` optionally gives the critic its own encoder
    hyperparameters (only meaningful together with ``obs_groups``/
    ``encoder_sharing="separate"``); ``encoder_sharing`` overrides the
    algorithm's own actor/critic encoder-sharing default.

    ``critic_encoder`` is intentionally not ``Optional[EncoderConfig]``:
    tyro represents ``Optional[<dataclass>]`` as a subcommand union
    (``--critic-encoder:encoder-config`` / ``--critic-encoder:none``)
    instead of flat ``--critic-encoder.*`` flags, and the YAML preset
    loader's ``apply_strict_mapping`` cannot type-check a mapping against a
    ``Union`` of dataclasses. "Not set" is instead the sentinel
    ``critic_encoder == EncoderConfig()`` (the default instance) --
    training entrypoints and ``inactive_config_paths`` treat that sentinel
    as "no override" the same way ``None`` would read.
    """

    obs: ObservationConfig = field(default_factory=ObservationConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    obs_groups: ObsGroups = field(default_factory=ObsGroups)
    critic_encoder: EncoderConfig = field(default_factory=EncoderConfig)
    encoder_sharing: Optional[EncoderSharing] = None


def resolve_critic_encoder_config(args: Any) -> Optional[EncoderConfig]:
    """``args.critic_encoder`` (see ``ObservationArgs``) is a sentinel-valued
    ``EncoderConfig``, not ``Optional[EncoderConfig]`` -- training
    entrypoints call this instead of reading ``args.critic_encoder``
    directly so "not set" (the default ``EncoderConfig()`` instance) reaches
    the algorithm as ``None`` (its own default), not as an explicit
    ``critic_encoder_config`` that would incorrectly read downstream as "an
    asymmetric critic encoder was requested"."""
    critic_encoder = args.critic_encoder
    return critic_encoder if critic_encoder != EncoderConfig() else None


def resolve_obs_groups_config(args: Any) -> Optional[ObsGroups]:
    """``args.obs_groups`` (see ``ObservationArgs``) is a sentinel-valued
    ``ObsGroups``, not ``Optional[ObsGroups]`` -- training entrypoints call
    this instead of reading ``args.obs_groups`` directly so "not set" (both
    ``actor``/``critic`` left at their default ``None``) reaches the
    algorithm as ``None`` (its own default: no restriction). Any explicit
    ``--obs-groups.actor``/``--obs-groups.critic`` override is forwarded
    as-is, even when the two happen to resolve to the same keys (symmetric):
    ``ObsGroups.is_symmetric`` is a syntactic actor==critic comparison, not
    "no restriction was requested" -- dropping an explicit-but-symmetric
    restriction to ``None`` would silently widen it back to every schema key.
    """
    obs_groups = args.obs_groups
    if obs_groups.actor is None and obs_groups.critic is None:
        return None
    return obs_groups


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

