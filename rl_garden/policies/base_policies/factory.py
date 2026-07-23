"""Factory for residual base-policy providers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from gymnasium import spaces

from rl_garden.policies.base_policies.act import (
    ACTBasePolicy,
    RoboTwinACTEEPoseBasePolicy,
)
from rl_garden.policies.base_policies.base import BasePolicyProvider
from rl_garden.policies.base_policies.sac import (
    SACBaseEncoder,
    SACBasePolicy,
    image_encoder_factory_from_name,
)
from rl_garden.policies.base_policies.zero import ZeroBasePolicy

BasePolicyKind = Literal["act", "sac", "zero"]


def make_base_policy(
    *,
    base_policy: BasePolicyKind,
    observation_space: spaces.Space,
    action_space: spaces.Box,
    env: Any | None = None,
    base_ckpt_path: str | Path | None = None,
    base_act_stats_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    base_act_temporal_agg: bool = True,
    base_act_temporal_agg_k: float = 0.01,
    base_act_image_size: int | tuple[int, int] | None = None,
    base_sac_encoder: SACBaseEncoder = "plain_conv",
    base_sac_encoder_features_dim: int = 256,
    base_sac_image_fusion_mode: str | None = None,
    base_sac_deterministic: bool = True,
) -> BasePolicyProvider:
    if base_policy == "zero":
        return ZeroBasePolicy(observation_space, action_space, device=device)
    if base_policy == "act":
        control_mode = getattr(getattr(env, "cfg", None), "control_mode", None)
        if control_mode == "ee_pose":
            if env is None or not hasattr(env, "qpos_targets_to_ee_pose"):
                raise ValueError(
                    "RoboTwin ee_pose ACT evaluation requires an environment "
                    "with qpos_targets_to_ee_pose()."
                )
            if tuple(action_space.shape) != (14,):
                raise ValueError(
                    "RoboTwin ee_pose ACT evaluation requires a 14D action "
                    f"space, got {action_space.shape}."
                )
            return RoboTwinACTEEPoseBasePolicy.from_checkpoint(
                env=env,
                observation_space=observation_space,
                action_space=action_space,
                ckpt_path=base_ckpt_path,
                stats_path=base_act_stats_path,
                temporal_agg=base_act_temporal_agg,
                temporal_agg_k=base_act_temporal_agg_k,
                image_size=base_act_image_size,
                device=device,
            )
        return ACTBasePolicy.from_checkpoint(
            observation_space=observation_space,
            action_space=action_space,
            ckpt_path=base_ckpt_path,
            stats_path=base_act_stats_path,
            env=env,
            temporal_agg=base_act_temporal_agg,
            temporal_agg_k=base_act_temporal_agg_k,
            image_size=base_act_image_size,
            device=device,
        )
    if base_policy == "sac":
        if base_ckpt_path is None:
            raise ValueError("--base_ckpt_path is required when --base_policy sac.")
        return SACBasePolicy.from_checkpoint(
            observation_space=observation_space,
            action_space=action_space,
            checkpoint_path=base_ckpt_path,
            device=device,
            deterministic=base_sac_deterministic,
            image_encoder_factory=image_encoder_factory_from_name(
                base_sac_encoder,
                features_dim=base_sac_encoder_features_dim,
            ),
            image_fusion_mode=base_sac_image_fusion_mode,
        )
    raise ValueError(f"Unsupported base_policy: {base_policy!r}.")
