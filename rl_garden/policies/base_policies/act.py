"""ACT base-policy adapter for residual RL."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.envs.robotwin.kinematics import (
    absolute_ee_pose_to_normalized_delta,
)
from rl_garden.policies.base_policies.base import BasePolicyOutput, BasePolicyProvider

if TYPE_CHECKING:
    from rl_garden.models.act.provider import StateObsGetter
else:
    StateObsGetter = Any


class ACTBasePolicy(BasePolicyProvider):
    """Wrap ``ACTBaseActionProvider`` behind the residual base-policy interface."""

    def __init__(
        self,
        provider,
        *,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(observation_space, action_space, device=device)
        self.provider = provider
        self.to(device)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        ckpt_path: str | Path | None = None,
        stats_path: str | Path | None = None,
        env: Any | None = None,
        state_obs_getter: Optional[StateObsGetter] = None,
        temporal_agg: bool = True,
        temporal_agg_k: float = 0.01,
        image_size: int | tuple[int, int] | None = None,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> "ACTBasePolicy":
        from rl_garden.models.act import ACTBaseActionProvider

        provider = ACTBaseActionProvider.from_checkpoint(
            observation_space=observation_space,
            action_space=action_space,
            ckpt_path=ckpt_path,
            stats_path=stats_path,
            env=env,
            state_obs_getter=state_obs_getter,
            temporal_agg=temporal_agg,
            temporal_agg_k=temporal_agg_k,
            image_size=image_size,
            strict=strict,
            device=device,
        )
        return cls(
            provider,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

    @property
    def checkpoint_path(self):
        return self.provider.checkpoint_path

    @property
    def spec(self):
        return self.provider.spec

    @property
    def config(self):
        return self.provider.config

    def to(self, *args, **kwargs):  # type: ignore[override]
        module = super().to(*args, **kwargs)
        self.provider.to(self.device)
        return module

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        self.provider.reset(env_ids=env_ids)

    def bind_env(self, env: Any) -> None:
        self.provider.bind_env(env)

    @torch.no_grad()
    def select_action(self, obs: Obs) -> BasePolicyOutput:
        actions = self.provider.select_action(self._obs_to_device(obs))
        return BasePolicyOutput(actions=self._format_actions(actions))


class RoboTwinACTEEPoseBasePolicy(BasePolicyProvider):
    """Local ACT base bridge from 14D qpos targets to 14D absolute EE poses.

    This wrapper is intentionally evaluation-only for now.  It does not imply
    that ResidualSAC's actor/critic/replay action spaces have been redesigned.
    """

    def __init__(
        self,
        act_policy: ACTBasePolicy,
        *,
        env: Any,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(observation_space, action_space, device=device)
        if tuple(action_space.shape) != (14,):
            raise ValueError(
                "RoboTwin ACT EE-pose base policy requires a 14D environment "
                f"action space, got {action_space.shape}."
            )
        self.act_policy = act_policy
        self.env = env
        self.to(device)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        env: Any,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        ckpt_path: str | Path | None = None,
        stats_path: str | Path | None = None,
        temporal_agg: bool = True,
        temporal_agg_k: float = 0.01,
        image_size: int | tuple[int, int] | None = None,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> "RoboTwinACTEEPoseBasePolicy":
        qpos_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(14,),
            dtype=action_space.dtype,
        )
        act_policy = ACTBasePolicy.from_checkpoint(
            observation_space=observation_space,
            action_space=qpos_space,
            ckpt_path=ckpt_path,
            stats_path=stats_path,
            env=env,
            temporal_agg=temporal_agg,
            temporal_agg_k=temporal_agg_k,
            image_size=image_size,
            strict=strict,
            device=device,
        )
        return cls(
            act_policy,
            env=env,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

    @property
    def checkpoint_path(self):
        return self.act_policy.checkpoint_path

    @property
    def spec(self):
        return self.act_policy.spec

    @property
    def config(self):
        return self.act_policy.config

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        self.act_policy.reset(env_ids=env_ids)

    def to(self, *args, **kwargs):  # type: ignore[override]
        module = super().to(*args, **kwargs)
        self.act_policy.to(self.device)
        return module

    def bind_env(self, env: Any) -> None:
        self.env = env
        self.act_policy.bind_env(env)

    @torch.no_grad()
    def select_action(self, obs: Obs) -> BasePolicyOutput:
        qpos_output = self.act_policy.select_action(obs)
        ee_actions = self.env.qpos_targets_to_ee_pose(qpos_output.actions)
        return BasePolicyOutput(
            actions=self._format_actions(ee_actions),
            info={"qpos_actions": qpos_output.actions.detach()},
        )


class RoboTwinACTEEDeltaPoseBasePolicy(RoboTwinACTEEPoseBasePolicy):
    """Convert ACT qpos targets to normalized world-frame EE deltas."""

    @torch.no_grad()
    def select_action(self, obs: Obs) -> BasePolicyOutput:
        if not isinstance(obs, dict) or "state" not in obs:
            raise ValueError(
                "RoboTwin ACT EE-delta base policy requires obs['state'] qpos."
            )
        qpos_output = self.act_policy.select_action(obs)
        current_qpos = obs["state"].detach().cpu().numpy()
        target_qpos = qpos_output.actions.detach().cpu().numpy()
        current_ee = np.asarray(
            self.env.qpos_targets_to_ee_pose(current_qpos),
            dtype=np.float32,
        )
        target_ee = np.asarray(
            self.env.qpos_targets_to_ee_pose(target_qpos),
            dtype=np.float32,
        )
        if current_ee.shape != target_ee.shape or current_ee.ndim != 2:
            raise ValueError(
                "RoboTwin ACT EE-delta FK outputs must have matching batched "
                f"shapes, got current={current_ee.shape}, target={target_ee.shape}."
            )
        cfg = self.env.cfg
        normalized_delta = np.stack(
            [
                absolute_ee_pose_to_normalized_delta(
                    current,
                    target,
                    ee_delta_pos_scale=cfg.ee_delta_pos_scale,
                    ee_delta_rot_scale=cfg.ee_delta_rot_scale,
                    gripper_delta_scale=cfg.gripper_delta_scale,
                )
                for current, target in zip(current_ee, target_ee)
            ]
        )
        return BasePolicyOutput(
            actions=self._format_actions(normalized_delta),
            info={"qpos_actions": qpos_output.actions.detach()},
        )
