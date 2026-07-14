"""Residual SAC following the resfit action-coordinate convention."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.sac import SAC
from rl_garden.buffers.residual_buffer import (
    ResidualDictReplayBuffer,
    ResidualTensorReplayBuffer,
)
from rl_garden.buffers.residual_h5 import (
    count_residual_h5_transitions,
    load_residual_h5_to_replay_buffer,
)
from rl_garden.common.action_scaler import ActionScaler
from rl_garden.common.types import ResidualReplayBufferSample
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.policies.base_policies import BasePolicyProvider
from rl_garden.policies.residual_policy import ResidualSACPolicy


ResidualActionCoordinates = Literal["normalized_final", "raw_joint_delta"]


class ResidualSAC(SAC):
    """SAC that learns a residual action on top of a base policy.

    ``normalized_final`` retains the legacy convention where replay and critic
    actions are normalized to ``[-1, 1]``. ``raw_joint_delta`` keeps the base,
    executed, replay, and critic actions in raw 14D joint-position coordinates
    while the actor continues to predict a unit residual in ``[-1, 1]``.
    """

    _compatible_checkpoint_algorithms = ("ResidualSAC",)
    _extra_batch_slice_keys = ("base_actions", "next_base_actions")

    def __init__(
        self,
        env: Any,
        *,
        base_action_provider: BasePolicyProvider,
        residual_action_scale: float = 0.1,
        residual_action_coordinates: ResidualActionCoordinates = "normalized_final",
        joint_delta_scale: float = 0.05,
        gripper_delta_scale: float = 0.2,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> None:
        if residual_action_coordinates not in {"normalized_final", "raw_joint_delta"}:
            raise ValueError(
                "residual_action_coordinates must be 'normalized_final' or "
                f"'raw_joint_delta', got {residual_action_coordinates!r}."
            )
        residual_action_scale = float(residual_action_scale)
        if not math.isfinite(residual_action_scale) or residual_action_scale < 0:
            raise ValueError(
                "residual_action_scale must be finite and non-negative, "
                f"got {residual_action_scale}."
            )
        joint_delta_scale = float(joint_delta_scale)
        gripper_delta_scale = float(gripper_delta_scale)
        if residual_action_coordinates == "raw_joint_delta":
            if env.single_action_space.shape != (14,):
                raise ValueError(
                    "raw_joint_delta requires a 14-dimensional action space, "
                    f"got shape={env.single_action_space.shape}."
                )
            if (
                not math.isfinite(joint_delta_scale)
                or joint_delta_scale < 0
                or not math.isfinite(gripper_delta_scale)
                or gripper_delta_scale < 0
            ):
                raise ValueError(
                    "raw joint and gripper delta scales must be finite and "
                    "non-negative, got "
                    f"joint_delta_scale={joint_delta_scale}, "
                    f"gripper_delta_scale={gripper_delta_scale}."
                )
        self.base_action_provider = base_action_provider
        self.residual_action_scale = residual_action_scale
        self.residual_action_coordinates = residual_action_coordinates
        self.joint_delta_scale = joint_delta_scale
        self.gripper_delta_scale = gripper_delta_scale
        self.action_scaler = action_scaler
        self.resolved_residual_scale: Optional[torch.Tensor] = None
        self._cached_base_actions: Optional[torch.Tensor] = None
        self.offline_replay_buffer = None
        self.offline_data_ratio = 0.0
        self._residual_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.single_action_space.shape,
            dtype=np.float32,
        )
        super().__init__(env=env, **kwargs)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = super()._checkpoint_metadata()
        meta.update(
            {
                "residual_action_scale": self.residual_action_scale,
                "residual_action_coordinates": self.residual_action_coordinates,
                "joint_delta_scale": self.joint_delta_scale,
                "gripper_delta_scale": self.gripper_delta_scale,
            }
        )
        if self.residual_action_coordinates == "normalized_final":
            assert self.action_scaler is not None
            meta.update(
                {
                    "action_scaler_low": self.action_scaler.low.detach().cpu().tolist(),
                    "action_scaler_high": self.action_scaler.high.detach().cpu().tolist(),
                }
            )
        else:
            assert self.resolved_residual_scale is not None
            meta["resolved_residual_scale"] = (
                self.resolved_residual_scale.detach().cpu().tolist()
            )
        return meta

    def _build_replay_buffer(self):
        return self._make_residual_replay_buffer(self.buffer_size)

    def _make_residual_replay_buffer(
        self, buffer_size: int, *, num_envs: Optional[int] = None
    ):
        num_envs = self.num_envs if num_envs is None else int(num_envs)
        obs_space = self.env.single_observation_space
        action_space = (
            self.env.single_action_space
            if self.residual_action_coordinates == "raw_joint_delta"
            else self._residual_action_space
        )
        if isinstance(obs_space, spaces.Dict):
            return ResidualDictReplayBuffer(
                observation_space=obs_space,
                action_space=action_space,
                num_envs=num_envs,
                buffer_size=buffer_size,
                storage_device=self.buffer_device,
                sample_device=self.device,
            )
        return ResidualTensorReplayBuffer(
            observation_space=obs_space,
            action_space=action_space,
            num_envs=num_envs,
            buffer_size=buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def load_offline_replay_buffer(
        self,
        path: str | Path,
        *,
        num_traj: Optional[int] = None,
        buffer_size: Optional[int] = None,
        offline_data_ratio: float = 0.5,
    ) -> int:
        if not (0.0 <= offline_data_ratio <= 1.0):
            raise ValueError(
                f"offline_data_ratio must be in [0, 1], got {offline_data_ratio}."
            )
        if buffer_size is None:
            buffer_size = count_residual_h5_transitions(
                path, num_traj=num_traj, num_envs=1
            )
            if buffer_size <= 0:
                raise ValueError(
                    f"Offline residual dataset has no transitions to load: path={path}."
                )
        self.offline_replay_buffer = self._make_residual_replay_buffer(
            int(buffer_size), num_envs=1
        )
        loaded = load_residual_h5_to_replay_buffer(
            self.offline_replay_buffer,
            path,
            num_traj=num_traj,
            bootstrap_at_done=self.bootstrap_at_done,
        )
        self.offline_data_ratio = float(offline_data_ratio)
        if self.logger is not None:
            self.logger.add_summary("residual/offline_loaded_transitions", loaded)
            self.logger.add_summary(
                "residual/offline_data_ratio", self.offline_data_ratio
            )
            self.logger.add_summary("residual/offline_buffer_size", int(buffer_size))
        return loaded

    def _sample_train_batch(self, batch_size: int):
        if self.offline_replay_buffer is None or self.offline_data_ratio <= 0.0:
            return self.replay_buffer.sample(batch_size)
        if len(self.offline_replay_buffer) == 0:
            return self.replay_buffer.sample(batch_size)
        if len(self.replay_buffer) == 0:
            return self.offline_replay_buffer.sample(batch_size)

        n_offline = int(round(batch_size * self.offline_data_ratio))
        n_offline = min(max(n_offline, 0), batch_size)
        n_online = batch_size - n_offline
        if n_offline == 0:
            return self.replay_buffer.sample(batch_size)
        if n_online == 0:
            return self.offline_replay_buffer.sample(batch_size)
        return self._concat_replay_samples(
            self.replay_buffer.sample(n_online),
            self.offline_replay_buffer.sample(n_offline),
        )

    @staticmethod
    def _concat_replay_samples(
        a: ResidualReplayBufferSample,
        b: ResidualReplayBufferSample,
    ) -> ResidualReplayBufferSample:
        def _cat(x, y):
            if isinstance(x, dict):
                return {k: torch.cat([x[k], y[k]], dim=0) for k in x}
            return torch.cat([x, y], dim=0)

        return ResidualReplayBufferSample(
            obs=_cat(a.obs, b.obs),
            next_obs=_cat(a.next_obs, b.next_obs),
            actions=_cat(a.actions, b.actions),
            rewards=_cat(a.rewards, b.rewards),
            dones=_cat(a.dones, b.dones),
            base_actions=_cat(a.base_actions, b.base_actions),
            next_base_actions=_cat(a.next_base_actions, b.next_base_actions),
        )

    def _policy_action_space(self) -> spaces.Box:
        return self._residual_action_space

    def _build_policy(self, features_extractor: BaseFeaturesExtractor) -> ResidualSACPolicy:
        return ResidualSACPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self._residual_action_space,
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
            critic_impl=self.critic_impl,
            actor_feature_dim=self.actor_feature_dim,
            critic_spatial_emb_dim=self.critic_spatial_emb_dim,
            critic_use_layer_norm=True,
        )

    def _setup_model(self) -> None:
        if self.residual_action_coordinates == "raw_joint_delta":
            if self.action_scaler is not None:
                raise ValueError(
                    "raw_joint_delta does not accept an ActionScaler because raw "
                    "joint actions must not be normalized."
                )
            self.resolved_residual_scale = self._build_raw_residual_scale()
        else:
            if self.action_scaler is None:
                self.action_scaler = ActionScaler.from_action_space(
                    self.env.single_action_space, device=self.device
                )
            else:
                self.action_scaler = self.action_scaler.to(self.device)
        self.base_action_provider.to(self.device)
        super()._setup_model()

    def _build_raw_residual_scale(self) -> torch.Tensor:
        scale = torch.tensor(
            [self.joint_delta_scale] * 6
            + [self.gripper_delta_scale]
            + [self.joint_delta_scale] * 6
            + [self.gripper_delta_scale],
            dtype=torch.float32,
            device=self.device,
        )
        scale = scale * self.residual_action_scale
        if not torch.isfinite(scale).all().item():
            raise ValueError("Resolved raw residual scale contains non-finite values.")
        return scale

    @staticmethod
    def _sanitize_raw_joint_action(action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 0 or action.shape[-1] != 14:
            raise ValueError(
                "Raw joint action must have 14 values in its last dimension, "
                f"got shape={tuple(action.shape)}."
            )
        if not torch.isfinite(action).all().item():
            raise ValueError("Raw joint action contains non-finite values.")
        return torch.cat(
            (
                action[..., :6],
                action[..., 6:7].clamp(0.0, 1.0),
                action[..., 7:13],
                action[..., 13:14].clamp(0.0, 1.0),
            ),
            dim=-1,
        )

    def _compose_raw_action(
        self,
        base_raw: torch.Tensor,
        unit_residual: torch.Tensor,
    ) -> torch.Tensor:
        base_raw = self._sanitize_raw_joint_action(base_raw)
        if unit_residual.shape != base_raw.shape:
            raise ValueError(
                "Unit residual shape must match the raw base action, "
                f"got residual={tuple(unit_residual.shape)} and "
                f"base={tuple(base_raw.shape)}."
            )
        if not torch.isfinite(unit_residual).all().item():
            raise ValueError("Unit residual action contains non-finite values.")
        assert self.resolved_residual_scale is not None
        raw_delta = unit_residual * self.resolved_residual_scale
        if not torch.isfinite(raw_delta).all().item():
            raise ValueError("Raw residual delta contains non-finite values.")
        return self._sanitize_raw_joint_action(base_raw + raw_delta)

    def _call_base_action_provider(self, obs) -> torch.Tensor:
        output = self.base_action_provider.select_action(obs)
        return torch.as_tensor(output.actions, dtype=torch.float32, device=self.device)

    def _base_action(self, obs) -> torch.Tensor:
        with torch.no_grad():
            policy_obs = self._obs_to_policy_device(obs)
            base_action = self._call_base_action_provider(policy_obs)
            if self.residual_action_coordinates == "raw_joint_delta":
                return self._sanitize_raw_joint_action(base_action).detach()
            assert self.action_scaler is not None
            return self.action_scaler.scale(base_action).clamp(-1.0, 1.0).detach()

    def _combine_base_residual(
        self, base_actions: torch.Tensor, unit_residual_actions: torch.Tensor
    ) -> torch.Tensor:
        if self.residual_action_coordinates == "raw_joint_delta":
            return self._compose_raw_action(base_actions, unit_residual_actions)
        residual_actions = unit_residual_actions * self.residual_action_scale
        return torch.clamp(base_actions + residual_actions, -1.0, 1.0)

    def _scaled_residual_action(
        self, unit_residual_actions: torch.Tensor
    ) -> torch.Tensor:
        if self.residual_action_coordinates == "raw_joint_delta":
            assert self.resolved_residual_scale is not None
            return unit_residual_actions * self.resolved_residual_scale
        return unit_residual_actions * self.residual_action_scale

    def _residual_actor_action_log_prob(
        self,
        obs,
        base_actions: torch.Tensor,
        *,
        stop_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unit_residual, log_prob, features = self.policy.actor_action_log_prob(
            obs,
            base_actions=base_actions,
            stop_gradient=stop_gradient,
        )
        final_action = self._combine_base_residual(base_actions, unit_residual)
        return final_action, log_prob, features

    def _target_action_log_prob(
        self, data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._residual_actor_action_log_prob(
            data.next_obs, data.next_base_actions, stop_gradient=False,
        )

    def _actor_loss_from_batch(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._current_alpha().detach()
        action, log_prob, features = self._residual_actor_action_log_prob(
            data.obs, data.base_actions, stop_gradient=self._actor_stop_gradient(),
        )
        min_q = self.policy.min_q_value(features, action, subsample_size=None, target=False)
        return (alpha * log_prob - min_q).mean(), log_prob.detach()

    def _compute_actor_diagnostics(self, data) -> dict[str, torch.Tensor]:
        return self.policy.actor_diagnostics(data.obs, data.base_actions)

    def _reset_base_action_provider(self, env_ids: Optional[torch.Tensor] = None) -> None:
        self.base_action_provider.reset(env_ids=env_ids)

    def _on_env_reset(self, obs) -> None:
        del obs
        self._reset_base_action_provider()
        self._cached_base_actions = None

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        if self._cached_base_actions is None:
            base_actions = self._base_action(obs)
        else:
            base_actions = self._cached_base_actions
            self._cached_base_actions = None
        if not learning_has_started:
            shape = (self.num_envs,) + self.env.single_action_space.shape
            unit_residual = 2 * torch.rand(shape, dtype=torch.float32, device=self.device) - 1
        else:
            with torch.no_grad():
                unit_residual = self.policy.predict(
                    self._obs_to_policy_device(obs),
                    base_actions=base_actions,
                    deterministic=False,
                ).detach()

        final_action = self._combine_base_residual(base_actions, unit_residual)
        if self.residual_action_coordinates == "raw_joint_delta":
            env_action = final_action
        else:
            assert self.action_scaler is not None
            env_action = self.action_scaler.unscale(final_action)
        return final_action, env_action, {"base_actions": base_actions}

    def _replay_buffer_add_kwargs(
        self,
        action_context: Optional[dict[str, Any]],
        obs,
        next_obs,
        real_next_obs,
        infos,
        need_final_obs: torch.Tensor,
    ) -> dict[str, Any]:
        del obs, next_obs, infos, need_final_obs
        assert action_context is not None
        next_base_actions = self._base_action(real_next_obs)
        self._cached_base_actions = next_base_actions.detach()
        return {
            "base_actions": action_context["base_actions"],
            "next_base_actions": next_base_actions,
        }

    def _post_rollout_step(
        self,
        action_context: Optional[dict[str, Any]],
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos,
    ) -> None:
        del action_context, infos
        done = terminations | truncations
        if done.any():
            self._reset_base_action_provider(torch.where(done)[0])
            self._cached_base_actions = None

    def get_action(
        self,
        obs,
        deterministic: bool = False,
        return_info: bool = False,
    ):
        with torch.no_grad():
            base_actions = self._base_action(obs)
            unit_residual = self.policy.predict(
                self._obs_to_policy_device(obs),
                base_actions=base_actions,
                deterministic=deterministic,
            ).detach()
            final_action = self._combine_base_residual(base_actions, unit_residual)
            if self.residual_action_coordinates == "raw_joint_delta":
                env_action = final_action
            else:
                assert self.action_scaler is not None
                env_action = self.action_scaler.unscale(final_action)
        if not return_info:
            return env_action
        return env_action, {
            "base_actions": base_actions,
            "unit_residual_actions": unit_residual,
            "residual_actions": self._scaled_residual_action(unit_residual),
            "final_actions": final_action,
        }

    def _policy_action(self, obs) -> torch.Tensor:
        _, info = self.get_action(obs, deterministic=False, return_info=True)
        return info["final_actions"]

    def _eval_action(self, obs) -> torch.Tensor:
        return self.get_action(obs, deterministic=True, return_info=False)

    def _eval_action_and_critic_action(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        env_action, info = self.get_action(obs, deterministic=True, return_info=True)
        return env_action, info["final_actions"]

    def _evaluate(self) -> dict[str, float]:
        self._reset_base_action_provider()
        if self.eval_env is None:
            return super()._evaluate()
        self.base_action_provider.bind_env(self.eval_env)
        try:
            return super()._evaluate()
        finally:
            self.base_action_provider.bind_env(self.env)
            self._reset_base_action_provider()
