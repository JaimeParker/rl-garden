"""Residual SAC following the resfit action-coordinate convention."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.sac import SAC
from rl_garden.buffers.residual_buffer import (
    ResidualDictReplayBuffer,
    ResidualTensorReplayBuffer,
)
from rl_garden.common.action_scaler import ActionScaler
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.policies.residual_policy import ResidualSACPolicy


class ResidualSAC(SAC):
    """SAC that learns a residual action on top of a base policy.

    Internally, replay and critic actions are normalized to ``[-1, 1]``. The
    base policy returns env-space actions; ``ActionScaler`` maps them into the
    normalized coordinates used by residual learning.
    """

    def __init__(
        self,
        env: Any,
        *,
        base_action_provider: Any,
        residual_action_scale: float = 0.1,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> None:
        if residual_action_scale < 0:
            raise ValueError(
                f"residual_action_scale must be non-negative, got {residual_action_scale}."
        )
        self.base_action_provider = base_action_provider
        self.residual_action_scale = float(residual_action_scale)
        self.action_scaler = action_scaler
        self._cached_base_actions: Optional[torch.Tensor] = None
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
                "action_scaler_low": self.action_scaler.low.detach().cpu().tolist(),
                "action_scaler_high": self.action_scaler.high.detach().cpu().tolist(),
            }
        )
        return meta

    def _build_replay_buffer(self):
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Dict):
            return ResidualDictReplayBuffer(
                observation_space=obs_space,
                action_space=self._residual_action_space,
                num_envs=self.num_envs,
                buffer_size=self.buffer_size,
                storage_device=self.buffer_device,
                sample_device=self.device,
            )
        return ResidualTensorReplayBuffer(
            observation_space=obs_space,
            action_space=self._residual_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
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
        )

    def _setup_model(self) -> None:
        if self.action_scaler is None:
            self.action_scaler = ActionScaler.from_action_space(
                self.env.single_action_space, device=self.device
            )
        else:
            self.action_scaler = self.action_scaler.to(self.device)
        provider_to = getattr(self.base_action_provider, "to", None)
        if provider_to is not None:
            provider_to(self.device)
        super()._setup_model()

    def _call_base_action_provider(self, obs) -> torch.Tensor:
        provider = self.base_action_provider
        if callable(provider):
            base_action = provider(obs)
        elif hasattr(provider, "select_action"):
            base_action = provider.select_action(obs)
        elif hasattr(provider, "predict"):
            base_action = provider.predict(obs)
        else:
            raise TypeError(
                "base_action_provider must be callable or expose select_action()/predict()."
            )
        return torch.as_tensor(base_action, dtype=torch.float32, device=self.device)

    def _base_naction(self, obs) -> torch.Tensor:
        with torch.no_grad():
            policy_obs = self._obs_to_policy_device(obs)
            base_action = self._call_base_action_provider(policy_obs)
            return self.action_scaler.scale(base_action).clamp(-1.0, 1.0).detach()

    def _combine_base_residual(
        self, base_actions: torch.Tensor, unit_residual_actions: torch.Tensor
    ) -> torch.Tensor:
        residual_actions = unit_residual_actions * self.residual_action_scale
        return torch.clamp(base_actions + residual_actions, -1.0, 1.0)

    def _actor_action_log_prob(
        self,
        obs,
        *,
        base_actions: Optional[torch.Tensor] = None,
        stop_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if base_actions is None:
            raise ValueError("ResidualSAC requires base_actions for actor action sampling.")
        unit_residual, log_prob, features = self.policy.actor_action_log_prob(
            obs,
            base_actions=base_actions,
            stop_gradient=stop_gradient,
        )
        final_naction = self._combine_base_residual(base_actions, unit_residual)
        return final_naction, log_prob, features

    def _target_action_log_prob(
        self, data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._actor_action_log_prob(
            data.next_obs,
            base_actions=data.next_base_actions,
            stop_gradient=False,
        )

    def _actor_loss_from_batch(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._current_alpha().detach()
        action, log_prob, features = self._actor_action_log_prob(
            data.obs,
            base_actions=data.base_actions,
            stop_gradient=self._actor_stop_gradient(),
        )
        min_q = self.policy.min_q_value(features, action, subsample_size=None, target=False)
        return (alpha * log_prob - min_q).mean(), log_prob.detach()

    def _reset_base_action_provider(self, env_ids: Optional[torch.Tensor] = None) -> None:
        reset = getattr(self.base_action_provider, "reset", None)
        if reset is None:
            return
        if env_ids is None:
            reset()
            return
        try:
            reset(env_ids=env_ids)
        except TypeError:
            reset(env_ids)

    def _on_env_reset(self, obs) -> None:
        del obs
        self._reset_base_action_provider()
        self._cached_base_actions = None

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        if self._cached_base_actions is None:
            base_actions = self._base_naction(obs)
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

        final_naction = self._combine_base_residual(base_actions, unit_residual)
        env_action = self.action_scaler.unscale(final_naction)
        return final_naction, env_action, {"base_actions": base_actions}

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
        next_base_actions = self._base_naction(real_next_obs)
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
            base_actions = self._base_naction(obs)
            unit_residual = self.policy.predict(
                self._obs_to_policy_device(obs),
                base_actions=base_actions,
                deterministic=deterministic,
            ).detach()
            final_naction = self._combine_base_residual(base_actions, unit_residual)
            env_action = self.action_scaler.unscale(final_naction)
        if not return_info:
            return env_action
        return env_action, {
            "base_actions": base_actions,
            "unit_residual_actions": unit_residual,
            "residual_actions": unit_residual * self.residual_action_scale,
            "final_actions": final_naction,
        }

    def _policy_action(self, obs) -> torch.Tensor:
        _, info = self.get_action(obs, deterministic=False, return_info=True)
        return info["final_actions"]

    def _eval_action(self, obs) -> torch.Tensor:
        return self.get_action(obs, deterministic=True, return_info=False)

    def _evaluate(self) -> dict[str, float]:
        self._reset_base_action_provider()
        bind_env = getattr(self.base_action_provider, "bind_env", None)
        if bind_env is None or self.eval_env is None:
            return super()._evaluate()
        bind_env(self.eval_env)
        try:
            return super()._evaluate()
        finally:
            bind_env(self.env)
            self._reset_base_action_provider()
