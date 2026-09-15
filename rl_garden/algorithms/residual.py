"""Residual SAC following the resfit action-coordinate convention."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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
from rl_garden.common.checkpoint import load_checkpoint_file, space_metadata
from rl_garden.common.types import ResidualReplayBufferSample
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.combined import CombinedExtractor
from rl_garden.policies.base_policies import BasePolicyProvider
from rl_garden.policies.residual_policy import ResidualSACPolicy


class ResidualSAC(SAC):
    """SAC that learns a residual action on top of a base policy.

    Internally, replay and critic actions are normalized to ``[-1, 1]``. The
    base policy returns env-space actions; ``ActionScaler`` maps them into the
    normalized coordinates used by residual learning.
    """

    _compatible_checkpoint_algorithms = ("ResidualSAC",)
    _extra_batch_slice_keys = ("base_actions", "next_base_actions")

    def __init__(
        self,
        env: Any,
        *,
        base_action_provider: BasePolicyProvider,
        residual_action_scale: float = 0.1,
        residual_gripper_action_scale: Optional[float] = None,
        residual_warmup_scale: float = 0.0,
        residual_actor_zero_init: bool = True,
        residual_log_std_init: float = -3.0,
        residual_warmup_policy_checkpoint: Optional[str | Path] = None,
        residual_warmup_policy_probability: float = 0.5,
        action_scaler: Optional[ActionScaler] = None,
        **kwargs,
    ) -> None:
        if residual_action_scale < 0:
            raise ValueError(
                f"residual_action_scale must be non-negative, got {residual_action_scale}."
            )
        self.base_action_provider = base_action_provider
        self.residual_action_scale = float(residual_action_scale)
        if (
            residual_gripper_action_scale is not None
            and residual_gripper_action_scale < 0
        ):
            raise ValueError(
                "residual_gripper_action_scale must be non-negative, got "
                f"{residual_gripper_action_scale}."
            )
        self.residual_gripper_action_scale = (
            None
            if residual_gripper_action_scale is None
            else float(residual_gripper_action_scale)
        )
        if self.residual_gripper_action_scale is not None and env.single_action_space.shape != (14,):
            raise ValueError("Per-gripper residual scale requires a 14D action space.")
        if not 0.0 <= residual_warmup_scale <= 1.0:
            raise ValueError(
                "residual_warmup_scale must be in [0, 1], got "
                f"{residual_warmup_scale}."
            )
        self.residual_warmup_scale = float(residual_warmup_scale)
        self.residual_actor_zero_init = bool(residual_actor_zero_init)
        self.residual_log_std_init = float(residual_log_std_init)
        if not 0.0 <= residual_warmup_policy_probability <= 1.0:
            raise ValueError(
                "residual_warmup_policy_probability must be in [0, 1], got "
                f"{residual_warmup_policy_probability}."
            )
        self.residual_warmup_policy_checkpoint = (
            None
            if residual_warmup_policy_checkpoint is None
            else str(residual_warmup_policy_checkpoint)
        )
        self.residual_warmup_policy_probability = float(
            residual_warmup_policy_probability
        )
        self._warmup_policy: Optional[ResidualSACPolicy] = None
        self._warmup_use_policy: Optional[torch.Tensor] = None
        self.action_scaler = action_scaler
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
        if self.residual_warmup_policy_checkpoint is not None:
            self._warmup_policy = self._load_state_warmup_policy(
                self.residual_warmup_policy_checkpoint
            )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = super()._checkpoint_metadata()
        meta.update(
            {
                "residual_action_scale": self.residual_action_scale,
                "residual_gripper_action_scale": (
                    self.residual_gripper_action_scale
                ),
                "residual_warmup_scale": self.residual_warmup_scale,
                "residual_actor_zero_init": self.residual_actor_zero_init,
                "residual_log_std_init": self.residual_log_std_init,
                "residual_warmup_policy_checkpoint": (
                    self.residual_warmup_policy_checkpoint
                ),
                "residual_warmup_policy_probability": (
                    self.residual_warmup_policy_probability
                ),
                "action_scaler_low": self.action_scaler.low.detach().cpu().tolist(),
                "action_scaler_high": self.action_scaler.high.detach().cpu().tolist(),
            }
        )
        return meta

    def _load_state_warmup_policy(self, path: str) -> ResidualSACPolicy:
        checkpoint = load_checkpoint_file(path, map_location=self.device)
        metadata = checkpoint.get("metadata", {})
        if metadata.get("algorithm_class") != "ResidualSAC":
            raise ValueError(
                "Warmup policy checkpoint must be a ResidualSAC checkpoint, got "
                f"{metadata.get('algorithm_class')!r}."
            )
        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Dict) or "state" not in obs_space.spaces:
            raise ValueError(
                "Warmup ResidualSAC checkpoint requires a Dict observation with state."
            )
        state_obs_space = spaces.Dict({"state": obs_space.spaces["state"]})
        expected_obs = space_metadata(state_obs_space)
        if metadata.get("observation_space") != expected_obs:
            raise ValueError(
                "Warmup ResidualSAC checkpoint observation space is not state-only "
                "and compatible with the current state observation."
            )
        expected_action = space_metadata(self.env.single_action_space)
        if metadata.get("action_space") != expected_action:
            raise ValueError(
                "Warmup ResidualSAC checkpoint action space does not match the "
                "current environment."
            )

        hparams = metadata.get("hyperparameters", {})
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index or torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            features_extractor = CombinedExtractor(
                state_obs_space,
                image_keys=hparams.get("image_keys", ("rgb", "depth")),
                state_key=hparams.get("state_key", "state"),
                proprio_latent_dim=int(hparams.get("proprio_latent_dim", 64)),
                use_proprio=bool(hparams.get("use_proprio", True)),
                fusion_mode=hparams.get("image_fusion_mode", "stack_channels"),
                enable_stacking=bool(hparams.get("enable_stacking", False)),
                image_augmentation="none",
            )
            policy = ResidualSACPolicy(
                observation_space=state_obs_space,
                action_space=self._residual_action_space,
                features_extractor=features_extractor,
                net_arch=hparams.get(
                    "net_arch", {"pi": [256, 256, 256], "qf": [256, 256, 256]}
                ),
                n_critics=int(hparams.get("n_critics", 2)),
                critic_subsample_size=hparams.get("critic_subsample_size"),
                critic_impl=hparams.get("critic_impl", "vmap"),
                actor_use_layer_norm=bool(
                    hparams.get("actor_use_layer_norm", False)
                ),
                critic_use_layer_norm=bool(
                    hparams.get("critic_use_layer_norm", False)
                ),
                log_std_mode=hparams.get("actor_log_std_mode", "clamp"),
                log_std_min=float(hparams.get("actor_log_std_min", -5.0)),
                residual_actor_zero_init=bool(
                    hparams.get("residual_actor_zero_init", True)
                ),
                residual_log_std_init=float(
                    hparams.get("residual_log_std_init", -3.0)
                ),
            ).to(self.device)

        source = checkpoint["state"]["policy"]
        target = policy.state_dict()
        prefixes = ("features_extractor.", "actor.", "_actor_adapter.")
        source_actor = {
            key: value for key, value in source.items() if key.startswith(prefixes)
        }
        target_actor = {
            key: value for key, value in target.items() if key.startswith(prefixes)
        }
        if set(source_actor) != set(target_actor):
            missing = sorted(set(target_actor) - set(source_actor))
            extra = sorted(set(source_actor) - set(target_actor))
            raise ValueError(
                "Warmup policy actor/encoder keys do not match: "
                f"missing={missing}, extra={extra}."
            )
        mismatched = [
            key
            for key in target_actor
            if tuple(source_actor[key].shape) != tuple(target_actor[key].shape)
        ]
        if mismatched:
            raise ValueError(
                "Warmup policy actor/encoder tensor shapes do not match: "
                + ", ".join(mismatched)
            )
        target.update(source_actor)
        policy.load_state_dict(target, strict=True)
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        return policy

    def _build_replay_buffer(self):
        return self._make_residual_replay_buffer(self.buffer_size)

    def _make_residual_replay_buffer(
        self, buffer_size: int, *, num_envs: Optional[int] = None
    ):
        num_envs = self.num_envs if num_envs is None else int(num_envs)
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Dict):
            return ResidualDictReplayBuffer(
                observation_space=obs_space,
                action_space=self._residual_action_space,
                num_envs=num_envs,
                buffer_size=buffer_size,
                storage_device=self.buffer_device,
                sample_device=self.device,
            )
        return ResidualTensorReplayBuffer(
            observation_space=obs_space,
            action_space=self._residual_action_space,
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
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            log_std_min=self.actor_log_std_min,
            log_std_mode=self.actor_log_std_mode,
            residual_actor_zero_init=self.residual_actor_zero_init,
            residual_log_std_init=self.residual_log_std_init,
        )

    def _setup_model(self) -> None:
        if self.action_scaler is None:
            self.action_scaler = ActionScaler.from_action_space(
                self.env.single_action_space, device=self.device
            )
        else:
            self.action_scaler = self.action_scaler.to(self.device)
        self.base_action_provider.to(self.device)
        super()._setup_model()

    def _call_base_action_provider(self, obs) -> torch.Tensor:
        output = self.base_action_provider.select_action(obs)
        return torch.as_tensor(output.actions, dtype=torch.float32, device=self.device)

    def _base_naction(self, obs) -> torch.Tensor:
        with torch.no_grad():
            policy_obs = self._obs_to_policy_device(obs)
            base_action = self._call_base_action_provider(policy_obs)
            return self.action_scaler.scale(base_action).clamp(-1.0, 1.0).detach()

    def _residual_scale_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        scales = torch.full_like(reference, self.residual_action_scale)
        if self.residual_gripper_action_scale is not None:
            scales[..., 6] = self.residual_gripper_action_scale
            scales[..., 13] = self.residual_gripper_action_scale
        return scales

    def _combine_base_residual(
        self, base_actions: torch.Tensor, unit_residual_actions: torch.Tensor
    ) -> torch.Tensor:
        residual_actions = (
            unit_residual_actions * self._residual_scale_tensor(unit_residual_actions)
        )
        return torch.clamp(base_actions + residual_actions, -1.0, 1.0)

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
        final_naction = self._combine_base_residual(base_actions, unit_residual)
        return final_naction, log_prob, features

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
        self._sample_warmup_policy_selection()

    def _sample_warmup_policy_selection(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> None:
        if self._warmup_policy is None:
            self._warmup_use_policy = None
            return
        if self._warmup_use_policy is None:
            self._warmup_use_policy = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        draws = torch.rand(env_ids.numel(), device=self.device)
        self._warmup_use_policy[env_ids] = (
            draws < self.residual_warmup_policy_probability
        )

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        if self._cached_base_actions is None:
            base_actions = self._base_naction(obs)
        else:
            base_actions = self._cached_base_actions
            self._cached_base_actions = None
        if not learning_has_started:
            if self._warmup_policy is not None:
                if self._warmup_use_policy is None:
                    self._sample_warmup_policy_selection()
                assert self._warmup_use_policy is not None
                with torch.no_grad():
                    checkpoint_residual = self._warmup_policy.predict(
                        {"state": self._obs_to_policy_device(obs)["state"]},
                        base_actions=base_actions,
                        deterministic=True,
                    )
                unit_residual = torch.where(
                    self._warmup_use_policy[:, None],
                    checkpoint_residual,
                    torch.zeros_like(checkpoint_residual),
                )
            else:
                shape = (self.num_envs,) + self.env.single_action_space.shape
                unit_residual = self.residual_warmup_scale * (
                    2 * torch.rand(shape, dtype=torch.float32, device=self.device)
                    - 1
                )
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
            done_ids = torch.where(done)[0]
            self._reset_base_action_provider(done_ids)
            self._cached_base_actions = None
            self._sample_warmup_policy_selection(done_ids)

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
            "residual_actions": (
                unit_residual * self._residual_scale_tensor(unit_residual)
            ),
            "final_actions": final_naction,
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
