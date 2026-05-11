"""Standalone CQL algorithm built on the SAC-family core."""
from __future__ import annotations

import warnings
from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.algorithms.sac_core import SACCore
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.sac_policy import CQLAlphaLagrange, SACPolicy, TemperatureLagrange


class CQLCore(SACCore):
    """Shared CQL setup and losses for online and offline algorithm shells."""

    _SUPPORTED_POLICY_KWARGS = frozenset({"features_extractor_class", "features_extractor_kwargs"})

    def _init_cql_params(
        self,
        *,
        tau: float,
        utd: float,
        policy_lr: float = 1e-4,
        q_lr: float = 3e-4,
        alpha_lr: float = 1e-4,
        cql_alpha_lr: float = 3e-4,
        policy_frequency: int = 1,
        target_network_frequency: int = 1,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        use_compile: bool = False,
        compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
        # Entropy
        ent_coef: float | str = "auto",
        target_entropy: float | str = "auto",
        backup_entropy: bool = False,
        # Network architecture
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        actor_use_layer_norm: bool = True,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[str] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        # Q-ensemble (REDQ)
        n_critics: int = 10,
        critic_subsample_size: Optional[int] = 2,
        # CQL parameters
        use_cql_loss: bool = True,
        cql_n_actions: int = 10,
        cql_alpha: float = 5.0,
        cql_autotune_alpha: bool = False,
        cql_alpha_lagrange_init: float = 1.0,
        cql_target_action_gap: float = 1.0,
        cql_importance_sample: bool = True,
        cql_max_target_backup: bool = True,
        cql_temp: float = 1.0,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        cql_action_sample_method: str = "uniform",
        # Phase control
        use_td_loss: bool = True,
        # General
        policy_kwargs: Optional[dict[str, Any]] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
    ) -> None:
        del seed, device, logger, std_log, log_freq
        self.tau = tau
        self.utd = utd
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.alpha_lr = alpha_lr
        self.cql_alpha_lr = cql_alpha_lr
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip_norm = grad_clip_norm
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive or None, got {grad_clip_norm}.")
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self._eager_critic_loss = None
        self._eager_actor_loss = None
        self._eager_target_q = None

        self.ent_coef_init = ent_coef
        self.target_entropy_arg = target_entropy
        self.backup_entropy = backup_entropy

        self.net_arch = self._resolve_net_arch(
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
        )
        self.actor_use_layer_norm = actor_use_layer_norm
        self.critic_use_layer_norm = critic_use_layer_norm
        self.actor_use_group_norm = actor_use_group_norm
        self.critic_use_group_norm = critic_use_group_norm
        self.num_groups = num_groups
        self.actor_dropout_rate = actor_dropout_rate
        self.critic_dropout_rate = critic_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.std_parameterization = std_parameterization
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size

        self.use_td_loss = use_td_loss
        self.use_cql_loss = use_cql_loss
        self.use_calql = False
        self.cql_n_actions = cql_n_actions
        self.cql_alpha = cql_alpha
        self.cql_autotune_alpha = cql_autotune_alpha
        self.cql_alpha_lagrange_init = cql_alpha_lagrange_init
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_importance_sample = cql_importance_sample
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_temp = cql_temp
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self.cql_action_sample_method = cql_action_sample_method

        self.policy_kwargs = self._normalize_policy_kwargs(policy_kwargs)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "tau": self.tau,
            "utd": self.utd,
            "policy_lr": self.policy_lr,
            "q_lr": self.q_lr,
            "alpha_lr": self.alpha_lr,
            "cql_alpha_lr": self.cql_alpha_lr,
            "policy_frequency": self.policy_frequency,
            "target_network_frequency": self.target_network_frequency,
            "weight_decay": self.weight_decay,
            "use_adamw": self.use_adamw,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "grad_clip_norm": self.grad_clip_norm,
            "use_compile": self.use_compile,
            "compile_mode": self.compile_mode,
            "ent_coef": self.ent_coef_init,
            "target_entropy": self.target_entropy_arg,
            "target_entropy_value": self.target_entropy,
            "backup_entropy": self.backup_entropy,
            "net_arch": self.net_arch,
            "actor_use_layer_norm": self.actor_use_layer_norm,
            "critic_use_layer_norm": self.critic_use_layer_norm,
            "actor_use_group_norm": self.actor_use_group_norm,
            "critic_use_group_norm": self.critic_use_group_norm,
            "num_groups": self.num_groups,
            "actor_dropout_rate": self.actor_dropout_rate,
            "critic_dropout_rate": self.critic_dropout_rate,
            "kernel_init": self.kernel_init,
            "backbone_type": self.backbone_type,
            "std_parameterization": self.std_parameterization,
            "n_critics": self.n_critics,
            "critic_subsample_size": self.critic_subsample_size,
            "use_td_loss": self.use_td_loss,
            "use_cql_loss": self.use_cql_loss,
            "cql_n_actions": self.cql_n_actions,
            "cql_alpha": self.cql_alpha,
            "cql_autotune_alpha": self.cql_autotune_alpha,
            "cql_alpha_lagrange_init": self.cql_alpha_lagrange_init,
            "cql_target_action_gap": self.cql_target_action_gap,
            "cql_importance_sample": self.cql_importance_sample,
            "cql_max_target_backup": self.cql_max_target_backup,
            "cql_temp": self.cql_temp,
            "cql_clip_diff_min": self.cql_clip_diff_min,
            "cql_clip_diff_max": self.cql_clip_diff_max,
            "cql_action_sample_method": self.cql_action_sample_method,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "autotune": self.autotune,
            "target_entropy": self.target_entropy,
            "use_cql_loss": self.use_cql_loss,
            "cql_alpha": self.cql_alpha,
            "use_td_loss": self.use_td_loss,
        }
        if self.autotune:
            state["temperature_lagrange"] = self.temperature_lagrange.state_dict()
        else:
            state["fixed_alpha"] = self._fixed_alpha.detach()
        if self.cql_autotune_alpha:
            state["cql_alpha_lagrange"] = self.cql_alpha_lagrange.state_dict()
        state["lr_scheduler_states"] = [
            sched.state_dict() if sched is not None else None
            for sched in self._lr_schedulers
        ]
        return state

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        if "target_entropy" in state:
            self.target_entropy = float(state["target_entropy"])
        if "use_cql_loss" in state:
            self.use_cql_loss = bool(state["use_cql_loss"])
        if "cql_alpha" in state:
            self.cql_alpha = float(state["cql_alpha"])
        if "use_td_loss" in state:
            self.use_td_loss = bool(state["use_td_loss"])
        if self.autotune and "temperature_lagrange" in state:
            self.temperature_lagrange.load_state_dict(state["temperature_lagrange"])
        elif not self.autotune and "fixed_alpha" in state:
            self._fixed_alpha = state["fixed_alpha"].to(self.device)
        if self.cql_autotune_alpha and "cql_alpha_lagrange" in state:
            self.cql_alpha_lagrange.load_state_dict(state["cql_alpha_lagrange"])
        if "lr_scheduler_states" in state:
            for sched, sched_state in zip(self._lr_schedulers, state["lr_scheduler_states"]):
                if sched is not None and sched_state is not None:
                    sched.load_state_dict(sched_state)

    def _default_features_extractor_class(self) -> type[BaseFeaturesExtractor]:
        assert isinstance(self.env.single_observation_space, spaces.Box), (
            "CQL expects a flat Box observation space; use a vision-specific subclass "
            "for dict observations."
        )
        return FlattenExtractor

    def _default_features_extractor_kwargs(self) -> dict[str, Any]:
        return {}

    def _normalize_policy_kwargs(
        self, policy_kwargs: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        normalized = dict(policy_kwargs or {})
        unknown_keys = sorted(set(normalized) - self._SUPPORTED_POLICY_KWARGS)
        if unknown_keys:
            raise ValueError(
                "Unsupported policy_kwargs keys: "
                + ", ".join(unknown_keys)
                + ". Supported keys are: features_extractor_class, "
                + "features_extractor_kwargs."
            )

        features_extractor_kwargs = normalized.get("features_extractor_kwargs", {})
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        if not isinstance(features_extractor_kwargs, dict):
            raise TypeError("policy_kwargs['features_extractor_kwargs'] must be a dict.")

        normalized["features_extractor_kwargs"] = dict(features_extractor_kwargs)
        return normalized

    def _resolve_policy_kwargs(self) -> dict[str, Any]:
        default_features_extractor_class = self._default_features_extractor_class()
        default_features_extractor_kwargs = dict(self._default_features_extractor_kwargs())
        resolved = {
            "features_extractor_class": default_features_extractor_class,
            "features_extractor_kwargs": default_features_extractor_kwargs,
        }

        if "features_extractor_class" in self.policy_kwargs:
            resolved["features_extractor_class"] = self.policy_kwargs["features_extractor_class"]
            if resolved["features_extractor_class"] is not default_features_extractor_class:
                resolved["features_extractor_kwargs"] = {}
        if "features_extractor_kwargs" in self.policy_kwargs:
            if resolved["features_extractor_class"] is default_features_extractor_class:
                resolved["features_extractor_kwargs"] = {
                    **resolved["features_extractor_kwargs"],
                    **self.policy_kwargs["features_extractor_kwargs"],
                }
            else:
                resolved["features_extractor_kwargs"] = dict(
                    self.policy_kwargs["features_extractor_kwargs"]
                )
        return resolved

    def _build_features_extractor(self) -> BaseFeaturesExtractor:
        resolved = self._resolve_policy_kwargs()
        features_extractor_class = resolved["features_extractor_class"]
        if not isinstance(features_extractor_class, type) or not issubclass(
            features_extractor_class, BaseFeaturesExtractor
        ):
            raise TypeError(
                "policy_kwargs['features_extractor_class'] must be a "
                "BaseFeaturesExtractor subclass."
            )
        return features_extractor_class(
            observation_space=self.env.single_observation_space,
            **resolved["features_extractor_kwargs"],
        )

    @staticmethod
    def _resolve_net_arch(
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]],
        actor_hidden_dims: Optional[Sequence[int]],
        critic_hidden_dims: Optional[Sequence[int]],
    ) -> Sequence[int] | dict[str, list[int]]:
        if net_arch is not None:
            if actor_hidden_dims is not None or critic_hidden_dims is not None:
                warnings.warn(
                    "actor_hidden_dims/critic_hidden_dims are deprecated and ignored "
                    "when net_arch is provided. Use net_arch only.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if isinstance(net_arch, dict):
                if "pi" not in net_arch or "qf" not in net_arch:
                    raise ValueError("net_arch dict must contain both 'pi' and 'qf' keys.")
                return {"pi": list(net_arch["pi"]), "qf": list(net_arch["qf"])}
            return list(net_arch)

        if actor_hidden_dims is not None or critic_hidden_dims is not None:
            warnings.warn(
                "actor_hidden_dims/critic_hidden_dims are deprecated. "
                "Use net_arch=list[...] or net_arch={'pi': [...], 'qf': [...]} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            pi_arch = list(actor_hidden_dims) if actor_hidden_dims is not None else [256, 256]
            qf_arch = list(critic_hidden_dims) if critic_hidden_dims is not None else list(pi_arch)
            return {"pi": pi_arch, "qf": qf_arch}

        return {"pi": [256, 256], "qf": [256, 256]}

    def _build_replay_buffer(self):
        return TensorReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _setup_model(self) -> None:
        features_extractor = self._build_features_extractor()
        self.policy = SACPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            actor_use_group_norm=self.actor_use_group_norm,
            critic_use_group_norm=self.critic_use_group_norm,
            num_groups=self.num_groups,
            actor_dropout_rate=self.actor_dropout_rate,
            critic_dropout_rate=self.critic_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            std_parameterization=self.std_parameterization,
            log_std_mode="clamp",
            log_std_min=-20.0,
        ).to(self.device)

        self.q_optimizer = make_optimizer(
            list(self.policy.critic_and_encoder_parameters()),
            lr=self.q_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.actor_optimizer = make_optimizer(
            list(self.policy.actor_parameters()),
            lr=self.policy_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )

        if self.cql_autotune_alpha:
            self.cql_alpha_lagrange = CQLAlphaLagrange(
                init_value=self.cql_alpha_lagrange_init
            ).to(self.device)
            self.cql_alpha_optimizer = make_optimizer(
                list(self.cql_alpha_lagrange.parameters()),
                lr=self.cql_alpha_lr,
                weight_decay=0.0,
                use_adamw=self.use_adamw,
            )
        else:
            self.cql_alpha_lagrange = None
            self.cql_alpha_optimizer = None

        self.autotune = isinstance(self.ent_coef_init, str) and self.ent_coef_init.startswith(
            "auto"
        )
        if self.autotune:
            init = 1.0
            if isinstance(self.ent_coef_init, str) and "_" in self.ent_coef_init:
                init = float(self.ent_coef_init.split("_")[1])
            self.temperature_lagrange = TemperatureLagrange(init_value=init).to(self.device)
            self.alpha_optimizer = make_optimizer(
                list(self.temperature_lagrange.parameters()),
                lr=self.alpha_lr,
                weight_decay=0.0,
                use_adamw=self.use_adamw,
            )
        else:
            self.temperature_lagrange = None
            self.alpha_optimizer = None
            self._fixed_alpha = torch.tensor(float(self.ent_coef_init), device=self.device)

        if self.target_entropy_arg == "auto":
            self.target_entropy = float(
                -np.prod(self.env.single_action_space.shape).astype(np.float32)
            )
        else:
            self.target_entropy = float(self.target_entropy_arg)

        self.replay_buffer = self._build_replay_buffer()
        self._lr_schedulers: list[Optional[Any]] = []
        for opt in (self.q_optimizer, self.actor_optimizer):
            self._lr_schedulers.append(
                make_lr_scheduler(
                    opt,
                    schedule_type=self.lr_schedule,
                    warmup_steps=self.lr_warmup_steps,
                    decay_steps=self.lr_decay_steps,
                    min_lr_ratio=self.lr_min_ratio,
                )
            )

        if self.use_compile:
            self._apply_compile()

    def _apply_compile(self) -> None:
        if self._eager_critic_loss is None:
            self._eager_critic_loss = self._critic_loss
            self._eager_actor_loss = self._actor_loss
            self._eager_target_q = self._target_q
        self._critic_loss = torch.compile(self._eager_critic_loss, mode=self.compile_mode)
        self._actor_loss = torch.compile(self._eager_actor_loss, mode=self.compile_mode)
        self._target_q = torch.compile(self._eager_target_q, mode=self.compile_mode)

    def _step_critic_scheduler(self) -> None:
        sched = self._lr_schedulers[0] if self._lr_schedulers else None
        if sched is not None:
            sched.step()

    def _step_actor_scheduler(self) -> None:
        sched = self._lr_schedulers[1] if len(self._lr_schedulers) >= 2 else None
        if sched is not None:
            sched.step()

    def _clip_grad_norm(self, params) -> None:
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(params), self.grad_clip_norm)

    def _current_alpha(self) -> torch.Tensor:
        if self.autotune:
            return self.temperature_lagrange()
        return self._fixed_alpha

    def _current_cql_alpha(self) -> torch.Tensor:
        if self.cql_autotune_alpha:
            return self.cql_alpha_lagrange()
        return torch.tensor(self.cql_alpha, device=self.device)

    def _sample_n_actions_with_log_probs(
        self, obs, n: int, features: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features is None:
            features = self.policy.extract_features(obs)
        actor = self.policy.actor
        mean, log_std = actor(features)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample(sample_shape=(n,))
        y_t = torch.tanh(x_t)
        action = y_t * actor.action_scale + actor.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1)
        return action.permute(1, 0, 2), log_prob.permute(1, 0)

    def _sample_random_actions(self, batch_size: int, action_dim: int) -> torch.Tensor:
        if self.cql_action_sample_method == "uniform":
            return (
                torch.rand(
                    batch_size, self.cql_n_actions, action_dim, device=self.device
                )
                * 2.0
                - 1.0
            )
        if self.cql_action_sample_method == "normal":
            return torch.randn(
                batch_size, self.cql_n_actions, action_dim, device=self.device
            )
        raise NotImplementedError(
            f"Unknown cql_action_sample_method: {self.cql_action_sample_method}"
        )

    def _calql_lower_bound(
        self,
        q_ood: torch.Tensor,
        mc_returns: torch.Tensor,
        n_samples: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, float]:
        del q_ood, mc_returns, n_samples, batch_size
        raise RuntimeError("Cal-QL lower bounds require the CalQL algorithm class.")

    def _cql_regularizer(self, data, q_pred: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size = data.rewards.shape[0]
        action_dim = self.env.single_action_space.shape[0]
        info: dict[str, float] = {}

        cql_random_actions = self._sample_random_actions(batch_size, action_dim)
        cql_current_actions, cql_current_log_pis = self._sample_n_actions_with_log_probs(
            data.obs, self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self._sample_n_actions_with_log_probs(
            data.next_obs, self.cql_n_actions
        )
        all_sampled_actions = torch.cat(
            [cql_random_actions, cql_current_actions, cql_next_actions], dim=1
        )

        features = self.policy.extract_features(data.obs)
        feat_dim = features.shape[-1]
        n_samples = 3 * self.cql_n_actions
        features_repeated = (
            features.unsqueeze(1)
            .expand(batch_size, n_samples, feat_dim)
            .reshape(-1, feat_dim)
        )
        actions_flat = all_sampled_actions.reshape(-1, action_dim)
        q_ood = self.policy.q_values_all(features_repeated, actions_flat, target=False)
        q_ood = q_ood.reshape(self.n_critics, batch_size, n_samples)

        if (
            self.critic_subsample_size is not None
            and self.critic_subsample_size < self.n_critics
        ):
            idx = torch.randint(
                0, self.n_critics, (self.critic_subsample_size,), device=self.device
            )
            q_ood = q_ood[idx]
            q_pred_for_diff = q_pred[idx].squeeze(-1)
        else:
            q_pred_for_diff = q_pred.squeeze(-1)

        if self.use_calql and getattr(data, "mc_returns", None) is not None:
            q_ood, bound_rate = self._calql_lower_bound(
                q_ood, data.mc_returns, n_samples, batch_size
            )
            info["calql_bound_rate"] = bound_rate

        if self.cql_importance_sample:
            random_density = float(np.log(0.5**action_dim))
            random_log = torch.full(
                (batch_size, self.cql_n_actions),
                random_density,
                device=self.device,
                dtype=q_ood.dtype,
            )
            importance_log = torch.cat(
                [random_log, cql_current_log_pis, cql_next_log_pis], dim=1
            )
            q_ood = q_ood - importance_log.unsqueeze(0)
        else:
            q_ood = torch.cat([q_ood, q_pred_for_diff.unsqueeze(-1)], dim=-1)
            q_ood = q_ood - float(np.log(q_ood.shape[-1])) * self.cql_temp

        cql_ood_values = torch.logsumexp(q_ood / self.cql_temp, dim=-1) * self.cql_temp
        cql_q_diff = cql_ood_values - q_pred_for_diff
        if not self.cql_autotune_alpha:
            cql_q_diff = torch.clamp(
                cql_q_diff, self.cql_clip_diff_min, self.cql_clip_diff_max
            )

        cql_loss_raw = cql_q_diff.mean()
        info.update(
            {
                "cql_q_diff": cql_q_diff.mean().item(),
                "cql_ood_values": cql_ood_values.mean().item(),
            }
        )
        return cql_loss_raw, info

    def _cql_loss(self, data, q_pred: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        return self._cql_regularizer(data, q_pred)

    def _target_q(self, data) -> torch.Tensor:
        alpha = self._current_alpha().detach()
        with torch.no_grad():
            if self.cql_max_target_backup and self.use_cql_loss:
                next_actions, next_log_probs = self._sample_n_actions_with_log_probs(
                    data.next_obs, self.cql_n_actions
                )
                batch_size = next_actions.shape[0]
                next_features = self.policy.extract_features(data.next_obs)
                feat_dim = next_features.shape[-1]
                action_dim = next_actions.shape[-1]
                features_repeated = (
                    next_features.unsqueeze(1)
                    .expand(batch_size, self.cql_n_actions, feat_dim)
                    .reshape(-1, feat_dim)
                )
                actions_flat = next_actions.reshape(-1, action_dim)
                q_next_sub = self.policy.q_values_subsampled(
                    features_repeated,
                    actions_flat,
                    subsample_size=self.critic_subsample_size,
                    target=True,
                )
                critic_size = q_next_sub.shape[0]
                q_next_sub = q_next_sub.reshape(critic_size, batch_size, self.cql_n_actions)
                q_next_min = q_next_sub.min(dim=0).values
                max_idx = q_next_min.argmax(dim=1, keepdim=True)
                min_q_next = q_next_min.gather(1, max_idx)
                next_log_prob = next_log_probs.gather(1, max_idx)
            else:
                next_action, next_log_prob, next_features = self.policy.actor_action_log_prob(
                    data.next_obs, stop_gradient=False
                )
                min_q_next = self.policy.min_q_value(
                    next_features,
                    next_action,
                    subsample_size=self.critic_subsample_size,
                    target=True,
                )

            if self.backup_entropy:
                min_q_next = min_q_next - alpha * next_log_prob

            return data.rewards.reshape(-1, 1) + (
                1 - data.dones.reshape(-1, 1)
            ) * self.gamma * min_q_next

    def _td_loss(self, data, q_pred: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        target_q = self._target_q(data)
        target_q_expanded = target_q.unsqueeze(0).repeat(self.n_critics, 1, 1)
        td_loss = F.mse_loss(q_pred, target_q_expanded)
        return td_loss, {"td_loss": td_loss.item(), "target_q": target_q.mean().item()}

    def _critic_loss(self, data) -> tuple[torch.Tensor, dict[str, float]]:
        info: dict[str, float] = {}
        q_pred = self._critic_forward(data.obs, data.actions, target=False)

        td_loss = torch.tensor(0.0, device=self.device)
        if self.use_td_loss:
            td_loss, td_info = self._td_loss(data, q_pred)
            info.update(td_info)

        cql_loss = torch.tensor(0.0, device=self.device)
        if self.use_cql_loss:
            cql_loss_raw, cql_info = self._cql_regularizer(data, q_pred)
            info.update(cql_info)
            cql_alpha = self._current_cql_alpha()
            cql_loss = cql_loss_raw - self.cql_target_action_gap if self.cql_autotune_alpha else cql_loss_raw
            cql_loss = cql_alpha * cql_loss
            info["cql_loss"] = cql_loss_raw.item()
            info["cql_alpha"] = cql_alpha.item()

        critic_loss = td_loss + cql_loss
        info["critic_loss"] = critic_loss.item()
        info["predicted_q"] = q_pred.mean().item()
        return critic_loss, info

    def _actor_loss(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._current_alpha().detach()
        action, log_prob, features = self.policy.actor_action_log_prob(
            obs, stop_gradient=False
        )
        min_q = self.policy.min_q_value(
            features, action, subsample_size=None, target=False
        )
        return (alpha * log_prob - min_q).mean(), log_prob.detach()

    def _cql_alpha_loss(self, data) -> torch.Tensor:
        with torch.no_grad():
            q_pred = self._critic_forward(data.obs, data.actions, target=False)
            cql_loss_raw, _ = self._cql_loss(data, q_pred)
        cql_alpha = self._current_cql_alpha()
        return -cql_alpha * (cql_loss_raw - self.cql_target_action_gap)

    def _backup_entropy_enabled(self) -> bool:
        return self.backup_entropy

    def _post_actor_update(self, data) -> dict[str, float]:
        info: dict[str, float] = {}
        if self.cql_autotune_alpha:
            cql_alpha_loss = self._cql_alpha_loss(data)
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward()
            self._clip_grad_norm(self.cql_alpha_lagrange.parameters())
            self.cql_alpha_optimizer.step()
            info["cql_alpha_loss"] = cql_alpha_loss.item()
        if self.cql_autotune_alpha or self.use_cql_loss:
            info["cql_alpha"] = self._current_cql_alpha().item()
        return info


class CQL(CQLCore, OffPolicyAlgorithm):
    """CQL with SAC/REDQ updates and conservative critic regularization."""

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        # Buffer and training
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        training_freq: int = 64,
        utd: float = 1.0,
        bootstrap_at_done: str = "always",
        # Optimizers
        policy_lr: float = 1e-4,
        q_lr: float = 3e-4,
        alpha_lr: float = 1e-4,
        cql_alpha_lr: float = 3e-4,
        policy_frequency: int = 1,
        target_network_frequency: int = 1,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        use_compile: bool = False,
        compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
        # Entropy
        ent_coef: float | str = "auto",
        target_entropy: float | str = "auto",
        backup_entropy: bool = False,
        # Network architecture
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        actor_use_layer_norm: bool = True,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[str] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        # Q-ensemble (REDQ)
        n_critics: int = 10,
        critic_subsample_size: Optional[int] = 2,
        # CQL parameters
        use_cql_loss: bool = True,
        cql_n_actions: int = 10,
        cql_alpha: float = 5.0,
        cql_autotune_alpha: bool = False,
        cql_alpha_lagrange_init: float = 1.0,
        cql_target_action_gap: float = 1.0,
        cql_importance_sample: bool = True,
        cql_max_target_backup: bool = True,
        cql_temp: float = 1.0,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        cql_action_sample_method: str = "uniform",
        # Phase control
        use_td_loss: bool = True,
        # General
        policy_kwargs: Optional[dict[str, Any]] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            eval_env=eval_env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            training_freq=training_freq,
            utd=utd,
            bootstrap_at_done=bootstrap_at_done,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
        )
        self._init_cql_params(
            tau=tau,
            utd=utd,
            policy_lr=policy_lr,
            q_lr=q_lr,
            alpha_lr=alpha_lr,
            cql_alpha_lr=cql_alpha_lr,
            policy_frequency=policy_frequency,
            target_network_frequency=target_network_frequency,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            use_compile=use_compile,
            compile_mode=compile_mode,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            use_cql_loss=use_cql_loss,
            cql_n_actions=cql_n_actions,
            cql_alpha=cql_alpha,
            cql_autotune_alpha=cql_autotune_alpha,
            cql_alpha_lagrange_init=cql_alpha_lagrange_init,
            cql_target_action_gap=cql_target_action_gap,
            cql_importance_sample=cql_importance_sample,
            cql_max_target_backup=cql_max_target_backup,
            cql_temp=cql_temp,
            cql_clip_diff_min=cql_clip_diff_min,
            cql_clip_diff_max=cql_clip_diff_max,
            cql_action_sample_method=cql_action_sample_method,
            use_td_loss=use_td_loss,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
        )
        self._setup_model()


class OfflineCQL(CQLCore, OfflineRLAlgorithm):
    """Pure offline CQL over a static replay buffer."""

    def __init__(
        self,
        env: OfflineEnvSpec,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        offline_sampling: str = "with_replace",
        utd: float = 1.0,
        policy_lr: float = 1e-4,
        q_lr: float = 3e-4,
        alpha_lr: float = 1e-4,
        cql_alpha_lr: float = 3e-4,
        policy_frequency: int = 1,
        target_network_frequency: int = 1,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        use_compile: bool = False,
        compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
        ent_coef: float | str = "auto",
        target_entropy: float | str = "auto",
        backup_entropy: bool = False,
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        actor_use_layer_norm: bool = True,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[str] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        n_critics: int = 10,
        critic_subsample_size: Optional[int] = 2,
        use_cql_loss: bool = True,
        cql_n_actions: int = 10,
        cql_alpha: float = 5.0,
        cql_autotune_alpha: bool = False,
        cql_alpha_lagrange_init: float = 1.0,
        cql_target_action_gap: float = 1.0,
        cql_importance_sample: bool = True,
        cql_max_target_backup: bool = True,
        cql_temp: float = 1.0,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        cql_action_sample_method: str = "uniform",
        use_td_loss: bool = True,
        policy_kwargs: Optional[dict[str, Any]] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            batch_size=batch_size,
            gamma=gamma,
            offline_sampling=offline_sampling,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
        )
        self._init_cql_params(
            tau=tau,
            utd=utd,
            policy_lr=policy_lr,
            q_lr=q_lr,
            alpha_lr=alpha_lr,
            cql_alpha_lr=cql_alpha_lr,
            policy_frequency=policy_frequency,
            target_network_frequency=target_network_frequency,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            use_compile=use_compile,
            compile_mode=compile_mode,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            use_cql_loss=use_cql_loss,
            cql_n_actions=cql_n_actions,
            cql_alpha=cql_alpha,
            cql_autotune_alpha=cql_autotune_alpha,
            cql_alpha_lagrange_init=cql_alpha_lagrange_init,
            cql_target_action_gap=cql_target_action_gap,
            cql_importance_sample=cql_importance_sample,
            cql_max_target_backup=cql_max_target_backup,
            cql_temp=cql_temp,
            cql_clip_diff_min=cql_clip_diff_min,
            cql_clip_diff_max=cql_clip_diff_max,
            cql_action_sample_method=cql_action_sample_method,
            use_td_loss=use_td_loss,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
        )
        self._setup_model()

    def _sample_train_batch(self, batch_size: int):
        if self.offline_sampling == "without_replace":
            return self.replay_buffer.sample_without_repeat(batch_size)
        return self.replay_buffer.sample(batch_size)
