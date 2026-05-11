"""WSRL (Warm-Start RL) algorithm with CQL and Cal-QL support.

Implements offline→online training:
- Offline phase: Cal-QL (Conservative Q-Learning with calibration)
- Online phase: SAC or CQL (configurable)

Key features:
- Q-ensemble (REDQ) with 10 critics by default
- CQL regularization to prevent Q-value overestimation
- Cal-QL lower bounds using Monte Carlo returns
- Seamless offline→online mode switching
- High-UTD training support

Based on:
- WSRL paper: https://arxiv.org/abs/2412.07762
- Cal-QL paper: https://arxiv.org/abs/2303.05479
- CQL paper: https://arxiv.org/abs/2006.04779
"""
from __future__ import annotations

import warnings
from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.buffers.mc_buffer import MCReplayBufferSample, MCTensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.wsrl_policy import WSRLPolicy, TemperatureLagrange



class WSRL(OffPolicyAlgorithm):
    """WSRL algorithm with CQL/Cal-QL for offline→online training.

    Supports:
    - Offline pre-training with Cal-QL
    - Online fine-tuning with SAC or CQL
    - Q-ensemble (REDQ) with configurable size
    - High-UTD training
    """

    _SUPPORTED_POLICY_KWARGS = frozenset({"features_extractor_class", "features_extractor_kwargs"})

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
        # NOTE: "reduce-overhead" enables CUDA graphs which conflict with our
        # separately-compiled critic/actor methods (tensors get overwritten
        # between callable boundaries). Default to "default" inductor mode,
        # which still gives kernel fusion without CUDA-graph constraints.
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
        # Cal-QL parameters
        use_calql: bool = True,
        calql_bound_random_actions: bool = False,
        # Phase control
        use_td_loss: bool = True,
        online_cql_alpha: Optional[float] = None,
        online_use_cql_loss: Optional[bool] = None,
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
        # Sparse-reward MC (for antmaze/adroit-style envs)
        sparse_reward_mc: bool = False,
        sparse_negative_reward: float = 0.0,
        success_threshold: float = 0.5,
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

        # Optimizers
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
        # Track the un-compiled originals so we can re-wrap after mode switches.
        self._eager_critic_loss = None
        self._eager_actor_loss = None
        self._eager_target_q = None

        # Entropy
        self.ent_coef_init = ent_coef
        self.target_entropy_arg = target_entropy
        self.backup_entropy = backup_entropy

        # Network architecture
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

        # CQL parameters
        self.use_cql_loss = use_cql_loss
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

        # Cal-QL parameters
        self.use_calql = use_calql
        self.calql_bound_random_actions = calql_bound_random_actions

        # Phase control
        self.use_td_loss = use_td_loss
        self.online_cql_alpha = online_cql_alpha
        self.online_use_cql_loss = online_use_cql_loss
        self._online_start_step: int | None = None
        self._offline_probe_batch: MCReplayBufferSample | None = None

        # Sparse-reward MC config
        self.sparse_reward_mc = sparse_reward_mc
        self.sparse_negative_reward = sparse_negative_reward
        self.success_threshold = success_threshold

        # Offline-phase sampling strategy. Online phase always uses with-replacement.
        self.offline_sampling: Literal["with_replace", "without_replace"] = offline_sampling

        # Mixed-batch online sampling (set by switch_to_online_mode("mixed", ...)).
        self.offline_replay_buffer: Optional[Any] = None
        self.offline_data_ratio: float = 0.0

        self.policy_kwargs = self._normalize_policy_kwargs(policy_kwargs)
        self._setup_model()

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
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
            "use_calql": self.use_calql,
            "calql_bound_random_actions": self.calql_bound_random_actions,
            "use_td_loss": self.use_td_loss,
            "online_cql_alpha": self.online_cql_alpha,
            "online_use_cql_loss": self.online_use_cql_loss,
            "sparse_reward_mc": self.sparse_reward_mc,
            "sparse_negative_reward": self.sparse_negative_reward,
            "success_threshold": self.success_threshold,
            "offline_sampling": self.offline_sampling,
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
        # LR scheduler state (per scheduler index; None if no scheduler).
        sched_states: list[Optional[dict]] = []
        for sched in self._lr_schedulers:
            sched_states.append(sched.state_dict() if sched is not None else None)
        state["lr_scheduler_states"] = sched_states
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
        if "lr_scheduler_states" in state:
            for sched, sched_state in zip(self._lr_schedulers, state["lr_scheduler_states"]):
                if sched is not None and sched_state is not None:
                    sched.load_state_dict(sched_state)

    # --- Construction hooks (WSRLRGBD overrides defaults only) ---

    def _default_features_extractor_class(self) -> type[BaseFeaturesExtractor]:
        assert isinstance(self.env.single_observation_space, spaces.Box), (
            "WSRL base class expects a flat Box observation space; "
            "use WSRLRGBD for dict observations."
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

    def _clear_replay_buffer(self) -> int:
        previous_len = len(self.replay_buffer)
        self.replay_buffer.pos = 0
        self.replay_buffer.full = False
        if hasattr(self.replay_buffer, "_mc_table"):
            self.replay_buffer._mc_table = None
        return previous_len

    def _sample_batch(self, batch_size: int) -> MCReplayBufferSample:
        """Dispatch to with/without-replacement sampling based on phase + config.

        Online phase (after ``switch_to_online_mode``) uses with-replacement and
        optionally a mixed-batch composed of online + offline samples when
        ``offline_replay_buffer`` is set.
        Offline phase honors ``self.offline_sampling``.
        """
        in_offline_phase = self._online_start_step is None
        if in_offline_phase and self.offline_sampling == "without_replace":
            return self.replay_buffer.sample_without_repeat(batch_size)

        # Online mixed-batch: combine online (current buffer) + offline buffer samples.
        if (
            not in_offline_phase
            and self.offline_replay_buffer is not None
            and self.offline_data_ratio > 0.0
        ):
            return self._sample_mixed_batch(batch_size)

        return self.replay_buffer.sample(batch_size)

    def _sample_mixed_batch(self, batch_size: int) -> MCReplayBufferSample:
        """Sample ``(1-r)*B`` from online + ``r*B`` from offline and concatenate."""
        ratio = self.offline_data_ratio
        n_online = batch_size - int(round(batch_size * ratio))
        online_size = len(self.replay_buffer)
        if online_size == 0:
            # Online buffer empty (e.g., before learning_starts collected enough);
            # fall back to all-offline for this batch.
            return self.offline_replay_buffer.sample(batch_size)
        n_offline = batch_size - n_online
        if n_online == 0:
            return self.offline_replay_buffer.sample(batch_size)
        if n_offline == 0:
            return self.replay_buffer.sample(batch_size)

        online_sample = self.replay_buffer.sample(n_online)
        offline_sample = self.offline_replay_buffer.sample(n_offline)
        return self._concat_replay_samples(online_sample, offline_sample)

    @staticmethod
    def _concat_replay_samples(
        a: MCReplayBufferSample, b: MCReplayBufferSample
    ) -> MCReplayBufferSample:
        def _cat(x, y):
            if isinstance(x, dict):
                return {k: torch.cat([x[k], y[k]], dim=0) for k in x}
            return torch.cat([x, y], dim=0)

        return MCReplayBufferSample(
            obs=_cat(a.obs, b.obs),
            next_obs=_cat(a.next_obs, b.next_obs),
            actions=_cat(a.actions, b.actions),
            rewards=_cat(a.rewards, b.rewards),
            dones=_cat(a.dones, b.dones),
            mc_returns=_cat(a.mc_returns, b.mc_returns),
        )

    def _build_replay_buffer(self):
        return MCTensorReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            storage_device=self.buffer_device,
            sample_device=self.device,
            sparse_reward_mc=self.sparse_reward_mc,
            sparse_negative_reward=self.sparse_negative_reward,
            success_threshold=self.success_threshold,
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
                return {
                    "pi": list(net_arch["pi"]),
                    "qf": list(net_arch["qf"]),
                }
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

    def _setup_model(self) -> None:
        features_extractor = self._build_features_extractor()
        self.policy = WSRLPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
            use_cql_alpha_lagrange=self.cql_autotune_alpha,
            cql_alpha_lagrange_init=self.cql_alpha_lagrange_init,
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

        # CQL alpha Lagrange multiplier optimizer (no weight decay on scalars).
        if self.cql_autotune_alpha:
            self.cql_alpha_optimizer = make_optimizer(
                list(self.policy.cql_alpha_lagrange_parameters()),
                lr=self.cql_alpha_lr,
                weight_decay=0.0,
                use_adamw=self.use_adamw,
            )
        else:
            self.cql_alpha_optimizer = None

        # Entropy coefficient (auto-tuned by default)
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

        # LR schedulers (one per actor/critic optimizer, sharing schedule config).
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

        # Optional torch.compile on the hot loss methods. We stash the eager
        # versions so ``switch_to_online_mode`` can re-wrap after Python-side
        # flag changes (e.g., ``use_cql_loss`` flipping false on online switch
        # would otherwise leave a stale specialization in the compiled graph).
        if self.use_compile:
            self._apply_compile()

    def _apply_compile(self) -> None:
        """Compile (or re-compile) the three hot loss methods.

        Called from ``_setup_model`` and ``switch_to_online_mode``. Stashes
        eager originals on first call so re-wrap is idempotent.
        """
        if self._eager_critic_loss is None:
            self._eager_critic_loss = self._critic_loss
            self._eager_actor_loss = self._actor_loss
            self._eager_target_q = self._target_q
        self._critic_loss = torch.compile(self._eager_critic_loss, mode=self.compile_mode)
        self._actor_loss = torch.compile(self._eager_actor_loss, mode=self.compile_mode)
        self._target_q = torch.compile(self._eager_target_q, mode=self.compile_mode)

    # --- Helper methods ---

    def _current_alpha(self) -> torch.Tensor:
        if self.autotune:
            return self.temperature_lagrange()
        return self._fixed_alpha

    def _current_cql_alpha(self) -> torch.Tensor:
        if self.cql_autotune_alpha:
            return self.policy.get_cql_alpha()
        return torch.tensor(self.cql_alpha, device=self.device)

    @staticmethod
    def canonical_eval_metrics(metrics: dict[str, float]) -> dict[str, float]:
        """Add paper-style score curves while preserving raw eval metrics."""
        out = dict(metrics)
        success = metrics.get("success_at_end", metrics.get("success_once"))
        if success is not None:
            out["normalized_score"] = float(success) * 100.0
        return out

    @staticmethod
    def _update_metric_tags(metrics: dict[str, float]) -> dict[str, float]:
        tagged: dict[str, float] = {}
        loss_keys = {
            "critic_loss",
            "actor_loss",
            "td_loss",
            "cql_loss",
            "alpha_loss",
            "cql_alpha_loss",
        }
        for key, value in metrics.items():
            if key in loss_keys:
                tagged[f"losses/{key}"] = value
            elif key == "predicted_q":
                tagged["q/predicted"] = value
            elif key == "target_q":
                tagged["q/target"] = value
            elif key == "cql_ood_values":
                tagged["q/cql_ood"] = value
            elif key == "cql_q_diff":
                tagged["q/cql_diff"] = value
            elif key == "cql_alpha":
                tagged["cql/alpha"] = value
            elif key == "calql_bound_rate":
                tagged["cql/bound_rate"] = value
            elif key == "alpha":
                tagged["entropy/alpha"] = value
            elif key == "utd_ratio":
                tagged["train/utd_ratio"] = value
            else:
                tagged[f"losses/{key}"] = value

        if "td_loss" in metrics:
            tagged["q/td_rmse"] = float(np.sqrt(max(metrics["td_loss"], 0.0)))
        return tagged

    def set_offline_probe_batch(self, batch: MCReplayBufferSample | None) -> None:
        """Keep a fixed offline batch for no-grad online forgetting diagnostics."""
        self._offline_probe_batch = batch

    def _offline_probe_metrics(self) -> dict[str, float]:
        if self._offline_probe_batch is None:
            return {}
        with torch.no_grad():
            data = self._offline_probe_batch
            q_pred = self._critic_forward(data.obs, data.actions, target=False)
            target_q = self._target_q(data)
            target_q_expanded = target_q.unsqueeze(0).repeat(self.n_critics, 1, 1)
            td_mse = F.mse_loss(q_pred, target_q_expanded)
        return {
            "q/offline_probe/predicted": float(q_pred.mean().item()),
            "q/offline_probe/target": float(target_q.mean().item()),
            "q/offline_probe/td_rmse": float(torch.sqrt(td_mse).item()),
        }

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        for key, value in self.canonical_eval_metrics(metrics).items():
            self.logger.add_scalar(f"eval/{key}", value, step)

    def _log_rollout_metric(self, key: str, value: float, step: int) -> None:
        if self.logger is None:
            return
        self.logger.add_scalar(f"train/{key}", value, step)
        if key in {"success_at_end", "success_once"}:
            self.logger.add_scalar("train/normalized_score", value * 100.0, step)

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        for tag, value in self._update_metric_tags(metrics).items():
            self.logger.add_scalar(tag, value, step)

        online_start = self._online_start_step
        is_online = online_start is not None and step >= online_start
        offline_step = min(step, online_start) if online_start is not None else step
        online_step = max(0, step - online_start) if online_start is not None else 0
        self.logger.add_scalar("phase/is_online", float(is_online), step)
        self.logger.add_scalar("phase/offline_step", float(offline_step), step)
        self.logger.add_scalar("phase/online_step", float(online_step), step)

        if is_online:
            for tag, value in self._offline_probe_metrics().items():
                self.logger.add_scalar(tag, value, step)

    def switch_to_online_mode(
        self,
        online_replay_mode: Literal["empty", "append", "mixed"] = "append",
        offline_data_ratio: float = 0.0,
    ) -> None:
        """Switch from offline to online training mode.

        Args:
            online_replay_mode:
                - ``"empty"``: clear the current replay buffer before online rollout.
                - ``"append"``: keep the offline data in the online replay buffer.
                - ``"mixed"``: freeze current buffer as ``self.offline_replay_buffer``
                  and replace ``self.replay_buffer`` with a fresh empty MC buffer.
                  Each train batch will be ``(1-r)`` from online + ``r`` from offline,
                  where ``r = offline_data_ratio``.
            offline_data_ratio: Used only when mode is ``"mixed"``. Must be in [0, 1].
        """
        self._online_start_step = self._global_step
        if self.online_use_cql_loss is not None:
            self.use_cql_loss = self.online_use_cql_loss
        if self.online_cql_alpha is not None:
            self.cql_alpha = self.online_cql_alpha

        if not (0.0 <= offline_data_ratio <= 1.0):
            raise ValueError(f"offline_data_ratio must be in [0, 1]; got {offline_data_ratio}.")
        self.offline_replay_buffer = None
        self.offline_data_ratio = 0.0
        cleared_transitions = 0
        if online_replay_mode == "empty":
            cleared_transitions = self._clear_replay_buffer()
        elif online_replay_mode == "append":
            pass
        elif online_replay_mode == "mixed":
            self.offline_replay_buffer = self.replay_buffer
            self.replay_buffer = self._build_replay_buffer()
            self.offline_data_ratio = offline_data_ratio
        else:
            raise ValueError(f"Unknown online_replay_mode: {online_replay_mode!r}")

        if self.logger:
            self.logger.add_summary("wsrl/online_start_step", self._global_step)
            self.logger.add_summary("wsrl/online_use_cql_loss", self.use_cql_loss)
            self.logger.add_summary("wsrl/online_cql_alpha", self.cql_alpha)
            self.logger.add_summary("wsrl/online_replay_mode", online_replay_mode)
            self.logger.add_summary("wsrl/online_replay_cleared", online_replay_mode == "empty")
            if online_replay_mode == "empty":
                self.logger.add_summary("wsrl/online_replay_size_before_clear", cleared_transitions)
            if online_replay_mode == "mixed":
                self.logger.add_summary("wsrl/offline_data_ratio", offline_data_ratio)

        # Re-compile the loss methods if compile is enabled — Python-side flags
        # (use_cql_loss, cql_alpha) may have flipped, invalidating the old graph.
        if self.use_compile and self._eager_critic_loss is not None:
            self._apply_compile()

    # --- CQL Loss Computation ---

    def _sample_n_actions_with_log_probs(
        self, obs, n: int, features: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``n`` actions per state from the current policy in a single
        vectorized forward pass; returns the actions and their log-probabilities.

        Mirrors wsrl/agents/sac.py::forward_policy_and_sample(repeat=n).

        Args:
            obs: Observation (Tensor or TensorDict). Ignored if ``features`` is
                supplied.
            n: Number of action samples per state.
            features: Optional precomputed features (avoids re-encoding).

        Returns:
            actions: (batch, n, action_dim)
            log_probs: (batch, n)
        """
        if features is None:
            features = self.policy.extract_features(obs)
        actor = self.policy.actor
        mean, log_std = actor(features)                      # (batch, action_dim)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample(sample_shape=(n,))              # (n, batch, action_dim)
        y_t = torch.tanh(x_t)
        action = y_t * actor.action_scale + actor.action_bias  # (n, batch, action_dim)
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1)                          # (n, batch)
        # Transpose to (batch, n, ...) to match original wsrl convention.
        return action.permute(1, 0, 2), log_prob.permute(1, 0)

    def _sample_random_actions(self, batch_size: int, action_dim: int) -> torch.Tensor:
        """Sample random OOD actions per the configured method."""
        if self.cql_action_sample_method == "uniform":
            return (
                torch.rand(
                    batch_size, self.cql_n_actions, action_dim, device=self.device
                )
                * 2.0
                - 1.0
            )
        elif self.cql_action_sample_method == "normal":
            return torch.randn(
                batch_size, self.cql_n_actions, action_dim, device=self.device
            )
        else:
            raise NotImplementedError(
                f"Unknown cql_action_sample_method: {self.cql_action_sample_method}"
            )

    # ------------------------------------------------------------------
    # Critic-loss component hooks. Subclasses can override individual
    # pieces without rewriting the whole critic loss.
    # ------------------------------------------------------------------

    def _calql_lower_bound(
        self,
        q_ood: torch.Tensor,
        mc_returns: torch.Tensor,
        n_samples: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, float]:
        """Apply Cal-QL infinite-horizon lower bound to OOD Q-values.

        Args:
            q_ood: ``(critic_size, batch, n_samples)`` raw OOD Q-values.
            mc_returns: ``(batch,)`` Monte Carlo returns from the replay sample.
            n_samples: ``3 * cql_n_actions`` — total OOD samples per state.
            batch_size: Convenience scalar.

        Returns:
            (q_ood_bounded, calql_bound_rate) where ``bound_rate`` is the
            fraction of OOD values that were below their MC lower bound prior
            to clamping (for logging).
        """
        mc_returns_b1 = mc_returns.reshape(batch_size, 1)  # (batch, 1)

        if self.calql_bound_random_actions:
            # Bound all 3*N action positions.
            mc_lower_bound = mc_returns_b1.expand(batch_size, n_samples)
        else:
            # Random actions: -inf (no bound). Current+next: real MC returns.
            fake = torch.full(
                (batch_size, self.cql_n_actions),
                float("-inf"),
                device=self.device,
                dtype=mc_returns_b1.dtype,
            )
            real = mc_returns_b1.expand(batch_size, 2 * self.cql_n_actions)
            mc_lower_bound = torch.cat([fake, real], dim=1)
        # Broadcast over the critic dim: (1, batch, n_samples)
        mc_lower_bound = mc_lower_bound.unsqueeze(0)

        # Track bound violation rate before clamping (for logging).
        num_vals = q_ood.numel()
        bound_rate = (q_ood < mc_lower_bound).sum().item() / max(num_vals, 1)
        return torch.maximum(q_ood, mc_lower_bound), bound_rate

    def _cql_regularizer(
        self,
        data: MCReplayBufferSample,
        q_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the CQL regularization term ``E[logsumexp_a Q] - E_data[Q]``.

        Faithful port of ``wsrl/agents/cql.py::CQLAgent._get_cql_q_diff``:
          1. Sample ``cql_n_actions`` random + current-policy + next-policy
             actions (with log-probs) → ``all_sampled_actions`` shape
             ``(batch, 3*N, action_dim)``.
          2. Evaluate Q over all sampled actions → ``(n_critics, batch, 3*N)``.
          3. Subsample critics with shared indices for both ``q_ood`` and
             ``q_pred`` (REDQ).
          4. Optionally apply Cal-QL lower bounds via ``_calql_lower_bound``.
          5. Apply importance sampling OR concat ``q_pred`` + ``-log(M)*temp``.
          6. ``logsumexp`` over actions; ``cql_q_diff = ood - q_pred``.

        Args:
            data: Replay buffer sample with ``mc_returns`` for Cal-QL.
            q_pred: Predicted Q-values for dataset actions, shape
                ``(n_critics, batch, 1)``.

        Returns:
            (cql_q_diff_mean, info). Caller multiplies by ``cql_alpha``.
        """
        batch_size = data.rewards.shape[0]
        action_dim = self.env.single_action_space.shape[0]
        info: dict[str, float] = {}

        # ---- 1. Sample OOD actions (random / current-policy / next-policy) ----
        cql_random_actions = self._sample_random_actions(batch_size, action_dim)
        cql_current_actions, cql_current_log_pis = self._sample_n_actions_with_log_probs(
            data.obs, self.cql_n_actions
        )
        cql_next_actions, cql_next_log_pis = self._sample_n_actions_with_log_probs(
            data.next_obs, self.cql_n_actions
        )
        # Concatenate along the action-sample axis: (batch, 3*N, action_dim)
        all_sampled_actions = torch.cat(
            [cql_random_actions, cql_current_actions, cql_next_actions], dim=1
        )

        # ---- 2. Evaluate Q over all sampled actions ----
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

        # ---- 3. Critic subsampling (shared indices for q_ood AND q_pred) ----
        if (
            self.critic_subsample_size is not None
            and self.critic_subsample_size < self.n_critics
        ):
            idx = torch.randint(
                0, self.n_critics, (self.critic_subsample_size,), device=self.device
            )
            q_ood = q_ood[idx]                              # (S, batch, n_samples)
            q_pred_for_diff = q_pred[idx].squeeze(-1)       # (S, batch)
        else:
            q_pred_for_diff = q_pred.squeeze(-1)            # (n_critics, batch)

        # ---- 4. Cal-QL lower bounds ----
        if self.use_calql and getattr(data, "mc_returns", None) is not None:
            q_ood, bound_rate = self._calql_lower_bound(
                q_ood, data.mc_returns, n_samples, batch_size
            )
            info["calql_bound_rate"] = bound_rate

        # ---- 5. Importance sampling OR concat-q_pred branch ----
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
            q_ood = q_ood - importance_log.unsqueeze(0)     # (S, batch, n_samples)
        else:
            # Original CQL: concatenate q_pred and apply -log(M)*temp.
            q_ood = torch.cat([q_ood, q_pred_for_diff.unsqueeze(-1)], dim=-1)
            q_ood = q_ood - float(np.log(q_ood.shape[-1])) * self.cql_temp

        # ---- 6. logsumexp over actions and CQL diff ----
        cql_ood_values = (
            torch.logsumexp(q_ood / self.cql_temp, dim=-1) * self.cql_temp
        )                                                    # (S, batch)
        cql_q_diff = cql_ood_values - q_pred_for_diff        # (S, batch)

        # Per-element clipping is applied ONLY when alpha is fixed (not autotune),
        # matching cql.py L262-268.
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

    def _cql_loss(
        self,
        data: MCReplayBufferSample,
        q_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Backward-compatible entry point for the CQL regularizer.

        Keep this as a wrapper rather than a class-level alias so subclass
        overrides of ``_cql_regularizer`` are dynamically dispatched.
        """
        return self._cql_regularizer(data, q_pred)

    # --- Training Methods ---

    def _critic_forward(self, obs, actions, target: bool = False):
        """Forward pass through critic network."""
        features = self.policy.extract_features(obs, stop_gradient=False)
        return self.policy.q_values_all(features, actions, target=target)

    def _target_q(self, data: MCReplayBufferSample) -> torch.Tensor:
        """Compute target Q-values for TD loss.

        Mirrors wsrl/agents/sac.py + cql.py target computation:
          - When ``cql_max_target_backup`` is on (CQL), sample
            ``cql_n_actions`` from the policy at ``next_obs`` and pick the
            argmax over actions of min-over-(subsampled)-critics-Q. Use the
            corresponding log-prob for the entropy bonus.
          - Otherwise, standard SAC target: one action sample, min over
            subsampled critics, subtract ``alpha * log_prob``.
        """
        alpha = self._current_alpha().detach()

        with torch.no_grad():
            if self.cql_max_target_backup and self.use_cql_loss:
                # Vectorized n-action sampling at next_obs.
                next_actions, next_log_probs = self._sample_n_actions_with_log_probs(
                    data.next_obs, self.cql_n_actions
                )
                # next_actions: (batch, N, action_dim); next_log_probs: (batch, N)
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
                # (S, batch * N, 1) → (S, batch, N)
                critic_size = q_next_sub.shape[0]
                q_next_sub = q_next_sub.reshape(
                    critic_size, batch_size, self.cql_n_actions
                )
                # min over subsampled critics: (batch, N)
                q_next_min = q_next_sub.min(dim=0).values

                # Take argmax over actions; gather corresponding log_probs.
                max_idx = q_next_min.argmax(dim=1, keepdim=True)         # (batch, 1)
                min_q_next = q_next_min.gather(1, max_idx)               # (batch, 1)
                next_log_prob = next_log_probs.gather(1, max_idx)        # (batch, 1)
            else:
                # Standard SAC target: single action sample.
                next_action, next_log_prob, _ = self.policy.actor_action_log_prob(
                    data.next_obs, stop_gradient=False
                )
                min_q_next = self.policy.min_q_value(
                    self.policy.extract_features(data.next_obs),
                    next_action,
                    subsample_size=self.critic_subsample_size,
                    target=True,
                )

            if self.backup_entropy:
                min_q_next = min_q_next - alpha * next_log_prob

            # Compute target
            target = data.rewards.reshape(-1, 1) + (
                1 - data.dones.reshape(-1, 1)
            ) * self.gamma * min_q_next

        return target

    def _td_loss(
        self,
        data: MCReplayBufferSample,
        q_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the TD loss term and ``target_q`` info.

        Returns ``(td_loss, info)``. ``q_pred`` is forwarded so callers don't
        recompute the Q forward.
        """
        target_q = self._target_q(data)
        target_q_expanded = target_q.unsqueeze(0).repeat(self.n_critics, 1, 1)
        td_loss = F.mse_loss(q_pred, target_q_expanded)
        info = {
            "td_loss": td_loss.item(),
            "target_q": target_q.mean().item(),
        }
        return td_loss, info

    def _critic_loss(self, data: MCReplayBufferSample) -> tuple[torch.Tensor, dict[str, float]]:
        """Combine ``_td_loss`` and ``_cql_regularizer`` into the critic loss."""
        info: dict[str, float] = {}
        q_pred = self._critic_forward(data.obs, data.actions, target=False)
        # q_pred: (n_critics, batch, 1)

        td_loss = torch.tensor(0.0, device=self.device)
        if self.use_td_loss:
            td_loss, td_info = self._td_loss(data, q_pred)
            info.update(td_info)

        cql_loss = torch.tensor(0.0, device=self.device)
        if self.use_cql_loss:
            cql_loss_raw, cql_info = self._cql_regularizer(data, q_pred)
            info.update(cql_info)

            cql_alpha = self._current_cql_alpha()
            if self.cql_autotune_alpha:
                # Lagrange penalty: alpha * (cql_loss - target_gap)
                cql_loss = cql_loss_raw - self.cql_target_action_gap
            else:
                cql_loss = cql_loss_raw
            cql_loss = cql_alpha * cql_loss

            info["cql_loss"] = cql_loss_raw.item()
            info["cql_alpha"] = cql_alpha.item()

        critic_loss = td_loss + cql_loss
        info["critic_loss"] = critic_loss.item()
        info["predicted_q"] = q_pred.mean().item()
        return critic_loss, info

    def _actor_loss(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute actor loss."""
        alpha = self._current_alpha().detach()
        action, log_prob, features = self.policy.actor_action_log_prob(
            obs, stop_gradient=False
        )
        min_q = self.policy.min_q_value(
            features, action, subsample_size=None, target=False
        )
        actor_loss = (alpha * log_prob - min_q).mean()
        return actor_loss, log_prob.detach()

    def _cql_alpha_loss(self, data: MCReplayBufferSample) -> torch.Tensor:
        """Compute CQL alpha Lagrange multiplier loss."""
        # Recompute CQL loss without gradients
        with torch.no_grad():
            q_pred = self._critic_forward(data.obs, data.actions, target=False)
            cql_loss_raw, _ = self._cql_loss(data, q_pred)

        # Lagrange penalty
        cql_alpha = self.policy.get_cql_alpha()
        cql_alpha_loss = -cql_alpha * (cql_loss_raw - self.cql_target_action_gap)
        return cql_alpha_loss

    @staticmethod
    def _mean_infos(infos: list[dict[str, float]]) -> dict[str, float]:
        if not infos:
            return {}
        keys = set().union(*(info.keys() for info in infos))
        return {
            key: float(np.mean([info[key] for info in infos if key in info]))
            for key in keys
        }

    def train(self, gradient_steps: int) -> dict[str, float]:
        """Training step with CQL/Cal-QL support."""
        high_utd_ratio = int(self.utd) if float(self.utd).is_integer() else 1
        if high_utd_ratio > 1:
            groups = gradient_steps // high_utd_ratio
            remainder = gradient_steps % high_utd_ratio
            infos: list[dict[str, float]] = []
            for _ in range(groups):
                infos.append(self.train_high_utd(utd_ratio=high_utd_ratio))
            if remainder:
                old_utd = self.utd
                self.utd = 1.0
                try:
                    infos.append(self.train(remainder))
                finally:
                    self.utd = old_utd
            return self._mean_infos(infos)

        critic_losses: list[float] = []
        actor_losses: list[float] = []
        alpha_losses: list[float] = []
        cql_alpha_losses: list[float] = []
        alphas: list[float] = []
        cql_alphas: list[float] = []

        # Aggregate info dicts
        info_keys = set()
        info_accum = {}

        for step in range(gradient_steps):
            self._global_update += 1
            data = self._sample_batch(self.batch_size)

            # --- Critic update ---
            critic_loss, critic_info = self._critic_loss(data)
            self.q_optimizer.zero_grad()
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
            self.q_optimizer.step()
            self._step_critic_scheduler()
            critic_losses.append(critic_loss.item())

            # Accumulate info
            for k, v in critic_info.items():
                if k not in info_accum:
                    info_accum[k] = []
                    info_keys.add(k)
                info_accum[k].append(v)

            # --- Actor + alpha updates ---
            if self._global_update % self.policy_frequency == 0:
                actor_loss, log_prob_detached = self._actor_loss(data.obs)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self._clip_grad_norm(self.policy.actor_parameters())
                self.actor_optimizer.step()
                self._step_actor_scheduler()
                actor_losses.append(actor_loss.item())

                # Entropy coefficient update
                if self.autotune:
                    alpha_loss = -(
                        self.temperature_lagrange() * (log_prob_detached + self.target_entropy)
                    ).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self._clip_grad_norm(self.temperature_lagrange.parameters())
                    self.alpha_optimizer.step()
                    alpha_losses.append(alpha_loss.item())

                # CQL alpha Lagrange multiplier update
                if self.cql_autotune_alpha:
                    cql_alpha_loss = self._cql_alpha_loss(data)
                    self.cql_alpha_optimizer.zero_grad()
                    cql_alpha_loss.backward()
                    self._clip_grad_norm(self.policy.cql_alpha_lagrange_parameters())
                    self.cql_alpha_optimizer.step()
                    cql_alpha_losses.append(cql_alpha_loss.item())

            alphas.append(self._current_alpha().item())
            if self.cql_autotune_alpha or self.use_cql_loss:
                cql_alphas.append(self._current_cql_alpha().item())

            # --- Target critic update ---
            if self._global_update % self.target_network_frequency == 0:
                polyak_update(
                    self.policy.critic.parameters(),
                    self.policy.critic_target.parameters(),
                    self.tau,
                )

        # Aggregate results
        out = {
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "alpha": float(np.mean(alphas)) if alphas else 0.0,
        }

        if alpha_losses:
            out["alpha_loss"] = float(np.mean(alpha_losses))
        if cql_alpha_losses:
            out["cql_alpha_loss"] = float(np.mean(cql_alpha_losses))
        if cql_alphas:
            out["cql_alpha"] = float(np.mean(cql_alphas))

        # Add accumulated info
        for k in info_keys:
            out[k] = float(np.mean(info_accum[k]))

        return out

    # ------------------------------------------------------------------
    # High-UTD training (Fix 13)
    # ------------------------------------------------------------------

    def train_high_utd(self, utd_ratio: int) -> dict[str, float]:
        """High-UTD training: ``utd_ratio`` critic-only updates per actor update.

        Mirrors ``wsrl/agents/sac.py::SACAgent.update_high_utd`` (L624-L678):
          - Sample one full batch.
          - Split it into ``utd_ratio`` minibatches.
          - For each minibatch: critic update + (optional) cql_alpha update +
            target polyak update.
          - Then ONE actor + temperature update on the full batch.

        Useful when ``utd_ratio >> 1`` (e.g., 20 for RLPD-style training)
        because the actor / alpha update is the cheap part of the loop and
        running it every step is wasteful.

        Args:
            utd_ratio: Number of critic updates per actor update. Must divide
                ``self.batch_size``.

        Returns:
            Aggregated info dict (means over minibatches for critic-side
            metrics, single value for actor/alpha metrics).
        """
        assert utd_ratio >= 1, f"utd_ratio must be >= 1, got {utd_ratio}"
        assert (
            self.batch_size % utd_ratio == 0
        ), f"batch_size ({self.batch_size}) must be divisible by utd_ratio ({utd_ratio})"

        full_batch = self._sample_batch(self.batch_size)
        minibatch_size = self.batch_size // utd_ratio

        critic_losses: list[float] = []
        info_accum: dict[str, list[float]] = {}

        # ----- Critic-only inner loop -----
        for j in range(utd_ratio):
            self._global_update += 1
            mb = self._slice_batch(full_batch, j * minibatch_size, minibatch_size)

            critic_loss, critic_info = self._critic_loss(mb)
            self.q_optimizer.zero_grad()
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
            self.q_optimizer.step()
            self._step_critic_scheduler()
            critic_losses.append(critic_loss.item())

            for k, v in critic_info.items():
                info_accum.setdefault(k, []).append(v)

            # Target critic polyak update.
            if self._global_update % self.target_network_frequency == 0:
                polyak_update(
                    self.policy.critic.parameters(),
                    self.policy.critic_target.parameters(),
                    self.tau,
                )

        # ----- Single actor + temperature update on the full batch -----
        actor_loss, log_prob_detached = self._actor_loss(full_batch.obs)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self._clip_grad_norm(self.policy.actor_parameters())
        self.actor_optimizer.step()
        self._step_actor_scheduler()

        alpha_loss_val = None
        if self.autotune:
            alpha_loss = -(
                self.temperature_lagrange() * (log_prob_detached + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._clip_grad_norm(self.temperature_lagrange.parameters())
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

        cql_alpha_loss_val = None
        if self.cql_autotune_alpha:
            cql_alpha_loss = self._cql_alpha_loss(full_batch)
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward()
            self._clip_grad_norm(self.policy.cql_alpha_lagrange_parameters())
            self.cql_alpha_optimizer.step()
            cql_alpha_loss_val = cql_alpha_loss.item()

        out: dict[str, float] = {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self._current_alpha().item()),
            "utd_ratio": float(utd_ratio),
        }
        if alpha_loss_val is not None:
            out["alpha_loss"] = alpha_loss_val
        if cql_alpha_loss_val is not None:
            out["cql_alpha_loss"] = cql_alpha_loss_val
        if self.cql_autotune_alpha or self.use_cql_loss:
            out["cql_alpha"] = float(self._current_cql_alpha().item())
        for k, vals in info_accum.items():
            out[k] = float(np.mean(vals))
        return out

    # ------------------------------------------------------------------
    # Helper: slice a replay sample (TensorDict-aware) for high-UTD splits.
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_batch(
        batch: MCReplayBufferSample, start: int, size: int
    ) -> MCReplayBufferSample:
        """Slice a replay sample along the batch dim. Handles dict obs."""
        end = start + size

        def _slice(x):
            if isinstance(x, dict):
                return {k: v[start:end] for k, v in x.items()}
            if x is None:
                return None
            return x[start:end]

        return MCReplayBufferSample(
            obs=_slice(batch.obs),
            next_obs=_slice(batch.next_obs),
            actions=_slice(batch.actions),
            rewards=_slice(batch.rewards),
            dones=_slice(batch.dones),
            mc_returns=_slice(batch.mc_returns) if batch.mc_returns is not None else None,
        )
