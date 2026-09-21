"""Implicit Q-Learning (IQL): expectile value regression + AWR actor.

``IQLCore`` holds the loss/network/optimizer logic shared by the pure offline
``IQL`` (built on ``OfflineRLAlgorithm``) and the rollout-capable
``_IQLRolloutTrainingShell`` (built on ``OffPolicyAlgorithm``, backing
``Off2OnIQL``). Box observations use ``FlattenExtractor`` and Dict
observations use ``CombinedExtractor``, so state, image, and image+state
inputs share the same policy path.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F

from rl_garden.algorithms._observation import EncoderSharing
from rl_garden.algorithms.off2on import Off2OnReplayMixin
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObsGroups
from rl_garden.policies.iql_policy import IQLPolicy


class IQLCore:
    """Shared IQL loss/network logic: expectile V-regression + AWR actor."""

    _SUPPORTED_POLICY_KWARGS = frozenset(
        {
            "actor_extractor_class",
            "actor_extractor_kwargs",
            "critic_extractor_class",
            "critic_extractor_kwargs",
        }
    )

    def _init_iql_params(
        self,
        *,
        tau: float = 0.005,
        utd: float = 1.0,
        actor_lr: float = 3e-4,
        critic_value_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        actor_lr_schedule: Optional[ScheduleType] = None,
        actor_lr_warmup_steps: Optional[int] = None,
        actor_lr_decay_steps: Optional[int] = None,
        actor_lr_min_ratio: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        expectile: float = 0.7,
        temperature: float = 3.0,
        adv_clip_max: float = 100.0,
        actor_distribution: Literal["squashed", "unsquashed"] = "squashed",
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        value_hidden_dims: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        value_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        kernel_init: Optional[
            Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
        ] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
    ) -> None:
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {tau}.")
        if not (0.0 < expectile < 1.0):
            raise ValueError(f"expectile must be in (0, 1), got {expectile}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if adv_clip_max <= 0:
            raise ValueError(f"adv_clip_max must be positive, got {adv_clip_max}.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )
        effective_actor_schedule = (
            actor_lr_schedule if actor_lr_schedule is not None else lr_schedule
        )
        effective_actor_decay_steps = (
            actor_lr_decay_steps if actor_lr_decay_steps is not None else lr_decay_steps
        )
        if effective_actor_schedule == "warmup_cosine" and effective_actor_decay_steps <= 0:
            raise ValueError(
                "warmup_cosine requires a positive decay step count for the actor "
                "schedule; set actor_lr_decay_steps (or lr_decay_steps, which the "
                f"actor schedule falls back to), got {effective_actor_decay_steps}."
            )

        self.tau = tau
        self.utd = utd
        self.actor_lr = actor_lr
        self.critic_value_lr = critic_value_lr
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.actor_lr_schedule = actor_lr_schedule
        self.actor_lr_warmup_steps = actor_lr_warmup_steps
        self.actor_lr_decay_steps = actor_lr_decay_steps
        self.actor_lr_min_ratio = actor_lr_min_ratio
        self.grad_clip_norm = grad_clip_norm
        self.expectile = expectile
        self.temperature = temperature
        self.adv_clip_max = adv_clip_max
        self.actor_distribution = actor_distribution
        self.net_arch = self._resolve_net_arch(
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            value_hidden_dims=value_hidden_dims,
        )
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size
        self.actor_use_layer_norm = actor_use_layer_norm
        self.critic_use_layer_norm = critic_use_layer_norm
        self.value_use_layer_norm = value_use_layer_norm
        self.actor_use_group_norm = actor_use_group_norm
        self.critic_use_group_norm = critic_use_group_norm
        self.value_use_group_norm = value_use_group_norm
        self.num_groups = num_groups
        self.actor_dropout_rate = actor_dropout_rate
        self.critic_dropout_rate = critic_dropout_rate
        self.value_dropout_rate = value_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.std_parameterization = std_parameterization

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("critic_value_optimizer", "actor_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = {
            **super()._checkpoint_metadata(),
            "tau": self.tau,
            "utd": self.utd,
            "actor_lr": self.actor_lr,
            "critic_value_lr": self.critic_value_lr,
            "weight_decay": self.weight_decay,
            "use_adamw": self.use_adamw,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "actor_lr_schedule": self.actor_lr_schedule,
            "actor_lr_warmup_steps": self.actor_lr_warmup_steps,
            "actor_lr_decay_steps": self.actor_lr_decay_steps,
            "actor_lr_min_ratio": self.actor_lr_min_ratio,
            "grad_clip_norm": self.grad_clip_norm,
            "expectile": self.expectile,
            "temperature": self.temperature,
            "adv_clip_max": self.adv_clip_max,
            "actor_distribution": self.actor_distribution,
            "net_arch": self.net_arch,
            "n_critics": self.n_critics,
            "critic_subsample_size": self.critic_subsample_size,
            "encoder_sharing": self.encoder_sharing,
            "encoder_sharing_origin": self.encoder_sharing_origin,
            "encoder_config": (
                dataclasses.asdict(self.encoder_config) if self.encoder_config is not None else None
            ),
            "obs_groups": (
                dataclasses.asdict(self.obs_groups) if self.obs_groups is not None else None
            ),
            "critic_encoder_config": (
                dataclasses.asdict(self.critic_encoder_config)
                if self.critic_encoder_config is not None
                else None
            ),
        }
        return meta

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "lr_scheduler_states": [
                sched.state_dict() if sched is not None else None
                for sched in self._lr_schedulers
            ]
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        for sched, sched_state in zip(
            self._lr_schedulers, state.get("lr_scheduler_states", [])
        ):
            if sched is not None and sched_state is not None:
                sched.load_state_dict(sched_state)

    @staticmethod
    def _resolve_net_arch(
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]],
        actor_hidden_dims: Optional[Sequence[int]],
        critic_hidden_dims: Optional[Sequence[int]],
        value_hidden_dims: Optional[Sequence[int]],
    ) -> Sequence[int] | dict[str, list[int]]:
        if net_arch is not None:
            if (
                actor_hidden_dims is not None
                or critic_hidden_dims is not None
                or value_hidden_dims is not None
            ):
                warnings.warn(
                    "actor_hidden_dims/critic_hidden_dims/value_hidden_dims are "
                    "ignored when net_arch is provided. Use net_arch only.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if isinstance(net_arch, dict):
                if "pi" not in net_arch or "qf" not in net_arch:
                    raise ValueError(
                        "net_arch dict must contain both 'pi' and 'qf' keys."
                    )
                return {
                    "pi": list(net_arch["pi"]),
                    "qf": list(net_arch["qf"]),
                    "vf": list(net_arch.get("vf", net_arch["qf"])),
                }
            return list(net_arch)

        if (
            actor_hidden_dims is not None
            or critic_hidden_dims is not None
            or value_hidden_dims is not None
        ):
            warnings.warn(
                "actor_hidden_dims/critic_hidden_dims/value_hidden_dims are "
                "deprecated. Use net_arch=list[...] or net_arch={'pi': [...], "
                "'qf': [...], 'vf': [...]} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            pi_arch = (
                list(actor_hidden_dims) if actor_hidden_dims is not None else [256, 256]
            )
            qf_arch = (
                list(critic_hidden_dims)
                if critic_hidden_dims is not None
                else list(pi_arch)
            )
            vf_arch = (
                list(value_hidden_dims)
                if value_hidden_dims is not None
                else list(qf_arch)
            )
            return {"pi": pi_arch, "qf": qf_arch, "vf": vf_arch}

        return [256, 256]

    def _normalize_policy_kwargs(
        self, policy_kwargs: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        normalized = dict(policy_kwargs or {})
        unsupported = sorted(set(normalized) - self._SUPPORTED_POLICY_KWARGS)
        if unsupported:
            raise ValueError(
                "Unsupported policy_kwargs keys: "
                + ", ".join(unsupported)
                + ". Supported keys are: "
                + ", ".join(sorted(self._SUPPORTED_POLICY_KWARGS))
                + "."
            )
        return normalized

    def _build_replay_buffer(self):
        # obs_space is always Dict (boundary normalization is unconditional).
        return ReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _setup_model(self) -> None:
        self.policy = IQLPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            value_use_layer_norm=self.value_use_layer_norm,
            actor_use_group_norm=self.actor_use_group_norm,
            critic_use_group_norm=self.critic_use_group_norm,
            value_use_group_norm=self.value_use_group_norm,
            num_groups=self.num_groups,
            actor_dropout_rate=self.actor_dropout_rate,
            critic_dropout_rate=self.critic_dropout_rate,
            value_dropout_rate=self.value_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            std_parameterization=self.std_parameterization,
            actor_distribution=self.actor_distribution,
            **self._policy_extractor_kwargs(self.env.single_observation_space),
        ).to(self.device)

        self.critic_value_optimizer = make_optimizer(
            list(self.policy.critic_value_and_encoder_parameters()),
            lr=self.critic_value_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.actor_optimizer = make_optimizer(
            list(self.policy.actor_parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.replay_buffer = self._build_replay_buffer()
        self._lr_schedulers = [
            make_lr_scheduler(
                self.critic_value_optimizer,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            ),
            make_lr_scheduler(
                self.actor_optimizer,
                schedule_type=(
                    self.actor_lr_schedule
                    if self.actor_lr_schedule is not None
                    else self.lr_schedule
                ),
                warmup_steps=(
                    self.actor_lr_warmup_steps
                    if self.actor_lr_warmup_steps is not None
                    else self.lr_warmup_steps
                ),
                decay_steps=(
                    self.actor_lr_decay_steps
                    if self.actor_lr_decay_steps is not None
                    else self.lr_decay_steps
                ),
                min_lr_ratio=(
                    self.actor_lr_min_ratio
                    if self.actor_lr_min_ratio is not None
                    else self.lr_min_ratio
                ),
            ),
        ]

    def _sample_train_batch(self, batch_size: int):
        if self.offline_sampling == "with_replace":
            return self.replay_buffer.sample(batch_size)
        if self.offline_sampling == "without_replace":
            sample = getattr(self.replay_buffer, "sample_without_repeat", None)
            if sample is None:
                raise ValueError(
                    "offline_sampling='without_replace' requires a replay buffer "
                    "with sample_without_repeat()."
                )
            return sample(batch_size)
        raise ValueError(f"Unknown offline_sampling: {self.offline_sampling!r}")

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        return weight * diff.pow(2)

    def _target_min_q(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.policy.min_q_value(
            features,
            actions,
            subsample_size=self.critic_subsample_size,
            target=True,
        )

    def _compute_losses(self, data) -> tuple[torch.Tensor, dict[str, float]]:
        # V and Q are critic-role heads (both trained here): critic-role
        # extraction, never detached.
        features = self.policy.extract_critic_features(data.obs)

        with torch.no_grad():
            target_q_for_value = self._target_min_q(features.detach(), data.actions)
        values = self.policy.value(features)
        value_loss = self._expectile_loss(target_q_for_value - values).mean()

        q_pred = self.policy.q_values_all(features, data.actions, target=False)
        with torch.no_grad():
            next_features = self.policy.extract_critic_features(data.next_obs)
            next_v = self.policy.value(next_features)
            target_q = (
                data.rewards.unsqueeze(-1)
                + self.gamma * (1.0 - data.dones.unsqueeze(-1)) * next_v
            )
        critic_loss = F.mse_loss(q_pred, target_q.unsqueeze(0).expand_as(q_pred))

        with torch.no_grad():
            adv = target_q_for_value - values
            exp_adv = torch.exp(adv * self.temperature).clamp(max=self.adv_clip_max)
        # AWR actor loss: actor-role read, no explicit stop_gradient -- applies
        # extract_actor_features's encoder_sharing rule (identical to the old
        # hardcoded stop_gradient=True under the only previously-reachable
        # "shared_critic_grad" sharing).
        log_prob, deterministic_action = self.policy.behavior_log_prob(
            data.obs, data.actions
        )
        actor_loss = -(exp_adv * log_prob).mean()

        total_loss = value_loss + critic_loss + actor_loss
        metrics = {
            "loss": float(total_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "q": float(q_pred.detach().mean().item()),
            "target_q": float(target_q.detach().mean().item()),
            "v": float(values.detach().mean().item()),
            "adv": float(adv.detach().mean().item()),
            "adv_max": float(adv.detach().max().item()),
            "adv_min": float(adv.detach().min().item()),
            "exp_adv": float(exp_adv.detach().mean().item()),
            "behavior_log_prob": float(log_prob.detach().mean().item()),
            "behavior_mse": float(
                F.mse_loss(deterministic_action.detach(), data.actions).item()
            ),
        }
        return total_loss, metrics

    def _polyak_update(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(
                self.policy.critic.parameters(), self.policy.critic_target.parameters()
            ):
                p_targ.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

    def _step_schedulers(self) -> None:
        for sched in self._lr_schedulers:
            if sched is not None:
                sched.step()

    def _clip_grad_norm(self) -> None:
        if self.grad_clip_norm is None:
            return
        params = list(self.policy.critic_value_and_encoder_parameters()) + list(
            self.policy.actor_parameters()
        )
        torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        del compute_info
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch(self.batch_size)

            self.critic_value_optimizer.zero_grad(set_to_none=True)
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._compute_losses(data)
            loss.backward()
            self._clip_grad_norm()
            self.critic_value_optimizer.step()
            self.actor_optimizer.step()
            self._step_schedulers()
            self._polyak_update()

            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        return {key: value / gradient_steps for key, value in metrics_sum.items()}


class _IQLRolloutTrainingShell(Off2OnReplayMixin, IQLCore, OffPolicyAlgorithm):
    """Internal rollout/eval shell that wires ``IQLCore`` into ``OffPolicyAlgorithm``.

    Generic offline->online transition mechanics (replay-buffer switching,
    mixed-batch sampling, checkpoint/probe/logging plumbing) are inherited
    from ``Off2OnReplayMixin``. IQL needs no algorithm-specific override at
    the online switch (confirmed against the reference WSRL/IQL JAX
    implementation: IQL is treated identically to plain SAC at the
    offline->online switch), so neither ``_apply_online_regularizer_override``
    nor ``_offline_probe_metrics`` is overridden here.

    .. warning::
       **Do not instantiate this class directly.** It exists only to back
       :class:`~rl_garden.algorithms.Off2OnIQL`. For standalone offline IQL
       pretraining use :class:`IQL`. The shape and arguments of this shell
       may change without notice.
    """

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        training_freq: int = 64,
        utd: float = 1.0,
        bootstrap_at_done: str = "truncated",
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
        actor_lr: float = 3e-4,
        critic_value_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        actor_lr_schedule: Optional[ScheduleType] = None,
        actor_lr_warmup_steps: Optional[int] = None,
        actor_lr_decay_steps: Optional[int] = None,
        actor_lr_min_ratio: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        expectile: float = 0.7,
        temperature: float = 3.0,
        adv_clip_max: float = 100.0,
        actor_distribution: Literal["squashed", "unsquashed"] = "squashed",
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        value_hidden_dims: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        value_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        kernel_init: Optional[
            Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
        ] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
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
        initial_training_phase: Optional[InitialTrainingPhase] = None,
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
            initial_training_phase=initial_training_phase,
        )
        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self.critic_encoder_config = critic_encoder_config
        self.encoder_sharing = encoder_sharing
        self._init_iql_params(
            tau=tau,
            utd=utd,
            actor_lr=actor_lr,
            critic_value_lr=critic_value_lr,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            actor_lr_schedule=actor_lr_schedule,
            actor_lr_warmup_steps=actor_lr_warmup_steps,
            actor_lr_decay_steps=actor_lr_decay_steps,
            actor_lr_min_ratio=actor_lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            expectile=expectile,
            temperature=temperature,
            adv_clip_max=adv_clip_max,
            actor_distribution=actor_distribution,
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            value_hidden_dims=value_hidden_dims,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            value_use_layer_norm=value_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            value_use_group_norm=value_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            value_dropout_rate=value_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
        )
        self.policy_kwargs = self._normalize_policy_kwargs(policy_kwargs)
        self._setup_model()
        self._init_off2on_params(offline_sampling=offline_sampling)


class IQL(IQLCore, OfflineRLAlgorithm):
    """Offline IQL with AWR actor loss and expectile value regression."""

    _compatible_checkpoint_algorithms = ("IQL",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        offline_sampling: str = "with_replace",
        utd: float = 1.0,
        actor_lr: float = 3e-4,
        critic_value_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        actor_lr_schedule: Optional[ScheduleType] = None,
        actor_lr_warmup_steps: Optional[int] = None,
        actor_lr_decay_steps: Optional[int] = None,
        actor_lr_min_ratio: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        expectile: float = 0.7,
        temperature: float = 3.0,
        adv_clip_max: float = 100.0,
        actor_distribution: Literal["squashed", "unsquashed"] = "squashed",
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        value_hidden_dims: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        value_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        kernel_init: Optional[
            Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
        ] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 0,
        num_eval_steps: int = 50,
        eval_env: Optional[Any] = None,
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
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            eval_env=eval_env,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
        )
        self._init_iql_params(
            tau=tau,
            utd=utd,
            actor_lr=actor_lr,
            critic_value_lr=critic_value_lr,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            actor_lr_schedule=actor_lr_schedule,
            actor_lr_warmup_steps=actor_lr_warmup_steps,
            actor_lr_decay_steps=actor_lr_decay_steps,
            actor_lr_min_ratio=actor_lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            expectile=expectile,
            temperature=temperature,
            adv_clip_max=adv_clip_max,
            actor_distribution=actor_distribution,
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            value_hidden_dims=value_hidden_dims,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            value_use_layer_norm=value_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            value_use_group_norm=value_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            value_dropout_rate=value_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
        )
        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self.critic_encoder_config = critic_encoder_config
        self.encoder_sharing = encoder_sharing

        self.policy_kwargs = self._normalize_policy_kwargs(policy_kwargs)
        self._setup_model()
