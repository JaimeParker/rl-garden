"""Behavioral Cloning (BC) for offline datasets.

Pure offline algorithm: trains an actor to imitate expert actions via
maximum log-likelihood. Inherits ``OfflineRLAlgorithm`` and is wired into
the same ``run_offline_pretraining`` loop as IQL/CQL/CalQL.

Observation encoding (Box state-only vs. Dict/RGBD) is resolved by the
shared ``ObservationEncoderMixin`` (see ``rl_garden/algorithms/_observation.py``)
from ``encoder_config``/``obs_groups``, matching the IQL convention so BC can
be used as a drop-in baseline on the same datasets. BC has no critic, so only
the actor-role encoder (``self.observation_encoders.actor``) is ever built.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObsGroups
from rl_garden.policies.bc_policy import BCPolicy


class BC(OfflineRLAlgorithm):
    """Behavioral Cloning: maximize log-likelihood of expert actions."""

    _compatible_checkpoint_algorithms = ("BC",)
    _SUPPORTED_POLICY_KWARGS = frozenset(
        {"actor_extractor_class", "actor_extractor_kwargs"}
    )
    # BC has no critic, so extract_actor_features must never stop-gradient
    # (the encoder is trained end-to-end by the actor loss, the only loss
    # that ever touches it) -- see BasePolicy.extract_actor_features's
    # formula, which would otherwise detach under the mixin's
    # "shared_critic_grad" default whenever critic_extractor is None.
    encoder_sharing = "shared"

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        offline_sampling: str = "with_replace",
        actor_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        net_arch: Optional[Sequence[int]] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        image_augmentation_seed: Optional[int] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        actor_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        kernel_init: Optional[
            Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
        ] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        tanh_squash: bool = True,
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
            gamma=0.99,  # not used by BC; inherited field
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
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )

        self.actor_lr = actor_lr
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip_norm = grad_clip_norm
        self.net_arch: list[int] = list(net_arch) if net_arch is not None else [256, 256]
        self.actor_use_layer_norm = actor_use_layer_norm
        self.actor_use_group_norm = actor_use_group_norm
        self.num_groups = num_groups
        self.actor_dropout_rate = actor_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.std_parameterization = std_parameterization
        self.tanh_squash = tanh_squash

        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self._image_augmentation_seed = image_augmentation_seed

        self.policy_kwargs = self._normalize_policy_kwargs(policy_kwargs)
        self._setup_model()

    # --- checkpoint ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("actor_optimizer",)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = {
            **super()._checkpoint_metadata(),
            "actor_lr": self.actor_lr,
            "weight_decay": self.weight_decay,
            "use_adamw": self.use_adamw,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "grad_clip_norm": self.grad_clip_norm,
            "net_arch": self.net_arch,
            "encoder_config": (
                dataclasses.asdict(self.encoder_config) if self.encoder_config is not None else None
            ),
            "obs_groups": (
                dataclasses.asdict(self.obs_groups) if self.obs_groups is not None else None
            ),
            "image_augmentation_seed": self._image_augmentation_seed,
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

    # --- model setup ---

    def _setup_model(self) -> None:
        extractor_kwargs = self._policy_extractor_kwargs(
            self.env.single_observation_space,
            augmentation_seed=self._image_augmentation_seed,
        )
        self.policy = BCPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            actor_extractor=extractor_kwargs["actor_extractor"],
            net_arch=self.net_arch,
            use_layer_norm=self.actor_use_layer_norm,
            use_group_norm=self.actor_use_group_norm,
            num_groups=self.num_groups,
            dropout_rate=self.actor_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            std_parameterization=self.std_parameterization,
            tanh_squash=self.tanh_squash,
            encoder_sharing=extractor_kwargs["encoder_sharing"],
        ).to(self.device)

        self.actor_optimizer = make_optimizer(
            list(self.policy.actor_parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.replay_buffer = self._build_replay_buffer()
        self._lr_schedulers = [
            make_lr_scheduler(
                self.actor_optimizer,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
        ]

    # --- training ---

    def _compute_losses(self, data) -> tuple[torch.Tensor, dict[str, float]]:
        log_prob, det_action = self.policy.behavior_log_prob(
            data.obs, data.actions, stop_gradient=False
        )
        actor_loss = -log_prob.mean()
        with torch.no_grad():
            behavior_mse = F.mse_loss(det_action, data.actions)

        metrics = {
            "loss": float(actor_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "behavior_log_prob": float(log_prob.detach().mean().item()),
            "behavior_mse": float(behavior_mse.item()),
        }
        return actor_loss, metrics

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch()

            self.actor_optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._compute_losses(data)
            loss.backward()
            self._clip_grad_norm()
            self.actor_optimizer.step()
            self._step_schedulers()

            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        return {key: value / gradient_steps for key, value in metrics_sum.items()}

    # --- helpers (mirrored from IQL) ---

    def _sample_train_batch(self):
        if self.offline_sampling == "with_replace":
            return self.replay_buffer.sample(self.batch_size)
        if self.offline_sampling == "without_replace":
            sample = getattr(self.replay_buffer, "sample_without_repeat", None)
            if sample is None:
                raise ValueError(
                    "offline_sampling='without_replace' requires a replay buffer "
                    "with sample_without_repeat()."
                )
            return sample(self.batch_size)
        raise ValueError(f"Unknown offline_sampling: {self.offline_sampling!r}")

    def _step_schedulers(self) -> None:
        for sched in self._lr_schedulers:
            if sched is not None:
                sched.step()

    def _clip_grad_norm(self) -> None:
        if self.grad_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.actor_parameters()), self.grad_clip_norm
        )

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
