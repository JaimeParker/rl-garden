"""Offline SAC entrypoint backed by the shared SACCore update path."""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.algorithms.sac import SAC
from rl_garden.algorithms.sac_core import SACCore
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.sac_policy import SACPolicy


class OfflineSAC(SACCore, OfflineRLAlgorithm):
    """Pure offline SAC over a static replay buffer.

    This class intentionally mirrors SAC's update defaults but inherits
    ``OfflineRLAlgorithm`` instead of the online rollout loop. Dataset loading is
    left to callers, which can fill ``replay_buffer`` before ``learn``.
    """

    _SUPPORTED_POLICY_KWARGS = SAC._SUPPORTED_POLICY_KWARGS
    _resolve_net_arch = staticmethod(SAC._resolve_net_arch)

    def __init__(
        self,
        env: OfflineEnvSpec,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        offline_sampling: str = "with_replace",
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        alpha_lr: Optional[float] = None,
        policy_frequency: int = 1,
        target_network_frequency: int = 1,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        ent_coef: float | str = "auto",
        target_entropy: float | str = "auto",
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
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
        self.tau = tau
        self.utd = 1.0
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.alpha_lr = alpha_lr if alpha_lr is not None else q_lr
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
        self.ent_coef_init = ent_coef
        self.target_entropy_arg = target_entropy
        self.net_arch = self._resolve_net_arch(
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
        )
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size
        self.policy_kwargs = SAC._normalize_policy_kwargs(self, policy_kwargs)
        self._setup_model()

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "tau": self.tau,
            "policy_lr": self.policy_lr,
            "q_lr": self.q_lr,
            "alpha_lr": self.alpha_lr,
            "policy_frequency": self.policy_frequency,
            "target_network_frequency": self.target_network_frequency,
            "weight_decay": self.weight_decay,
            "use_adamw": self.use_adamw,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "grad_clip_norm": self.grad_clip_norm,
            "ent_coef": self.ent_coef_init,
            "target_entropy": self.target_entropy_arg,
            "target_entropy_value": self.target_entropy,
            "net_arch": self.net_arch,
            "n_critics": self.n_critics,
            "critic_subsample_size": self.critic_subsample_size,
        }

    def _default_features_extractor_class(self) -> type[BaseFeaturesExtractor]:
        assert isinstance(self.env.single_observation_space, spaces.Box), (
            "OfflineSAC expects a flat Box observation space."
        )
        return FlattenExtractor

    def _default_features_extractor_kwargs(self) -> dict[str, Any]:
        return {}

    _normalize_policy_kwargs = SAC._normalize_policy_kwargs
    _resolve_policy_kwargs = SAC._resolve_policy_kwargs
    _build_features_extractor = SAC._build_features_extractor

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
        self.autotune = isinstance(self.ent_coef_init, str) and self.ent_coef_init.startswith(
            "auto"
        )
        if self.autotune:
            init = 1.0
            if isinstance(self.ent_coef_init, str) and "_" in self.ent_coef_init:
                init = float(self.ent_coef_init.split("_")[1])
            self.log_alpha = torch.log(
                torch.ones(1, device=self.device) * init
            ).requires_grad_(True)
            self.alpha_optimizer = make_optimizer(
                [self.log_alpha], lr=self.alpha_lr, weight_decay=0.0, use_adamw=self.use_adamw
            )
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self._fixed_alpha = torch.tensor(float(self.ent_coef_init), device=self.device)

        if self.target_entropy_arg == "auto":
            self.target_entropy = float(
                -np.prod(self.env.single_action_space.shape).astype(np.float32)
            )
        else:
            self.target_entropy = float(self.target_entropy_arg)

        self.replay_buffer = self._build_replay_buffer()
        self._lr_schedulers = [
            make_lr_scheduler(
                opt,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
            for opt in (self.q_optimizer, self.actor_optimizer)
        ]

    def _sample_train_batch(self, batch_size: int):
        if self.offline_sampling == "without_replace":
            return self.replay_buffer.sample_without_repeat(batch_size)
        return self.replay_buffer.sample(batch_size)

    def _current_alpha(self) -> torch.Tensor:
        if self.autotune:
            return self.log_alpha.exp()
        return self._fixed_alpha

    def _td_loss(self, data, q_pred: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        target_q = self._target_q(data)
        q_loss = sum(F.mse_loss(q, target_q) for q in q_pred)
        return q_loss, {"td_loss": q_loss.item(), "target_q": target_q.mean().item()}

    _step_critic_scheduler = SAC._step_critic_scheduler
    _step_actor_scheduler = SAC._step_actor_scheduler
    _clip_grad_norm = SAC._clip_grad_norm
