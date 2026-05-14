"""SAC algorithm for Box and Dict observations.

Template: ManiSkill's ``examples/baselines/sac/sac.py``, restructured as
an ``OffPolicyAlgorithm`` subclass with a ``SACPolicy`` that owns a
features extractor. Like Stable-Baselines3, input modality is handled by the
extractor: Box observations use ``FlattenExtractor`` and Dict observations use
``CombinedExtractor``.
"""
from __future__ import annotations

import warnings
from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.sac_core import SACCore
from rl_garden.buffers.dict_buffer import DictReplayBuffer
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.combined import (
    CombinedExtractor,
    ImageEncoderFactory,
    default_image_encoder_factory,
)
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.sac_policy import SACPolicy


class SAC(SACCore, OffPolicyAlgorithm):
    _SUPPORTED_POLICY_KWARGS = frozenset({"features_extractor_class", "features_extractor_kwargs"})

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 1024,
        gamma: float = 0.8,
        tau: float = 0.01,
        training_freq: int = 64,
        utd: float = 0.5,
        bootstrap_at_done: str = "always",
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
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: Optional[tuple[str, ...]] = None,
        state_key: Optional[str] = None,
        use_proprio: Optional[bool] = None,
        proprio_latent_dim: Optional[int] = None,
        image_fusion_mode: Optional[str] = None,
        enable_stacking: Optional[bool] = None,
        detach_encoder_on_actor: bool = True,
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

        obs_space = self.env.single_observation_space
        image_kwargs_explicit = {
            "image_encoder_factory": image_encoder_factory,
            "image_keys": image_keys,
            "state_key": state_key,
            "use_proprio": use_proprio,
            "proprio_latent_dim": proprio_latent_dim,
            "image_fusion_mode": image_fusion_mode,
            "enable_stacking": enable_stacking,
        }
        explicitly_set = [k for k, v in image_kwargs_explicit.items() if v is not None]

        if isinstance(obs_space, spaces.Box):
            if explicitly_set:
                raise ValueError(
                    f"SAC with Box observation space does not accept image-related "
                    f"kwargs (got {explicitly_set}). Use a Dict observation space, "
                    f"or remove these kwargs."
                )
            self._is_dict_obs = False
        elif isinstance(obs_space, spaces.Dict):
            if not detach_encoder_on_actor:
                raise ValueError(
                    "SAC always uses stop_gradient=True on the actor image path for "
                    "Dict observations so image encoders are trained only by critic loss."
                )
            self._is_dict_obs = True
            self._image_encoder_factory = (
                image_encoder_factory or default_image_encoder_factory()
            )
            self._image_keys = image_keys if image_keys is not None else ("rgb", "depth")
            self._state_key = state_key if state_key is not None else "state"
            self._use_proprio = use_proprio if use_proprio is not None else True
            self._proprio_latent_dim = (
                proprio_latent_dim if proprio_latent_dim is not None else 64
            )
            self._image_fusion_mode = (
                image_fusion_mode if image_fusion_mode is not None else "stack_channels"
            )
            self._enable_stacking = enable_stacking if enable_stacking is not None else False
        else:
            raise TypeError(
                f"SAC supports Box or Dict observation spaces, got {type(obs_space)}"
            )

        self.policy_kwargs = self._normalize_policy_kwargs(policy_kwargs)

        self._setup_model()

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = {
            **super()._checkpoint_metadata(),
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
        if self._is_dict_obs:
            meta.update(
                {
                    "image_keys": self._image_keys,
                    "state_key": self._state_key,
                    "use_proprio": self._use_proprio,
                    "proprio_latent_dim": self._proprio_latent_dim,
                    "image_fusion_mode": self._image_fusion_mode,
                    "enable_stacking": self._enable_stacking,
                }
            )
        return meta

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "autotune": self.autotune,
            "target_entropy": self.target_entropy,
        }
        if self.autotune:
            state["log_alpha"] = self.log_alpha.detach()
        else:
            state["fixed_alpha"] = self._fixed_alpha.detach()
        sched_states: list[Optional[dict]] = []
        for sched in getattr(self, "_lr_schedulers", []):
            sched_states.append(sched.state_dict() if sched is not None else None)
        state["lr_scheduler_states"] = sched_states
        return state

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        if "target_entropy" in state:
            self.target_entropy = float(state["target_entropy"])
        if self.autotune and "log_alpha" in state:
            self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        elif not self.autotune and "fixed_alpha" in state:
            self._fixed_alpha = state["fixed_alpha"].to(self.device)
        if "lr_scheduler_states" in state:
            for sched, sched_state in zip(self._lr_schedulers, state["lr_scheduler_states"]):
                if sched is not None and sched_state is not None:
                    sched.load_state_dict(sched_state)

    # --- construction hooks ---

    def _default_features_extractor_class(self) -> type[BaseFeaturesExtractor]:
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Box):
            return FlattenExtractor
        if isinstance(obs_space, spaces.Dict):
            return CombinedExtractor
        raise TypeError(
            "SAC supports Box or Dict observation spaces, got "
            + str(type(obs_space))
        )

    def _default_features_extractor_kwargs(self) -> dict[str, Any]:
        if self._is_dict_obs:
            return {
                "image_keys": self._image_keys,
                "state_key": self._state_key,
                "image_encoder_factory": self._image_encoder_factory,
                "proprio_latent_dim": self._proprio_latent_dim,
                "use_proprio": self._use_proprio,
                "fusion_mode": self._image_fusion_mode,
                "enable_stacking": self._enable_stacking,
            }
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

    def _build_replay_buffer(self):
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Dict):
            return DictReplayBuffer(
                observation_space=obs_space,
                action_space=self.env.single_action_space,
                num_envs=self.num_envs,
                buffer_size=self.buffer_size,
                storage_device=self.buffer_device,
                sample_device=self.device,
            )
        return TensorReplayBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _policy_action_space(self) -> spaces.Box:
        return self.env.single_action_space

    def _build_policy(self, features_extractor: BaseFeaturesExtractor) -> SACPolicy:
        return SACPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self._policy_action_space(),
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            critic_subsample_size=self.critic_subsample_size,
        )

    def _actor_stop_gradient(self) -> bool:
        return self._is_dict_obs

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
            pi_arch = list(actor_hidden_dims) if actor_hidden_dims is not None else [256, 256, 256]
            qf_arch = list(critic_hidden_dims) if critic_hidden_dims is not None else list(pi_arch)
            return {"pi": pi_arch, "qf": qf_arch}

        return [256, 256, 256]

    def _setup_model(self) -> None:
        features_extractor = self._build_features_extractor()
        self.policy = self._build_policy(features_extractor).to(self.device)

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

        # Entropy coefficient (auto-tuned by default).
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

    def _current_alpha(self) -> torch.Tensor:
        if self.autotune:
            return self.log_alpha.exp()
        return self._fixed_alpha

    def _td_loss(self, data, q_pred: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        target_q = self._target_q(data)
        q_loss = sum(F.mse_loss(q, target_q) for q in q_pred)
        return q_loss, {
            "td_loss": q_loss.item(),
            "target_q": target_q.mean().item(),
        }

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
