"""Proximal Policy Optimization for Box and Dict observations."""

from __future__ import annotations

import warnings
from typing import Any, Iterator, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.on_policy import OnPolicyAlgorithm
from rl_garden.buffers.rollout_buffer import DictRolloutBuffer, RolloutBuffer, RolloutBufferSample
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.combined import (
    CombinedExtractor,
    ImageEncoderFactory,
    default_image_encoder_factory,
)
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.ppo_policy import PPOPolicy


class PPO(OnPolicyAlgorithm):
    """SB3/ManiSkill-style PPO with rl-garden feature extractors."""

    _compatible_checkpoint_algorithms = ("PPO",)
    _SUPPORTED_POLICY_KWARGS = frozenset(
        {"features_extractor_class", "features_extractor_kwargs"}
    )

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        num_steps: int = 50,
        gamma: float = 0.8,
        gae_lambda: float = 0.9,
        learning_rate: float = 3e-4,
        num_minibatches: int = 32,
        update_epochs: int = 4,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        clip_vloss: bool = False,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.1,
        anneal_lr: bool = False,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        value_hidden_dims: Optional[Sequence[int]] = None,
        log_std_init: float = -0.5,
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: Optional[tuple[str, ...]] = None,
        state_key: Optional[str] = None,
        use_proprio: Optional[bool] = None,
        proprio_latent_dim: Optional[int] = None,
        image_fusion_mode: Optional[str] = None,
        enable_stacking: Optional[bool] = None,
        detach_encoder_on_actor: Optional[bool] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        actor_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        value_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        kernel_init: Optional[
            Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
        ] = None,
        backbone_type: Literal["mlp", "mlp_resnet"] = "mlp",
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        finite_horizon_gae: bool = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            eval_env=eval_env,
            num_steps=num_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            finite_horizon_gae=finite_horizon_gae,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_final_checkpoint=save_final_checkpoint,
        )
        if num_minibatches <= 0:
            raise ValueError(
                f"num_minibatches must be positive, got {num_minibatches}."
            )
        if update_epochs <= 0:
            raise ValueError(f"update_epochs must be positive, got {update_epochs}.")
        if self.batch_size <= 1 and norm_adv:
            raise ValueError("num_steps * num_envs must be > 1 when norm_adv=True.")
        if clip_coef <= 0:
            raise ValueError(f"clip_coef must be positive, got {clip_coef}.")
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}.")
        self.learning_rate = learning_rate
        self.num_minibatches = num_minibatches
        self.minibatch_size = max(1, self.batch_size // num_minibatches)
        if self.batch_size % self.minibatch_size != 0:
            warnings.warn(
                "PPO batch_size is not divisible by minibatch_size; the last minibatch "
                "will be smaller.",
                UserWarning,
                stacklevel=2,
            )
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.anneal_lr = anneal_lr
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.net_arch = self._resolve_net_arch(
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            value_hidden_dims=value_hidden_dims,
        )
        self.log_std_init = log_std_init
        self.actor_use_layer_norm = actor_use_layer_norm
        self.value_use_layer_norm = value_use_layer_norm
        self.actor_use_group_norm = actor_use_group_norm
        self.value_use_group_norm = value_use_group_norm
        self.num_groups = num_groups
        self.actor_dropout_rate = actor_dropout_rate
        self.value_dropout_rate = value_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type

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
                    "PPO with Box observation space does not accept image-related "
                    f"kwargs (got {explicitly_set}). Use Dict observations instead."
                )
            self._is_dict_obs = False
            self.detach_encoder_on_actor = (
                False if detach_encoder_on_actor is None else detach_encoder_on_actor
            )
        elif isinstance(obs_space, spaces.Dict):
            self._is_dict_obs = True
            self.detach_encoder_on_actor = (
                True if detach_encoder_on_actor is None else detach_encoder_on_actor
            )
            self._image_encoder_factory = (
                image_encoder_factory or default_image_encoder_factory()
            )
            self._image_keys = (
                image_keys if image_keys is not None else ("rgb", "depth")
            )
            self._state_key = state_key if state_key is not None else "state"
            self._use_proprio = use_proprio if use_proprio is not None else True
            self._proprio_latent_dim = (
                proprio_latent_dim if proprio_latent_dim is not None else 64
            )
            self._image_fusion_mode = (
                image_fusion_mode if image_fusion_mode is not None else "stack_channels"
            )
            self._enable_stacking = (
                enable_stacking if enable_stacking is not None else False
            )
        else:
            raise TypeError(
                f"PPO supports Box or Dict observations, got {type(obs_space)}"
            )

        self.policy_kwargs = self._normalize_policy_kwargs(policy_kwargs)
        self._setup_model()

    def _actor_stop_gradient(self) -> bool:
        return bool(self.detach_encoder_on_actor)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = {
            **super()._checkpoint_metadata(),
            "learning_rate": self.learning_rate,
            "num_minibatches": self.num_minibatches,
            "update_epochs": self.update_epochs,
            "norm_adv": self.norm_adv,
            "clip_coef": self.clip_coef,
            "clip_vloss": self.clip_vloss,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "anneal_lr": self.anneal_lr,
            "weight_decay": self.weight_decay,
            "use_adamw": self.use_adamw,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "net_arch": self.net_arch,
            "log_std_init": self.log_std_init,
            "detach_encoder_on_actor": self.detach_encoder_on_actor,
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
        return {
            "lr_scheduler_state": (
                self._lr_scheduler.state_dict()
                if self._lr_scheduler is not None
                else None
            )
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        sched_state = state.get("lr_scheduler_state")
        if self._lr_scheduler is not None and sched_state is not None:
            self._lr_scheduler.load_state_dict(sched_state)

    @staticmethod
    def _resolve_net_arch(
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]],
        actor_hidden_dims: Optional[Sequence[int]],
        value_hidden_dims: Optional[Sequence[int]],
    ) -> Sequence[int] | dict[str, list[int]]:
        if net_arch is not None:
            if actor_hidden_dims is not None or value_hidden_dims is not None:
                warnings.warn(
                    "actor_hidden_dims/value_hidden_dims are ignored when net_arch "
                    "is provided. Use net_arch only.",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if isinstance(net_arch, dict):
                if "pi" not in net_arch or "vf" not in net_arch:
                    raise ValueError("PPO net_arch dict must contain 'pi' and 'vf'.")
                return {"pi": list(net_arch["pi"]), "vf": list(net_arch["vf"])}
            return list(net_arch)
        if actor_hidden_dims is not None or value_hidden_dims is not None:
            warnings.warn(
                "actor_hidden_dims/value_hidden_dims are deprecated. Use "
                "net_arch=list[...] or net_arch={'pi': [...], 'vf': [...]} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            pi_arch = (
                list(actor_hidden_dims)
                if actor_hidden_dims is not None
                else [256, 256, 256]
            )
            vf_arch = (
                list(value_hidden_dims)
                if value_hidden_dims is not None
                else list(pi_arch)
            )
            return {"pi": pi_arch, "vf": vf_arch}
        return [256, 256, 256]

    def _default_features_extractor_class(self) -> type[BaseFeaturesExtractor]:
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Box):
            return FlattenExtractor
        if isinstance(obs_space, spaces.Dict):
            return CombinedExtractor
        raise TypeError(f"PPO supports Box or Dict observations, got {type(obs_space)}")

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
            raise TypeError(
                "policy_kwargs['features_extractor_kwargs'] must be a dict."
            )
        normalized["features_extractor_kwargs"] = dict(features_extractor_kwargs)
        return normalized

    def _resolve_policy_kwargs(self) -> dict[str, Any]:
        default_class = self._default_features_extractor_class()
        default_kwargs = dict(self._default_features_extractor_kwargs())
        resolved = {
            "features_extractor_class": default_class,
            "features_extractor_kwargs": default_kwargs,
        }
        if "features_extractor_class" in self.policy_kwargs:
            resolved["features_extractor_class"] = self.policy_kwargs[
                "features_extractor_class"
            ]
            if resolved["features_extractor_class"] is not default_class:
                resolved["features_extractor_kwargs"] = {}
        if "features_extractor_kwargs" in self.policy_kwargs:
            if resolved["features_extractor_class"] is default_class:
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

    def _setup_model(self) -> None:
        features_extractor = self._build_features_extractor()
        self.policy = PPOPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            log_std_init=self.log_std_init,
            actor_use_layer_norm=self.actor_use_layer_norm,
            value_use_layer_norm=self.value_use_layer_norm,
            actor_use_group_norm=self.actor_use_group_norm,
            value_use_group_norm=self.value_use_group_norm,
            num_groups=self.num_groups,
            actor_dropout_rate=self.actor_dropout_rate,
            value_dropout_rate=self.value_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
        ).to(self.device)
        self.policy_optimizer = make_optimizer(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self._lr_scheduler = make_lr_scheduler(
            self.policy_optimizer,
            schedule_type=self.lr_schedule,
            warmup_steps=self.lr_warmup_steps,
            decay_steps=self.lr_decay_steps,
            min_lr_ratio=self.lr_min_ratio,
        )
        buffer_cls = DictRolloutBuffer if self._is_dict_obs else RolloutBuffer
        self.rollout_buffer = buffer_cls(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

    def _current_clip_coef(self) -> float:
        return self.clip_coef

    def _step_lr(self) -> None:
        if self.anneal_lr:
            # Decay over completed policy updates. total_timesteps is not known
            # inside train(), so this simple schedule remains update-count based.
            frac = max(0.0, 1.0 - self._global_step / max(1, self._target_timesteps))
            for group in self.policy_optimizer.param_groups:
                group["lr"] = frac * self.learning_rate
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def _policy_loss(
        self,
        *,
        advantages: torch.Tensor,
        ratio: torch.Tensor,
        clip_coef: float,
    ) -> torch.Tensor:
        # TODO(agent): BPPO/Uni-O4 can override this hook to inject behavior-policy
        # advantages, asymmetric advantage weighting, or KL regularization.
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        return torch.max(pg_loss1, pg_loss2).mean()

    def _ppo_minibatch_update(
        self,
        *,
        values: torch.Tensor,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        old_values: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        clip_coef: float,
    ) -> dict[str, float | bool]:
        """One PPO minibatch gradient step on 1-D tensors. ``result["stop"]`` is
        True if target_kl triggered an early stop (caller should break after
        recording ``clipfrac``, matching the pre-refactor loop's behavior of
        recording clipfrac for the aborting minibatch but no other metric)."""
        if self.norm_adv and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logratio = log_prob - old_log_prob
        ratio = logratio.exp()
        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1.0) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()
        if self.target_kl is not None and approx_kl > self.target_kl:
            return {"stop": True, "clipfrac": clipfrac}

        policy_loss = self._policy_loss(advantages=advantages, ratio=ratio, clip_coef=clip_coef)
        if self.clip_vloss:
            v_loss_unclipped = (values - returns) ** 2
            values_clipped = old_values + torch.clamp(values - old_values, -clip_coef, clip_coef)
            v_loss_clipped = (values_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * F.mse_loss(values, returns)
        entropy_loss = entropy.mean()
        loss = policy_loss - self.ent_coef * entropy_loss + self.vf_coef * value_loss

        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        return {
            "stop": False,
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "entropy_loss": float(entropy_loss.detach().item()),
            "approx_kl": float(approx_kl.detach().item()),
            "old_approx_kl": float(old_approx_kl.detach().item()),
            "clipfrac": clipfrac,
            "loss": float(loss.detach().item()),
        }

    def _iter_minibatches(self) -> Iterator[RolloutBufferSample]:
        return self.rollout_buffer.get(self.minibatch_size)

    def _evaluate_minibatch(
        self, data: RolloutBufferSample
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns 1-D (values, log_prob, entropy, old_values, old_log_prob, advantages, returns)."""
        values, log_prob, entropy = self.policy.evaluate_actions(
            data.obs, data.actions, stop_gradient_actor=self._actor_stop_gradient()
        )
        return (
            values.flatten(), log_prob.flatten(), entropy.flatten(),
            data.old_values, data.old_log_prob, data.advantages, data.returns,
        )

    def train(self) -> dict[str, float]:
        self.policy.train()
        self._global_update += 1
        self._target_timesteps = max(
            getattr(self, "_target_timesteps", 1), self._global_step
        )
        clip_coef = self._current_clip_coef()

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kls: list[float] = []
        old_approx_kls: list[float] = []
        clipfracs: list[float] = []
        losses: list[float] = []
        continue_training = True

        for _ in range(self.update_epochs):
            for data in self._iter_minibatches():
                values, log_prob, entropy, old_values, old_log_prob, advantages, returns = (
                    self._evaluate_minibatch(data)
                )
                result = self._ppo_minibatch_update(
                    values=values,
                    log_prob=log_prob,
                    entropy=entropy,
                    old_values=old_values,
                    old_log_prob=old_log_prob,
                    advantages=advantages,
                    returns=returns,
                    clip_coef=clip_coef,
                )

                clipfracs.append(result["clipfrac"])
                if result["stop"]:
                    continue_training = False
                    break

                policy_losses.append(result["policy_loss"])
                value_losses.append(result["value_loss"])
                entropy_losses.append(result["entropy_loss"])
                approx_kls.append(result["approx_kl"])
                old_approx_kls.append(result["old_approx_kl"])
                losses.append(result["loss"])
            if not continue_training:
                break

        self._step_lr()
        b_values = self.rollout_buffer.values.reshape(-1)
        b_returns = self.rollout_buffer.returns.reshape(-1)
        y_var = torch.var(b_returns)
        explained_var = (
            torch.nan if y_var == 0 else 1 - torch.var(b_returns - b_values) / y_var
        )
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "old_approx_kl": float(np.mean(old_approx_kls)) if old_approx_kls else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_variance": float(explained_var.detach().item()),
            "clip_coef": clip_coef,
            "learning_rate": float(self.policy_optimizer.param_groups[0]["lr"]),
        }

    def learn(self, total_timesteps: int) -> "PPO":
        self._target_timesteps = total_timesteps
        return super().learn(total_timesteps=total_timesteps)
