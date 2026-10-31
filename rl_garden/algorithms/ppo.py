"""Proximal Policy Optimization for Box and Dict observations."""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any, Iterator, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from rl_garden.algorithms._decision_mask import (
    DecisionMaskReader,
    warn_if_decision_keys_ignored,
)
from rl_garden.algorithms._observation import EncoderSharing
from rl_garden.algorithms.on_policy import OnPolicyAlgorithm
from rl_garden.buffers.rollout_buffer import RolloutBuffer, RolloutBufferSample
from rl_garden.common.ddp import allreduce_grads, allreduce_mean, is_ddp_active
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks.actor_critic import gaussian_kl_divergence
from rl_garden.observations import ObsGroups
from rl_garden.policies.ppo_policy import PPOPolicy


def ppo_clip_policy_loss(
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    clip_coef: float,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """PPO's clipped surrogate objective: ``mean(max(-adv*ratio, -adv*clip(ratio)))``.

    Shared between ``PPO._policy_loss`` and ``BPPOCore`` (``bppo.py``) --
    BPPO reuses only this formula, not ``_ppo_minibatch_update``, since it
    normalizes/omega-weights advantages in a different order and has no
    rollout buffer or value-clipping to hook into.

    ``mask`` (bool, same shape as ``advantages``) averages over the true rows
    only; masked rows contribute exactly zero (0 when no row is true).
    """
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    if mask is None:
        return torch.max(pg_loss1, pg_loss2).mean()
    return _masked_mean(torch.max(pg_loss1, pg_loss2), mask, mask.sum())


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over the rows where ``mask`` is true; 0 when none is.
    ``torch.where`` (not a multiply) keeps non-finite masked rows out."""
    return torch.where(mask, x, 0.0).sum() / count.clamp_min(1)


def _normalize_masked_advantages(
    advantages: torch.Tensor, mask: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    """Standardize ``advantages`` with the mean and unbiased std of the true
    rows. Left unchanged when fewer than two rows are true (tensor ``where``,
    no host branch)."""
    mean = _masked_mean(advantages, mask, count)
    variance = torch.where(mask, (advantages - mean) ** 2, 0.0).sum() / (count - 1).clamp_min(1)
    normalized = (advantages - mean) / (variance.sqrt() + 1e-8)
    return torch.where(count > 1, normalized, advantages)


def _as_float_list_or_none(values: Optional[Sequence[float]]) -> Optional[list[float]]:
    """Normalize tuple/list init options so checkpoint metadata round-trips."""
    return None if values is None else [float(v) for v in values]


class PPO(OnPolicyAlgorithm):
    """SB3/ManiSkill-style PPO with rl-garden feature extractors."""

    _compatible_checkpoint_algorithms = ("PPO",)
    # decision_level=True reads the env's per-step decision keys and computes
    # SMDP GAE over decision steps (see rl_garden/algorithms/_decision_mask.py).
    _supports_decision_level: bool = True
    _SUPPORTED_POLICY_KWARGS = frozenset(
        {
            "actor_extractor_class",
            "actor_extractor_kwargs",
            "critic_extractor_class",
            "critic_extractor_kwargs",
        }
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
        lr_schedule: Literal[
            "constant", "linear_warmup", "warmup_cosine", "adaptive_kl"
        ] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        desired_kl: float = 0.01,
        adaptive_lr_min: float = 1e-5,
        adaptive_lr_max: float = 1e-2,
        net_arch: Optional[Sequence[int] | dict[str, Sequence[int]]] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        value_hidden_dims: Optional[Sequence[int]] = None,
        log_std_init: float = -0.5,
        log_std_init_per_dim: Optional[Sequence[float]] = None,
        action_mean_bias_init: Optional[Sequence[float]] = None,
        action_mean_head_gain: Optional[float] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
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
        critic_backbone_type: Optional[Literal["mlp", "mlp_resnet"]] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        finite_horizon_gae: bool = False,
        bootstrap_at_done: str = "truncated",
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
        decision_level: bool = False,
        per_step_critic: bool = True,
    ) -> None:
        """``decision_level=True`` reads the env's per-step decision keys:
        returns, advantages and the policy loss are computed over decision
        steps. ``per_step_critic`` only matters for a rollout window that
        carries a decision mask (it has no effect when ``decision_level`` is
        off): True trains the critic on every step, False on decision steps
        only. See ``docs/guides/decision-level-ppo.md`` for the env contract
        and the training rules."""
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
            bootstrap_at_done=bootstrap_at_done,
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
        if lr_schedule == "adaptive_kl" and anneal_lr:
            raise ValueError(
                "anneal_lr=True is incompatible with lr_schedule='adaptive_kl': "
                "_step_lr()'s anneal_lr branch unconditionally overwrites the "
                "optimizer's lr from self.learning_rate every train() call, "
                "silently clobbering the adaptive-KL mutation."
            )
        if is_ddp_active() and target_kl is not None:
            raise ValueError(
                "target_kl is not None under an active torch.distributed "
                "process group: target_kl's early-stop is a per-rank decision "
                "computed from that rank's own local minibatch data. If one "
                "rank stops early while another continues to the next "
                "minibatch's backward()+grad-allreduce, ranks call "
                "all_reduce a different number of times and the job "
                "deadlocks. Pass target_kl=None for multi-GPU runs (rsl_rl "
                "has no early-stop mechanism at all, for the same reason)."
            )
        if decision_level and not self._supports_decision_level:
            raise ValueError(f"{type(self).__name__} does not support decision_level=True.")
        if decision_level and finite_horizon_gae:
            raise ValueError(
                "decision_level=True is incompatible with finite_horizon_gae=True: "
                "SMDP GAE is implemented for the standard GAE recurrence only."
            )
        self.decision_level = decision_level
        self.per_step_critic = per_step_critic
        self._decision_keys_warned = False
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
        self.lr_schedule: ScheduleType | Literal["adaptive_kl"] = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.desired_kl = desired_kl
        self.adaptive_lr_min = adaptive_lr_min
        self.adaptive_lr_max = adaptive_lr_max
        self.net_arch = self._resolve_net_arch(
            net_arch=net_arch,
            actor_hidden_dims=actor_hidden_dims,
            value_hidden_dims=value_hidden_dims,
        )
        self.log_std_init = log_std_init
        self.log_std_init_per_dim = _as_float_list_or_none(log_std_init_per_dim)
        self.action_mean_bias_init = _as_float_list_or_none(action_mean_bias_init)
        self.action_mean_head_gain = action_mean_head_gain
        self.actor_use_layer_norm = actor_use_layer_norm
        self.value_use_layer_norm = value_use_layer_norm
        self.actor_use_group_norm = actor_use_group_norm
        self.value_use_group_norm = value_use_group_norm
        self.num_groups = num_groups
        self.actor_dropout_rate = actor_dropout_rate
        self.value_dropout_rate = value_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.critic_backbone_type = critic_backbone_type

        self.encoder_sharing = encoder_sharing
        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self.critic_encoder_config = critic_encoder_config

        self.policy_kwargs = self._normalize_policy_kwargs(policy_kwargs)
        # Built here rather than in _setup_model(): SequencePPO replaces
        # _setup_model() wholesale and needs the same reader.
        self._decision_reader = (
            DecisionMaskReader(
                self.num_envs, tuple(self.env.single_action_space.shape), self.device
            )
            if decision_level
            else None
        )
        self._setup_model()

    def _rollout_policy(
        self, obs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_obs = self._obs_to_policy_device(obs)
        self.policy.update_normalizer(policy_obs)
        if self.decision_level:
            return self._decision_level_rollout_policy(policy_obs)
        # No explicit stop_gradient_actor: PPOPolicy applies
        # BasePolicy.extract_actor_features's encoder_sharing rule by
        # default (see rl_garden/policies/base.py); both call sites are
        # under torch.no_grad() anyway (rollout, no training gradients).
        if self.lr_schedule == "adaptive_kl":
            with torch.no_grad():
                actions, values, log_prob, entropy, mean, log_std = (
                    self.policy.act_with_value_logprob_and_dist_params(
                        policy_obs,
                        deterministic=False,
                    )
                )
            self._rollout_mean, self._rollout_log_std = mean, log_std
            return actions, values, log_prob, entropy
        with torch.no_grad():
            return self.policy(
                policy_obs,
                deterministic=False,
            )

    def _decision_level_rollout_policy(
        self, policy_obs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``_rollout_policy`` under ``decision_level``: the policy returns
        per-component log-probs, kept for a step whose env reports a
        component mask (see ``_sum_rollout_components``)."""
        with torch.no_grad():
            if self.lr_schedule == "adaptive_kl":
                actions, values, log_prob, entropy, mean, log_std = (
                    self.policy.act_with_value_logprob_and_dist_params(
                        policy_obs,
                        deterministic=False,
                        sum_dims=False,
                    )
                )
                self._rollout_mean, self._rollout_log_std = mean, log_std
            else:
                actions, values, log_prob, entropy = self.policy(
                    policy_obs,
                    deterministic=False,
                    sum_dims=False,
                )
        return actions, values, *self._sum_rollout_components(log_prob, entropy)

    def _sum_rollout_components(
        self, log_prob: torch.Tensor, entropy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stash the ``(N, A)`` per-component rollout log-probs and return the
        per-step ``(N, 1)`` sums: the same reduction the actor applies with
        ``sum_dims=True``, so the summed values are bit-identical to it."""
        self._rollout_component_log_probs = log_prob
        return log_prob.sum(-1, keepdim=True), entropy.sum(-1, keepdim=True)

    def _extra_rollout_buffer_kwargs(self, infos: dict | None = None) -> dict:
        out = (
            {"mean": self._rollout_mean, "log_std": self._rollout_log_std}
            if self.lr_schedule == "adaptive_kl"
            else {}
        )
        if infos is None:
            return out
        if not self.decision_level:
            if not self._decision_keys_warned:
                self._decision_keys_warned = warn_if_decision_keys_ignored(infos)
            return out
        masks = self._decision_reader.read(infos)
        if masks is not None:
            decision, component_mask = masks
            out["decision"] = decision
            if component_mask is not None:
                out["component_mask"] = component_mask
                out["component_log_probs"] = self._rollout_component_log_probs
        return out

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
            "desired_kl": self.desired_kl,
            "adaptive_lr_min": self.adaptive_lr_min,
            "adaptive_lr_max": self.adaptive_lr_max,
            "net_arch": self.net_arch,
            "log_std_init": self.log_std_init,
            "log_std_init_per_dim": self.log_std_init_per_dim,
            "action_mean_bias_init": self.action_mean_bias_init,
            "action_mean_head_gain": self.action_mean_head_gain,
            "decision_level": self.decision_level,
            "per_step_critic": self.per_step_critic,
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

    def _setup_model(self) -> None:
        self.policy = PPOPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            net_arch=self.net_arch,
            log_std_init=self.log_std_init,
            log_std_init_per_dim=self.log_std_init_per_dim,
            action_mean_bias_init=self.action_mean_bias_init,
            action_mean_head_gain=self.action_mean_head_gain,
            actor_use_layer_norm=self.actor_use_layer_norm,
            value_use_layer_norm=self.value_use_layer_norm,
            actor_use_group_norm=self.actor_use_group_norm,
            value_use_group_norm=self.value_use_group_norm,
            num_groups=self.num_groups,
            actor_dropout_rate=self.actor_dropout_rate,
            value_dropout_rate=self.value_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            critic_backbone_type=self.critic_backbone_type,
            **self._policy_extractor_kwargs(self.env.single_observation_space),
        ).to(self.device)
        self.policy_optimizer = make_optimizer(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self._lr_scheduler = (
            None
            if self.lr_schedule == "adaptive_kl"
            else make_lr_scheduler(
                self.policy_optimizer,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
        )
        self.rollout_buffer = RolloutBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            store_dist_params=(self.lr_schedule == "adaptive_kl"),
            store_decisions=self.decision_level,
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
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return ppo_clip_policy_loss(advantages, ratio, clip_coef, mask=mask)

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
        old_mean: Optional[torch.Tensor] = None,
        old_log_std: Optional[torch.Tensor] = None,
        new_mean: Optional[torch.Tensor] = None,
        new_log_std: Optional[torch.Tensor] = None,
        decision_mask: torch.Tensor | None = None,
        component_mask: torch.Tensor | None = None,
    ) -> dict[str, float | bool | torch.Tensor]:
        """One PPO minibatch gradient step on 1-D tensors. ``result["stop"]`` is
        True if target_kl triggered an early stop (caller should break after
        recording ``clipfrac``, matching the pre-refactor loop's behavior of
        recording clipfrac for the aborting minibatch but no other metric).

        ``old_mean``/``old_log_std``/``new_mean``/``new_log_std`` are only
        passed when ``lr_schedule == "adaptive_kl"``, in which case this also
        mutates ``self.policy_optimizer``'s LR in place per rsl_rl's exact
        adaptive-KL rule, before the gradient step below. Note the early
        return above (on ``target_kl``) skips the LR update for that
        minibatch too -- rsl_rl has no early-stop mechanism to interact with,
        so pass ``target_kl=None`` for a numeric parity run against it.

        ``decision_mask`` ((B,) bool, same device) restricts advantage
        normalization, the policy loss, entropy and the KL statistics to
        decision rows; the value loss covers every row when
        ``per_step_critic`` and the decision rows otherwise. Masked rows are
        replaced by finite values before any arithmetic, so they contribute
        exactly zero to the loss and its gradients. Each DDP rank normalizes
        by its own decision count. ``result["decisions"]`` is that count, a
        device tensor.

        ``component_mask`` ((B, A) bool, only together with ``decision_mask``)
        marks the action components that took effect; ``log_prob`` and
        ``entropy`` are then per component, (B, A), and ``old_log_prob`` is
        already the sum over the components that took effect. The log-ratio,
        entropy and adaptive-KL terms of a step sum its components that took
        effect, so a component that never took effect gets no policy
        gradient; everything after that reduction is the decision-mask path."""
        if component_mask is not None and decision_mask is None:
            raise ValueError("component_mask requires decision_mask.")
        if decision_mask is None:
            if self.norm_adv and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            logratio = log_prob - old_log_prob
            row_mean = torch.mean
            decision_stats = {}
        else:
            decisions = decision_mask.sum()

            def row_mean(x: torch.Tensor) -> torch.Tensor:
                return _masked_mean(x, decision_mask, decisions)

            if component_mask is not None:
                # ``where`` before the sum: components that did not take
                # effect contribute exactly zero, gradients included.
                log_prob = torch.where(component_mask, log_prob, 0.0).sum(-1)
                entropy = torch.where(component_mask, entropy, 0.0).sum(-1)
            advantages = torch.where(decision_mask, advantages, 0.0)
            logratio = torch.where(decision_mask, log_prob - old_log_prob, 0.0)
            if self.norm_adv:
                advantages = _normalize_masked_advantages(advantages, decision_mask, decisions)
            decision_stats = {"decisions": decisions}

        ratio = logratio.exp()
        with torch.no_grad():
            old_approx_kl = row_mean(-logratio)
            approx_kl = row_mean((ratio - 1.0) - logratio)
            clipfrac = row_mean(((ratio - 1.0).abs() > clip_coef).float()).item()
        if self.target_kl is not None and approx_kl > self.target_kl:
            return {"stop": True, "clipfrac": clipfrac, **decision_stats}

        if self.lr_schedule == "adaptive_kl":
            with torch.no_grad():
                if component_mask is None:
                    kl = gaussian_kl_divergence(old_mean, old_log_std, new_mean, new_log_std)
                else:
                    kl = torch.where(
                        component_mask,
                        gaussian_kl_divergence(
                            old_mean, old_log_std, new_mean, new_log_std, sum_dims=False
                        ),
                        0.0,
                    ).sum(-1)
                kl_mean = row_mean(kl)
                kl_mean = allreduce_mean(kl_mean)
                current_lr = self.policy_optimizer.param_groups[0]["lr"]
                if kl_mean > self.desired_kl * 2.0:
                    new_lr = max(self.adaptive_lr_min, current_lr / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    new_lr = min(self.adaptive_lr_max, current_lr * 1.5)
                else:
                    new_lr = current_lr
                for group in self.policy_optimizer.param_groups:
                    group["lr"] = new_lr

        policy_loss = self._policy_loss(
            advantages=advantages, ratio=ratio, clip_coef=clip_coef, mask=decision_mask
        )
        if decision_mask is not None:
            value_loss = self._masked_value_loss(
                values=values,
                old_values=old_values,
                returns=returns,
                decision_mask=decision_mask,
                clip_coef=clip_coef,
            )
        elif self.clip_vloss:
            v_loss_unclipped = (values - returns) ** 2
            values_clipped = old_values + torch.clamp(values - old_values, -clip_coef, clip_coef)
            v_loss_clipped = (values_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            value_loss = 0.5 * F.mse_loss(values, returns)
        entropy_loss = row_mean(entropy)
        loss = policy_loss - self.ent_coef * entropy_loss + self.vf_coef * value_loss

        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        allreduce_grads(self.policy)
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
            **decision_stats,
        }

    def _masked_value_loss(
        self,
        *,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        decision_mask: torch.Tensor,
        clip_coef: float,
    ) -> torch.Tensor:
        """Value loss of a window with a decision mask, over every row when
        ``per_step_critic`` and over decision rows otherwise. One formula for
        both cases, so with an all-true mask they give identical losses."""
        value_mask = torch.ones_like(decision_mask) if self.per_step_critic else decision_mask
        detached_values = values.detach()
        returns = torch.where(value_mask, returns, detached_values)
        v_loss = (values - returns) ** 2
        if self.clip_vloss:
            old_values = torch.where(value_mask, old_values, detached_values)
            values_clipped = old_values + torch.clamp(values - old_values, -clip_coef, clip_coef)
            v_loss = torch.max(v_loss, (values_clipped - returns) ** 2)
        return 0.5 * _masked_mean(v_loss, value_mask, value_mask.sum())

    def _minibatch_masks(self, data) -> dict[str, torch.Tensor]:
        """Mask kwargs of ``_ppo_minibatch_update`` for one minibatch: empty
        unless the window carries decisions. ``reshape(-1)`` is a no-op on flat
        samples and T-major on ``(T, mb_envs)`` sequence samples, the same
        order as ``_evaluate_minibatch``'s flattening; the component mask keeps
        its component axis, ``(B, A)``."""
        if not self.rollout_buffer.window_has_decisions:
            return {}
        masks = {"decision_mask": data.decision_mask.reshape(-1)}
        if self.rollout_buffer.window_has_components:
            action_dim = math.prod(self.env.single_action_space.shape)
            masks["component_mask"] = data.component_mask.reshape(-1, action_dim)
        return masks

    def _iter_minibatches(self) -> Iterator[RolloutBufferSample]:
        return self.rollout_buffer.get(self.minibatch_size)

    def _evaluate_minibatch(
        self, data: RolloutBufferSample
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Returns 1-D (values, log_prob, entropy, old_values, old_log_prob,
        advantages, returns, old_mean, old_log_std, new_mean, new_log_std).
        The last four are ``None`` unless ``lr_schedule == "adaptive_kl"``.
        In a window with a component mask ``log_prob`` and ``entropy`` are
        per component, (B, A)."""
        if self.rollout_buffer.window_has_components:
            return self._evaluate_minibatch_per_component(data)
        # No explicit stop_gradient_actor: PPOPolicy applies
        # BasePolicy.extract_actor_features's encoder_sharing rule by default.
        if self.lr_schedule == "adaptive_kl":
            values, log_prob, entropy, new_mean, new_log_std = (
                self.policy.evaluate_actions_with_dist_params(data.obs, data.actions)
            )
            old_mean, old_log_std = data.old_mean, data.old_log_std
        else:
            values, log_prob, entropy = self.policy.evaluate_actions(data.obs, data.actions)
            new_mean = new_log_std = old_mean = old_log_std = None
        return (
            values.flatten(), log_prob.flatten(), entropy.flatten(),
            data.old_values, data.old_log_prob, data.advantages, data.returns,
            old_mean, old_log_std, new_mean, new_log_std,
        )

    @staticmethod
    def _component_masked_actions(data) -> torch.Tensor:
        """Stored actions with the components that did not take effect set to
        0. Their log-probs are dropped by ``where`` anyway, but an arbitrary
        value there could make that log-prob non-finite, and the zero gradient
        ``where`` sends back would then turn into NaN (``0 * inf``)."""
        return torch.where(data.component_mask, data.actions, 0.0)

    def _evaluate_minibatch_per_component(self, data: RolloutBufferSample):
        """``_evaluate_minibatch`` for a window with a component mask:
        ``log_prob`` and ``entropy`` keep their component axis, (B, A)."""
        actions = self._component_masked_actions(data)
        if self.lr_schedule == "adaptive_kl":
            values, log_prob, entropy, new_mean, new_log_std = (
                self.policy.evaluate_actions_with_dist_params(data.obs, actions, sum_dims=False)
            )
            old_mean, old_log_std = data.old_mean, data.old_log_std
        else:
            values, log_prob, entropy = self.policy.evaluate_actions(
                data.obs, actions, sum_dims=False
            )
            new_mean = new_log_std = old_mean = old_log_std = None
        action_dim = log_prob.shape[-1]
        return (
            values.flatten(), log_prob.reshape(-1, action_dim), entropy.reshape(-1, action_dim),
            data.old_values, data.old_log_prob, data.advantages, data.returns,
            old_mean, old_log_std, new_mean, new_log_std,
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
        decision_counts: list[torch.Tensor] = []
        continue_training = True

        for _ in range(self.update_epochs):
            for data in self._iter_minibatches():
                (
                    values, log_prob, entropy, old_values, old_log_prob, advantages, returns,
                    old_mean, old_log_std, new_mean, new_log_std,
                ) = self._evaluate_minibatch(data)
                result = self._ppo_minibatch_update(
                    values=values,
                    log_prob=log_prob,
                    entropy=entropy,
                    old_values=old_values,
                    old_log_prob=old_log_prob,
                    advantages=advantages,
                    returns=returns,
                    clip_coef=clip_coef,
                    old_mean=old_mean,
                    old_log_std=old_log_std,
                    new_mean=new_mean,
                    new_log_std=new_log_std,
                    **self._minibatch_masks(data),
                )
                if "decisions" in result:
                    decision_counts.append(result.pop("decisions"))

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
        if self.rollout_buffer.window_has_decisions and not self.per_step_critic:
            # Explained variance over the rows the critic is trained on.
            value_rows = self.rollout_buffer.decisions.reshape(-1)
            b_values, b_returns = b_values[value_rows], b_returns[value_rows]
        y_var = torch.var(b_returns)
        explained_var = (
            float("nan")
            if y_var == 0
            else float((1 - torch.var(b_returns - b_values) / y_var).item())
        )
        metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "old_approx_kl": float(np.mean(old_approx_kls)) if old_approx_kls else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_variance": explained_var,
            "clip_coef": clip_coef,
            "learning_rate": float(self.policy_optimizer.param_groups[0]["lr"]),
        }
        if decision_counts:
            counts = torch.stack(decision_counts).float()
            (
                metrics["decisions_per_minibatch_min"],
                metrics["decisions_per_minibatch_mean"],
            ) = torch.stack([counts.min(), counts.mean()]).tolist()
        return metrics

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        super()._log_update_metrics(metrics, step)
        if self.logger is None or not self.decision_level:
            return
        if self.rollout_buffer.window_has_decisions:
            # The buffer still holds the window train() just used.
            env_steps = self.rollout_buffer.buffer_size
            decisions = self.rollout_buffer.decisions.sum().float()
            count, fraction, env_steps_per_decision = torch.stack(
                [decisions, decisions / env_steps, env_steps / decisions.clamp_min(1)]
            ).tolist()
            self._log_rollout_metric("decisions", count, step)
            self._log_rollout_metric("decision_fraction", fraction, step)
            self._log_rollout_metric("env_steps_per_decision", env_steps_per_decision, step)
        if self.rollout_buffer.window_has_components:
            self._log_component_metrics(step)
        log_stds = self.policy.actor.log_std.detach().view(-1).tolist()
        for dim, log_std in enumerate(log_stds):
            self.logger.add_scalar(f"policy/log_std/{dim}", log_std, step)

    def _log_component_metrics(self, step: int) -> None:
        """``train/component_usage/<j>``: fraction of env steps on which
        component j took effect; ``train/components_per_decision``: mean
        number of components that took effect per decision step."""
        buffer = self.rollout_buffer
        component_mask = buffer.component_mask.reshape(buffer.num_steps, buffer.num_envs, -1)
        usage = component_mask.float().mean((0, 1))
        components_per_decision = torch.where(
            buffer.decisions, component_mask.sum(-1), 0
        ).sum() / buffer.decisions.sum().clamp_min(1)
        *usage_values, per_decision = torch.cat(
            [usage, components_per_decision.float().reshape(1)]
        ).tolist()
        for component, value in enumerate(usage_values):
            self._log_rollout_metric(f"component_usage/{component}", value, step)
        self._log_rollout_metric("components_per_decision", per_decision, step)

    def learn(self, total_timesteps: int) -> "PPO":
        self._target_timesteps = total_timesteps
        return super().learn(total_timesteps=total_timesteps)
