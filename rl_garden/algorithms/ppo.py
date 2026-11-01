"""Proximal Policy Optimization for Box and Dict observations."""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
)

import numpy as np
import torch
import torch.nn.functional as F

from rl_garden.algorithms.decision_mask import (
    DecisionMaskReader,
    warn_if_decision_keys_ignored,
)
from rl_garden.algorithms.observation_encoders import EncoderSharing
from rl_garden.algorithms.on_policy import OnPolicyAlgorithm
from rl_garden.buffers.rollout_buffer import (
    ExtraSpecs,
    RolloutBuffer,
    RolloutBufferSample,
)
from rl_garden.common.ddp import allreduce_grads, allreduce_mean
from rl_garden.common.effective_config import ConfigError
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks.actor_critic import gaussian_kl_divergence
from rl_garden.networks.graph.builtin import (
    builtin_graph,
    builtin_graph_name,
    resolve_builtin_stage,
)
from rl_garden.networks.graph.losses import (
    INFO_TARGET_PREFIX,
    find_graph_loss,
    read_info_targets,
    target_rows,
)
from rl_garden.networks.graph.spec import (
    GraphSpec,
    LossSpec,
    is_builtin_graph_source,
    read_graph_source,
)
from rl_garden.observations import (
    ObservationSchema,
    ObsGroups,
    normalize_observation_space,
)
from rl_garden.policies.graph_ppo_policy import GraphPPOPolicy
from rl_garden.policies.ppo_policy import EvaluationOutputs, PPOPolicy


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
    return masked_mean(torch.max(pg_loss1, pg_loss2), mask, mask.sum())


def masked_mean(x: torch.Tensor, mask: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over the rows where ``mask`` is true; 0 when none is.
    ``count`` is ``mask.sum()`` (passed in so callers reuse it).
    ``torch.where`` (not a multiply) keeps non-finite masked rows out. The
    reduction ``PPO._aux_loss`` applies to a per-row loss over decision rows."""
    return torch.where(mask, x, 0.0).sum() / count.clamp_min(1)


def _normalize_masked_advantages(
    advantages: torch.Tensor, mask: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    """Standardize ``advantages`` with the mean and unbiased std of the true
    rows. Left unchanged when fewer than two rows are true (tensor ``where``,
    no host branch)."""
    mean = masked_mean(advantages, mask, count)
    variance = torch.where(mask, (advantages - mean) ** 2, 0.0).sum() / (count - 1).clamp_min(1)
    normalized = (advantages - mean) / (variance.sqrt() + 1e-8)
    return torch.where(count > 1, normalized, advantages)


def _as_float_list_or_none(values: Optional[Sequence[float]]) -> Optional[list[float]]:
    """Normalize tuple/list init options so checkpoint metadata round-trips."""
    return None if values is None else [float(v) for v in values]


@dataclasses.dataclass(slots=True)
class MinibatchEval:
    """The current policy evaluated on one minibatch, flattened to rows, with
    the stored rollout terms it is compared against (device tensors).

    Every tensor is 1-D over the minibatch rows, except ``log_prob`` and
    ``entropy`` in a window with a component mask, which keep the component
    axis, ``(B, A)``. Sequence minibatches are flattened T-major. The four
    distribution parameters are set only when ``lr_schedule ==
    "adaptive_kl"``. ``outputs`` is the trailing dict the policy's evaluate
    method returns when its ``evaluation_outputs`` is True (``None``
    otherwise), with one row per minibatch row in the same order: the graph
    outputs (node name -> output) of a ``GraphPPOPolicy``, ``{"latent": <the
    features the actor read>}`` of a ``PPOPolicy``/``RecurrentPPOPolicy``
    subclass that opts in. (Before ``_evaluate_minibatch`` flattens them,
    the ``MinibatchEval`` of ``_evaluate_terms`` holds ``values``,
    ``log_prob`` and ``entropy`` as the policy returned them.)
    """

    values: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_mean: torch.Tensor | None = None
    old_log_std: torch.Tensor | None = None
    new_mean: torch.Tensor | None = None
    new_log_std: torch.Tensor | None = None
    outputs: EvaluationOutputs | None = None


@dataclasses.dataclass(slots=True)
class PolicyTerms:
    """The policy side of one minibatch update (``PPO._policy_terms``),
    1-D over the minibatch rows. Under a decision mask the non-decision rows
    of ``advantages`` and ``logratio`` are 0 and a component mask is already
    summed out of ``logratio`` and ``entropy``."""

    #: Normalized when ``norm_adv`` (over the decision rows under a mask).
    advantages: torch.Tensor
    #: ``log_prob - old_log_prob``.
    logratio: torch.Tensor
    #: ``logratio.exp()``.
    ratio: torch.Tensor
    entropy: torch.Tensor
    #: Mean over the rows the policy loss averages: every row, or the
    #: decision rows under a decision mask (0 when there are none).
    row_mean: Callable[[torch.Tensor], torch.Tensor]
    #: ``decision_mask.sum()`` (a device tensor), ``None`` without a mask.
    decisions: torch.Tensor | None = None


@dataclasses.dataclass(slots=True)
class KLDiagnostics:
    """``PPO._kl_diagnostics`` of one minibatch: the two KL estimators
    (detached scalar tensors) and the clip fraction (a host float)."""

    old_approx_kl: torch.Tensor
    approx_kl: torch.Tensor
    clipfrac: float


@dataclasses.dataclass(slots=True)
class UpdateStats:
    """Per-minibatch results of one ``PPO.train()`` call, appended by
    ``_minibatch_step`` and reduced by ``_update_metrics``. ``clipfracs``
    (and ``decision_counts``) also hold the minibatch that stopped the update
    early; the other lists only the minibatches that took a gradient step."""

    policy_losses: list[float] = dataclasses.field(default_factory=list)
    value_losses: list[float] = dataclasses.field(default_factory=list)
    entropy_losses: list[float] = dataclasses.field(default_factory=list)
    approx_kls: list[float] = dataclasses.field(default_factory=list)
    old_approx_kls: list[float] = dataclasses.field(default_factory=list)
    clipfracs: list[float] = dataclasses.field(default_factory=list)
    losses: list[float] = dataclasses.field(default_factory=list)
    #: Decision rows per minibatch (device tensors), under a decision mask.
    decision_counts: list[torch.Tensor] = dataclasses.field(default_factory=list)
    #: ``_aux_loss`` term name -> one detached scalar per minibatch.
    aux_term_values: dict[str, list[torch.Tensor]] = dataclasses.field(default_factory=dict)
    #: Whether ``target_kl`` stopped this update before its last minibatch
    #: (decided on the approx-KL averaged over DDP ranks, so the same on
    #: every rank).
    stopped_early: bool = False


class PPO(OnPolicyAlgorithm):
    """SB3/ManiSkill-style PPO with rl-garden feature extractors.

    Extension points for a subclass: ``policy_class``,
    ``_policy_head_kwargs`` and ``_build_policy`` (which policy is built and
    how), ``rollout_buffer_class`` and ``_rollout_extra_specs`` /
    ``_rollout_extras`` (the rollout buffer and its per-step extras),
    ``_aux_loss`` (an auxiliary loss added to every minibatch's PPO loss),
    the update's terms ``_policy_loss``, ``_value_loss``, ``_entropy_term``
    and ``_total_loss``, ``_extra_update_metrics`` and
    ``_log_update_metrics``; from ``OnPolicyAlgorithm`` the episode cut
    ``_next_done``, ``_after_env_step`` and the bootstrap
    (``_bootstrap_mask``, ``_compute_final_values``); on the policy side
    see ``PPOPolicy``'s ``_build_actor``, ``rollout_extra_specs`` and
    ``evaluation_outputs``. ``docs/guides/extending.md`` lists them."""

    _compatible_checkpoint_algorithms = ("PPO",)
    #: Extension point: the policy ``_build_policy`` builds when no
    #: ``policy_graph`` is given, called with keywords ``observation_space``,
    #: ``action_space``, ``_policy_head_kwargs(observation_space)`` and the
    #: extractors (``actor_extractor``, ``critic_extractor``, ...). A
    #: subclass of ``PPOPolicy``; ``SequencePPO`` needs a
    #: ``RecurrentPPOPolicy`` and also passes ``recurrent_encoder``.
    policy_class: ClassVar[type[PPOPolicy]] = PPOPolicy
    #: Extension point: the rollout buffer ``_build_rollout_buffer``
    #: allocates, called with ``RolloutBuffer``'s constructor keywords.
    #: ``SequencePPO`` needs a ``RecurrentRolloutBuffer`` (BPTT minibatches).
    rollout_buffer_class: ClassVar[type[RolloutBuffer]] = RolloutBuffer
    # decision_level=True reads the env's per-step decision keys and computes
    # SMDP GAE over decision steps (see rl_garden/algorithms/decision_mask.py).
    _supports_decision_level: bool = True
    #: Whether this algorithm's policy carries hidden state; a policy graph
    #: must match (a stateful graph needs a ``SequencePPO`` algorithm).
    stateful_policy: bool = False
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
        policy_graph: Optional[str] = None,
    ) -> None:
        """``decision_level=True`` reads the env's per-step decision keys:
        returns, advantages and the policy loss are computed over decision
        steps. ``per_step_critic`` only matters for a rollout window that
        carries a decision mask (it has no effect when ``decision_level`` is
        off): True trains the critic on every step, False on decision steps
        only. See ``docs/guides/decision-level-ppo.md`` for the env contract
        and the training rules.

        ``target_kl`` stops an update at the first minibatch whose
        approx-KL exceeds it, before that minibatch's gradient step; under
        DDP the approx-KL is averaged over the ranks first, so every rank
        stops at the same minibatch (``_should_stop_early``).

        ``policy_graph`` (YAML text, or ``"builtin:<mlp|recurrent|gru|lstm|gtrxl>"``)
        builds a ``GraphPPOPolicy`` instead; see
        ``rl_garden.networks.graph.spec`` for the format. The graph expresses
        the encoder routing, so ``encoder_sharing``/``obs_groups``/
        ``critic_encoder_config``/``policy_kwargs`` must be unset, and
        ``encoder_config`` is only read by a built-in graph. The graph's
        ``losses:`` are added to the PPO loss on every minibatch and logged
        as ``losses/aux/<node>``; an ``info.<key>`` target is stored per
        step next to the observation (the env must report it on every
        step, checked from the first rollout step on)."""
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
        self.policy_graph = self._resolve_policy_graph(policy_graph)
        self.policy_graph_yaml = (
            self.policy_graph.to_yaml() if self.policy_graph is not None else None
        )
        # Built here rather than in _setup_model(), so a subclass that
        # replaces _setup_model() still has the reader.
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
        extras = self._step_extras(infos)
        if extras:
            out["extras"] = extras
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

    def _rollout_extra_specs(self) -> ExtraSpecs:
        """Extension point: the per-step extras the rollout buffer stores,
        ``{key: (per-env shape, dtype)}``; read once, when
        ``_build_rollout_buffer`` allocates the buffer. Each key arrives in
        ``data.extras[key]`` of every minibatch. Default: the policy's
        ``rollout_extra_specs()``, whose keys are read from ``infos[key]``
        on every env step. A subclass that computes a tensor itself returns
        it from ``_rollout_extras`` and declares it here:
        ``{**super()._rollout_extra_specs(), key: (shape, dtype)}``."""
        return self.policy.rollout_extra_specs()

    def _rollout_extras(self) -> dict[str, torch.Tensor]:
        """Extension point: per-step tensors this algorithm computed itself
        during the current rollout step (e.g. stashed on ``self`` in
        ``_rollout_policy`` or ``_after_env_step``), ``{key: (num_envs,
        *shape)}``, called once per env step before ``rollout_buffer.add``.
        Every key must be declared by ``_rollout_extra_specs``; a declared
        key returned here is not read from ``infos``. Default: ``{}``."""
        return {}

    def _step_extras(self, infos: dict) -> dict[str, torch.Tensor]:
        """The declared extras of this rollout step: ``_rollout_extras()``,
        and every other key of ``rollout_buffer.extra_specs`` read from
        ``infos``."""
        specs = self.rollout_buffer.extra_specs
        rollout_extras = self._rollout_extras()
        undeclared = sorted(set(rollout_extras) - set(specs))
        if undeclared:
            raise ValueError(
                f"_rollout_extras() returned {undeclared}, which _rollout_extra_specs() "
                "does not declare; declare every rollout-time extra there."
            )
        info_specs = {key: spec for key, spec in specs.items() if key not in rollout_extras}
        missing = [key for key in info_specs if key not in infos]
        if missing:
            # The policy's own specs are info targets: read_info_targets
            # reports those with the env-side message.
            policy_specs = self.policy.rollout_extra_specs()
            own = [key for key in missing if key not in policy_specs]
            if own:
                raise ValueError(
                    f"rollout extra {own[0]!r} (declared by _rollout_extra_specs()) is "
                    f"missing from infos at rollout step {self.rollout_buffer.pos} and "
                    "was not returned by _rollout_extras(); return it from "
                    "_rollout_extras() on every step."
                )
        extras = read_info_targets(infos, info_specs, self.num_envs, self.rollout_buffer.pos)
        return {**extras, **rollout_extras}

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
            "policy_graph": self.policy_graph_yaml,
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

    def _resolve_policy_graph(self, source: Optional[str]) -> Optional[GraphSpec]:
        """Parse ``policy_graph`` (expanding a built-in name with this
        algorithm's own settings) and check it against the other encoder
        options and ``stateful_policy``."""
        if source is None:
            return None
        conflicts = [
            name
            for name, value in (
                ("encoder_sharing", self.encoder_sharing),
                ("obs_groups", self.obs_groups),
                ("critic_encoder_config", self.critic_encoder_config),
                ("policy_kwargs", self.policy_kwargs or None),
            )
            if value is not None
        ]
        if conflicts:
            raise ConfigError(
                f"policy_graph expresses the encoder routing (edges decide gradient "
                f"flow); {conflicts} must not be set alongside it."
            )
        source = read_graph_source(source)
        if is_builtin_graph_source(source):
            stage = resolve_builtin_stage(
                builtin_graph_name(source),
                type(self).configured_sequence_stage(vars(self)),
                type(self).__name__,
            )
            spec = builtin_graph(
                stage or "mlp",
                encoder_config=self.encoder_config,
                encoder_sharing=type(self).encoder_sharing,
                stage_params=self._builtin_stage_params() if stage is not None else None,
            )
            if stage is None:
                self._warn_if_builtin_mlp_differs_from_ppo_policy()
        else:
            if self.encoder_config is not None:
                raise ConfigError(
                    "encoder_config is only read by a built-in policy graph; a custom "
                    "graph's encoder nodes carry their own 'config'."
                )
            spec = GraphSpec.parse(source)
        if spec.is_stateful != self.stateful_policy:
            needed = "a stateful" if self.stateful_policy else "a stateless"
            raise ConfigError(
                f"{type(self).__name__} needs {needed} policy graph; this graph's "
                f"stateful nodes are {list(spec.stateful_nodes)} (stateful graphs run "
                "with recurrent_ppo/transformer_ppo, stateless ones with ppo)."
            )
        self.encoder_sharing_origin = "policy_graph"
        return spec

    def _warn_if_builtin_mlp_differs_from_ppo_policy(self) -> None:
        space = normalize_observation_space(self.env.single_observation_space)
        if (
            type(self).encoder_sharing == "shared_critic_grad"
            and ObservationSchema.from_space(space).has_images
        ):
            warnings.warn(
                "policy_graph 'builtin:mlp' on image observations is not PPOPolicy's "
                "objective: its detach edge cuts the whole encoder latent from the actor "
                "loss, while PPOPolicy (encoder_sharing='shared_critic_grad') detaches only "
                "the image branches and still trains the proprio branch with the actor "
                "loss. Write a graph to choose the routing explicitly.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def configured_sequence_stage(cls, settings: Mapping[str, Any]) -> Optional[tuple[str, str]]:
        """``(kind, setting that decides it)`` of the sequence stage this
        algorithm builds from ``settings`` (its own attributes, or the CLI
        args at preflight); ``None``: no stage."""
        del settings
        return None

    def _builtin_stage_params(self) -> dict[str, Any]:
        """Node parameters of this algorithm's own sequence stage in a
        built-in graph (node defaults when empty)."""
        return {}

    def _policy_head_kwargs(self, observation_space) -> dict[str, Any]:
        """Extension point: the head keywords every built policy receives
        (``policy_class`` and ``GraphPPOPolicy`` alike): ``net_arch``, the
        actor init options, the norm/dropout/``kernel_init`` options and the
        backbone types. Read once per ``_build_policy`` call, before any
        module is built. A subclass adds keywords for its own
        ``policy_class`` with ``{**super()._policy_head_kwargs(observation_space),
        ...}``; they also reach ``GraphPPOPolicy`` (and through it
        ``PPOPolicy.__init__``) when a ``policy_graph`` is given."""
        del observation_space  # the defaults do not depend on it
        return {
            "net_arch": self.net_arch,
            "log_std_init": self.log_std_init,
            "log_std_init_per_dim": self.log_std_init_per_dim,
            "action_mean_bias_init": self.action_mean_bias_init,
            "action_mean_head_gain": self.action_mean_head_gain,
            "actor_use_layer_norm": self.actor_use_layer_norm,
            "value_use_layer_norm": self.value_use_layer_norm,
            "actor_use_group_norm": self.actor_use_group_norm,
            "value_use_group_norm": self.value_use_group_norm,
            "num_groups": self.num_groups,
            "actor_dropout_rate": self.actor_dropout_rate,
            "value_dropout_rate": self.value_dropout_rate,
            "kernel_init": self.kernel_init,
            "backbone_type": self.backbone_type,
            "critic_backbone_type": self.critic_backbone_type,
        }

    def _build_graph_policy(self, observation_space) -> GraphPPOPolicy:
        return GraphPPOPolicy(
            observation_space,
            self.env.single_action_space,
            graph=self.policy_graph,
            **self._policy_head_kwargs(observation_space),
        )

    def _build_policy(self, observation_space) -> PPOPolicy:
        """Extension point: build the policy (a ``PPOPolicy``) for
        ``observation_space``; called once from ``_setup_model``, which then
        moves it to ``self.device`` and builds the optimizer over all of its
        parameters. With a ``policy_graph`` it is a ``GraphPPOPolicy``,
        otherwise ``policy_class`` with ``_policy_head_kwargs`` and the
        observation encoders; prefer those two seams to overriding this."""
        if self.policy_graph is not None:
            return self._build_graph_policy(observation_space)
        return self.policy_class(
            observation_space=observation_space,
            action_space=self.env.single_action_space,
            **self._policy_head_kwargs(observation_space),
            **self._policy_extractor_kwargs(observation_space),
        )

    def _setup_model(self) -> None:
        self.policy = self._build_policy(self.env.single_observation_space).to(self.device)
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
        self.rollout_buffer = self._build_rollout_buffer()

    def _build_rollout_buffer(self) -> RolloutBuffer:
        """Allocate this run's ``rollout_buffer_class``; called once from
        ``_setup_model``, after the policy, the optimizer and the LR
        scheduler. Its ``extra_specs`` are ``_rollout_extra_specs()``."""
        return self.rollout_buffer_class(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            store_dist_params=(self.lr_schedule == "adaptive_kl"),
            store_decisions=self.decision_level,
            extra_specs=self._rollout_extra_specs(),
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
        """Extension point: the scalar policy loss of one minibatch from
        its (normalized) ``advantages`` and ``ratio``, 1-D over the rows;
        ``mask`` is the decision mask (reduce over its true rows) or
        ``None``. Default: PPO's clipped surrogate."""
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
        aux_loss: torch.Tensor | None = None,
    ) -> dict[str, float | bool | torch.Tensor]:
        """One PPO minibatch gradient step on 1-D tensors. ``result["stop"]`` is
        True if target_kl triggered an early stop (caller should break after
        recording ``clipfrac``, matching the pre-refactor loop's behavior of
        recording clipfrac for the aborting minibatch but no other metric).
        Under DDP the stop is decided on the approx-KL averaged over the
        ranks (``_should_stop_early``), so every rank stops at the same
        minibatch; the returned ``approx_kl``/``clipfrac`` stay this rank's.

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
        gradient; everything after that reduction is the decision-mask path.

        ``aux_loss`` (a scalar, e.g. the policy graph's losses) is added to
        the PPO loss as is; its gradients join the global grad-norm clip and
        the optimizer's weight decay like every other parameter's.

        The step is composed of ``_policy_terms``, ``_kl_diagnostics``, the
        ``target_kl`` early return (``_should_stop_early``),
        ``_adaptive_kl_step``, ``_policy_loss``,
        ``_value_loss``, ``_entropy_term``, ``_total_loss`` and
        ``_optimizer_step``, in that order; override those, not this."""
        if component_mask is not None and decision_mask is None:
            raise ValueError("component_mask requires decision_mask.")
        terms = self._policy_terms(
            log_prob=log_prob,
            entropy=entropy,
            old_log_prob=old_log_prob,
            advantages=advantages,
            decision_mask=decision_mask,
            component_mask=component_mask,
        )
        decision_stats = {} if terms.decisions is None else {"decisions": terms.decisions}
        kl = self._kl_diagnostics(terms, clip_coef)
        if self._should_stop_early(kl.approx_kl):
            return {"stop": True, "clipfrac": kl.clipfrac, **decision_stats}

        if self.lr_schedule == "adaptive_kl":
            self._adaptive_kl_step(
                terms,
                old_mean=old_mean,
                old_log_std=old_log_std,
                new_mean=new_mean,
                new_log_std=new_log_std,
                component_mask=component_mask,
            )

        policy_loss = self._policy_loss(
            advantages=terms.advantages, ratio=terms.ratio, clip_coef=clip_coef, mask=decision_mask
        )
        value_loss = self._value_loss(
            values=values,
            old_values=old_values,
            returns=returns,
            clip_coef=clip_coef,
            decision_mask=decision_mask,
        )
        entropy_loss = self._entropy_term(terms)
        loss = self._total_loss(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            aux_loss=aux_loss,
        )
        self._optimizer_step(loss)

        return {
            "stop": False,
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "entropy_loss": float(entropy_loss.detach().item()),
            "approx_kl": float(kl.approx_kl.detach().item()),
            "old_approx_kl": float(kl.old_approx_kl.detach().item()),
            "clipfrac": kl.clipfrac,
            "loss": float(loss.detach().item()),
            **decision_stats,
        }

    def _policy_terms(
        self,
        *,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        decision_mask: torch.Tensor | None,
        component_mask: torch.Tensor | None,
    ) -> PolicyTerms:
        """Advantages (normalized per minibatch when ``norm_adv``), the
        log-ratio and ratio, the per-row entropy and the row reduction of
        one minibatch; see ``_ppo_minibatch_update`` for the mask rules."""
        if decision_mask is None:
            if self.norm_adv and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            logratio = log_prob - old_log_prob
            return PolicyTerms(
                advantages=advantages,
                logratio=logratio,
                ratio=logratio.exp(),
                entropy=entropy,
                row_mean=torch.mean,
            )
        decisions = decision_mask.sum()

        def row_mean(x: torch.Tensor) -> torch.Tensor:
            return masked_mean(x, decision_mask, decisions)

        if component_mask is not None:
            # ``where`` before the sum: components that did not take
            # effect contribute exactly zero, gradients included.
            log_prob = torch.where(component_mask, log_prob, 0.0).sum(-1)
            entropy = torch.where(component_mask, entropy, 0.0).sum(-1)
        advantages = torch.where(decision_mask, advantages, 0.0)
        logratio = torch.where(decision_mask, log_prob - old_log_prob, 0.0)
        if self.norm_adv:
            advantages = _normalize_masked_advantages(advantages, decision_mask, decisions)
        return PolicyTerms(
            advantages=advantages,
            logratio=logratio,
            ratio=logratio.exp(),
            entropy=entropy,
            row_mean=row_mean,
            decisions=decisions,
        )

    @staticmethod
    def _kl_diagnostics(terms: PolicyTerms, clip_coef: float) -> KLDiagnostics:
        """The two approximate-KL estimators and the clip fraction over the
        policy-loss rows, without gradients. ``clipfrac`` is read to the host
        here (one sync per minibatch)."""
        with torch.no_grad():
            old_approx_kl = terms.row_mean(-terms.logratio)
            approx_kl = terms.row_mean((terms.ratio - 1.0) - terms.logratio)
            clipfrac = terms.row_mean(((terms.ratio - 1.0).abs() > clip_coef).float()).item()
        return KLDiagnostics(old_approx_kl=old_approx_kl, approx_kl=approx_kl, clipfrac=clipfrac)

    def _should_stop_early(self, approx_kl: torch.Tensor) -> bool:
        """Whether ``target_kl`` stops the update at this minibatch, before
        its gradient step: the minibatch's ``approx_kl`` averaged over the
        DDP ranks (each rank's value weighted equally; the rank's own value
        outside DDP) exceeds ``target_kl``. Every rank reduces the same value
        on every minibatch, so all ranks take the same branch and the
        gradient all-reduce stays matched. ``approx_kl`` itself is not
        modified. Never stops when ``target_kl`` is ``None`` (no collective
        then, on any rank)."""
        if self.target_kl is None:
            return False
        return bool(allreduce_mean(approx_kl.clone()) > self.target_kl)

    def _adaptive_kl_step(
        self,
        terms: PolicyTerms,
        *,
        old_mean: torch.Tensor,
        old_log_std: torch.Tensor,
        new_mean: torch.Tensor,
        new_log_std: torch.Tensor,
        component_mask: torch.Tensor | None,
    ) -> None:
        """``lr_schedule="adaptive_kl"``: set every param group's LR from
        the Gaussian KL between the rollout and the current policy (mean over
        the policy-loss rows, then averaged across DDP ranks), per rsl_rl's
        rule. Runs before the gradient step, on every rank."""
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
            kl_mean = terms.row_mean(kl)
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

    def _value_loss(
        self,
        *,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        clip_coef: float,
        decision_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Extension point: the scalar value loss of one minibatch (weighted
        by ``vf_coef`` in ``_total_loss``). ``values`` carries gradients;
        ``old_values`` and ``returns`` are the rollout's, all 1-D over the
        rows. ``0.5 * mse``, clipped around ``old_values`` by ``clip_coef``
        when ``clip_vloss``.

        With a ``decision_mask`` it covers every row when
        ``per_step_critic`` and the decision rows otherwise, with one formula
        for both cases, so an all-true mask gives the same loss either way."""
        if decision_mask is not None:
            value_mask = torch.ones_like(decision_mask) if self.per_step_critic else decision_mask
            detached_values = values.detach()
            returns = torch.where(value_mask, returns, detached_values)
            v_loss = (values - returns) ** 2
            if self.clip_vloss:
                old_values = torch.where(value_mask, old_values, detached_values)
                values_clipped = old_values + torch.clamp(
                    values - old_values, -clip_coef, clip_coef
                )
                v_loss = torch.max(v_loss, (values_clipped - returns) ** 2)
            return 0.5 * masked_mean(v_loss, value_mask, value_mask.sum())
        if self.clip_vloss:
            v_loss_unclipped = (values - returns) ** 2
            values_clipped = old_values + torch.clamp(values - old_values, -clip_coef, clip_coef)
            v_loss_clipped = (values_clipped - returns) ** 2
            return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        return 0.5 * F.mse_loss(values, returns)

    def _entropy_term(self, terms: PolicyTerms) -> torch.Tensor:
        """Extension point: the scalar entropy bonus of one minibatch
        (subtracted with ``ent_coef`` in ``_total_loss``): the mean entropy
        over the policy-loss rows, ``terms.row_mean(terms.entropy)``."""
        return terms.row_mean(terms.entropy)

    def _total_loss(
        self,
        *,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        aux_loss: torch.Tensor | None,
    ) -> torch.Tensor:
        """Extension point: the scalar loss of the minibatch's single
        ``backward()``: ``policy_loss - ent_coef * entropy_loss + vf_coef *
        value_loss``, plus ``aux_loss`` (``_aux_loss``) when it is not
        ``None``. ``train()`` reports this value as ``loss``."""
        loss = policy_loss - self.ent_coef * entropy_loss + self.vf_coef * value_loss
        if aux_loss is not None:
            loss = loss + aux_loss
        return loss

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        """One gradient step of ``policy_optimizer`` on ``loss``: zero the
        gradients, backward, average them across DDP ranks, clip their
        global norm to ``max_grad_norm``, step."""
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        allreduce_grads(self.policy)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

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

    def _evaluate_minibatch(self, data: RolloutBufferSample) -> MinibatchEval:
        """Evaluate the current policy on ``data`` (``_evaluate_terms``),
        with ``values``, ``log_prob`` and ``entropy`` flattened to rows. In a
        window with a component mask the stored actions that did not take
        effect are zeroed and ``log_prob``/``entropy`` keep their component
        axis."""
        per_component = self.rollout_buffer.window_has_components
        actions = self._component_masked_actions(data) if per_component else data.actions
        evaluate_kwargs = {"sum_dims": False} if per_component else {}
        evaluation = self._evaluate_terms(data, actions, evaluate_kwargs)
        log_prob, entropy = self._policy_term_rows(
            evaluation.log_prob, evaluation.entropy, per_component
        )
        return dataclasses.replace(
            evaluation, values=evaluation.values.flatten(), log_prob=log_prob, entropy=entropy
        )

    def _evaluate_terms(
        self, data: RolloutBufferSample, actions: torch.Tensor, evaluate_kwargs: dict[str, Any]
    ) -> MinibatchEval:
        """The policy's evaluate method on ``data.obs`` and ``actions`` (the
        stored actions, component-masked), called with ``evaluate_kwargs``,
        and the stored rollout terms. ``values``, ``log_prob`` and
        ``entropy`` are left as the evaluate method returns them
        (``_evaluate_minibatch`` flattens them); every other tensor is
        already 1-D over the rows. ``SequencePPO`` overrides this for its
        ``(T, B)`` sequence minibatches."""
        # No explicit stop_gradient_actor: PPOPolicy applies
        # BasePolicy.extract_actor_features's encoder_sharing rule by default.
        if self.lr_schedule == "adaptive_kl":
            (values, log_prob, entropy, new_mean, new_log_std), outputs = (
                self._evaluate_policy(
                    self.policy.evaluate_actions_with_dist_params,
                    data.obs,
                    actions,
                    **evaluate_kwargs,
                )
            )
            old_mean, old_log_std = data.old_mean, data.old_log_std
        else:
            (values, log_prob, entropy), outputs = self._evaluate_policy(
                self.policy.evaluate_actions, data.obs, actions, **evaluate_kwargs
            )
            new_mean = new_log_std = old_mean = old_log_std = None
        return MinibatchEval(
            values=values,
            log_prob=log_prob,
            entropy=entropy,
            old_values=data.old_values,
            old_log_prob=data.old_log_prob,
            advantages=data.advantages,
            returns=data.returns,
            old_mean=old_mean,
            old_log_std=old_log_std,
            new_mean=new_mean,
            new_log_std=new_log_std,
            outputs=outputs,
        )

    def _evaluate_policy(
        self, evaluate, *args, **kwargs
    ) -> tuple[tuple[torch.Tensor, ...], EvaluationOutputs | None]:
        """Call a policy evaluate method; a policy whose
        ``evaluation_outputs`` is True is asked for its outputs dict too
        (``return_outputs=True``), which is split off the returned tuple."""
        if not self.policy.evaluation_outputs:
            return evaluate(*args, **kwargs), None
        *evaluated, outputs = evaluate(*args, return_outputs=True, **kwargs)
        return tuple(evaluated), outputs

    @staticmethod
    def _policy_term_rows(
        log_prob: torch.Tensor, entropy: torch.Tensor, per_component: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten ``log_prob``/``entropy`` to rows (T-major for sequences);
        per component they keep the component axis, ``(rows, A)``
        (``reshape``: ``flatten`` would merge it into the rows)."""
        if not per_component:
            return log_prob.flatten(), entropy.flatten()
        action_dim = log_prob.shape[-1]
        return log_prob.reshape(-1, action_dim), entropy.reshape(-1, action_dim)

    @staticmethod
    def _component_masked_actions(data) -> torch.Tensor:
        """Stored actions with the components that did not take effect set to
        0. Their log-probs are dropped by ``where`` anyway, but an arbitrary
        value there could make that log-prob non-finite, and the zero gradient
        ``where`` sends back would then turn into NaN (``0 * inf``)."""
        return torch.where(data.component_mask, data.actions, 0.0)

    def train(self) -> dict[str, float]:
        """One PPO update on the collected window: ``update_epochs`` passes
        over its minibatches (stopped early by ``target_kl``), the LR step,
        then the metrics (``_update_metrics``, then
        ``_extra_update_metrics``)."""
        clip_coef = self._begin_update()
        stats = UpdateStats()
        self._run_epochs(stats, clip_coef)
        self._step_lr()
        metrics = self._update_metrics(stats, clip_coef)
        metrics.update(self._extra_update_metrics(stats))
        return metrics

    def _begin_update(self) -> float:
        """Put the policy in train mode, count the update and return this
        update's clip coefficient (``_current_clip_coef``)."""
        self.policy.train()
        self._global_update += 1
        self._target_timesteps = max(
            getattr(self, "_target_timesteps", 1), self._global_step
        )
        return self._current_clip_coef()

    def _run_epochs(self, stats: UpdateStats, clip_coef: float) -> None:
        """``update_epochs`` passes over ``_iter_minibatches()`` (which
        reshuffles per pass), one ``_minibatch_step`` per minibatch, until
        one of them stops the update."""
        for _ in range(self.update_epochs):
            for data in self._iter_minibatches():
                if self._minibatch_step(data, clip_coef, stats):
                    return

    def _minibatch_step(self, data, clip_coef: float, stats: UpdateStats) -> bool:
        """Evaluate the policy on one minibatch, add ``_aux_loss`` and take
        the PPO gradient step (``_ppo_minibatch_update``), recording the
        results in ``stats``. Returns True when ``target_kl`` stopped the
        update (no gradient step was taken; only the clip fraction and the
        decision count of this minibatch are recorded)."""
        evaluation = self._evaluate_minibatch(data)
        masks = self._minibatch_masks(data)
        aux_loss, aux_terms = self._aux_loss(data, evaluation, masks.get("decision_mask"))
        result = self._ppo_minibatch_update(
            values=evaluation.values,
            log_prob=evaluation.log_prob,
            entropy=evaluation.entropy,
            old_values=evaluation.old_values,
            old_log_prob=evaluation.old_log_prob,
            advantages=evaluation.advantages,
            returns=evaluation.returns,
            clip_coef=clip_coef,
            old_mean=evaluation.old_mean,
            old_log_std=evaluation.old_log_std,
            new_mean=evaluation.new_mean,
            new_log_std=evaluation.new_log_std,
            aux_loss=aux_loss,
            **masks,
        )
        if "decisions" in result:
            stats.decision_counts.append(result.pop("decisions"))

        stats.clipfracs.append(result["clipfrac"])
        if result["stop"]:
            stats.stopped_early = True
            return True

        stats.policy_losses.append(result["policy_loss"])
        stats.value_losses.append(result["value_loss"])
        stats.entropy_losses.append(result["entropy_loss"])
        stats.approx_kls.append(result["approx_kl"])
        stats.old_approx_kls.append(result["old_approx_kl"])
        stats.losses.append(result["loss"])
        for name, term in aux_terms.items():
            stats.aux_term_values.setdefault(name, []).append(term)
        return False

    def _explained_variance(self) -> float:
        """Explained variance of the window's rollout values for its
        returns, over the rows the critic is trained on; NaN when the
        returns have no variance."""
        b_values = self.rollout_buffer.values.reshape(-1)
        b_returns = self.rollout_buffer.returns.reshape(-1)
        if self.rollout_buffer.window_has_decisions and not self.per_step_critic:
            value_rows = self.rollout_buffer.decisions.reshape(-1)
            b_values, b_returns = b_values[value_rows], b_returns[value_rows]
        y_var = torch.var(b_returns)
        return (
            float("nan")
            if y_var == 0
            else float((1 - torch.var(b_returns - b_values) / y_var).item())
        )

    def _update_metrics(self, stats: UpdateStats, clip_coef: float) -> dict[str, float]:
        """``train()``'s metrics: per-minibatch means of ``stats`` (0 when
        empty), ``explained_variance``, ``clip_coef``, the current
        ``learning_rate``, the decision counts under a decision mask and
        ``aux/<name>`` per ``_aux_loss`` term."""
        explained_var = self._explained_variance()
        metrics = {
            "loss": float(np.mean(stats.losses)) if stats.losses else 0.0,
            "policy_loss": float(np.mean(stats.policy_losses)) if stats.policy_losses else 0.0,
            "value_loss": float(np.mean(stats.value_losses)) if stats.value_losses else 0.0,
            "entropy": float(np.mean(stats.entropy_losses)) if stats.entropy_losses else 0.0,
            "old_approx_kl": (
                float(np.mean(stats.old_approx_kls)) if stats.old_approx_kls else 0.0
            ),
            "approx_kl": float(np.mean(stats.approx_kls)) if stats.approx_kls else 0.0,
            "clipfrac": float(np.mean(stats.clipfracs)) if stats.clipfracs else 0.0,
            "explained_variance": explained_var,
            "clip_coef": clip_coef,
            "learning_rate": float(self.policy_optimizer.param_groups[0]["lr"]),
        }
        if stats.decision_counts:
            counts = torch.stack(stats.decision_counts).float()
            (
                metrics["decisions_per_minibatch_min"],
                metrics["decisions_per_minibatch_mean"],
            ) = torch.stack([counts.min(), counts.mean()]).tolist()
        if stats.aux_term_values:
            means = torch.stack(
                [torch.stack(values).mean() for values in stats.aux_term_values.values()]
            )
            for name, value in zip(stats.aux_term_values, means.tolist()):
                metrics[f"aux/{name}"] = value
        return metrics

    def _extra_update_metrics(self, stats: UpdateStats) -> dict[str, float]:
        """Extension point: extra ``train()`` metrics, merged last into its
        dict (a key here replaces a built-in one) and logged as
        ``losses/<key>``; ``stats`` holds the update's per-minibatch
        results. Called once per update after the LR step; reduce
        per-minibatch tensors here with one host sync. Default: ``{}``."""
        del stats  # no extra metrics by default
        return {}

    def _aux_loss(
        self,
        data,
        evaluation: MinibatchEval,
        decision_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """Extension point: the auxiliary loss of one minibatch, ``(loss,
        terms)``. ``loss`` (a scalar tensor, or ``None`` for no auxiliary
        loss) is added to the PPO loss before its single ``backward()``, so
        it trains every policy parameter it reaches, under the same
        grad-norm clip and optimizer step. ``terms`` maps a name to a
        detached scalar tensor; ``train()`` averages each over the
        minibatches it trained on and returns it as ``aux/<name>`` (logged as
        ``losses/aux/<name>``), with no device sync per minibatch. A
        subclass may combine its own terms with ``super()._aux_loss(...)``.

        A subclass may assume: ``data`` is the minibatch sample (flat rows,
        or ``(T, B, ...)`` for a sequence minibatch; ``data.extras`` holds
        the policy's ``rollout_extra_specs`` entries); ``evaluation`` holds
        the current policy's terms flattened to rows (T-major for
        sequences) and, when the policy's ``evaluation_outputs`` is True,
        ``evaluation.outputs`` with tensors of the same rows;
        ``decision_mask`` is the ``(rows,)`` bool mask of decision rows
        under decision-level PPO and ``None`` otherwise (``masked_mean``
        reduces over it). Under DDP every rank must return a loss reaching
        the same parameters on every minibatch.

        Default: the policy graph's ``losses:`` (none without a
        ``policy_graph``), each named after its node. Graph outputs and
        targets are both laid out as minibatch rows: the ``(T, B, ...)``
        targets of a sequence minibatch are reshaped T-major like the
        outputs. Each loss is reduced like the policy loss: over decision
        rows when its ``mask`` is ``decision`` and the window has a decision
        mask (0, still connected to the graph, when there are none), over
        every row otherwise. Every loss is computed on every minibatch, so
        DDP ranks backpropagate through the same parameters; a weight-0 loss
        is logged but stays out of the backward pass on every rank."""
        graph_losses = self.policy_graph.losses if self.policy_graph is not None else ()
        if evaluation.outputs is None or not graph_losses:
            return None, {}
        aux_loss = None
        terms = {}
        for loss in graph_losses:
            loss_type = find_graph_loss(loss.type)
            pred, target = target_rows(
                loss_type, evaluation.outputs[loss.node], self._graph_loss_target(data, loss)
            )
            row_loss = loss_type.fn(pred, target)
            if loss.mask == "decision" and decision_mask is not None:
                value = masked_mean(row_loss, decision_mask, decision_mask.sum())
            else:
                value = row_loss.mean()
            terms[loss.node] = value.detach()
            if loss.weight > 0:
                weighted = loss.weight * value
                aux_loss = weighted if aux_loss is None else aux_loss + weighted
        return aux_loss, terms

    @staticmethod
    def _graph_loss_target(data, loss: LossSpec) -> torch.Tensor:
        if loss.target == "batch.rewards":
            return data.rewards
        if loss.target == "batch.dones":
            return data.dones
        key = loss.target[len(INFO_TARGET_PREFIX):]
        if key not in data.extras:
            raise ValueError(
                f"graph loss on node {loss.node!r}: target {loss.target!r} was not stored "
                "this window (the rollout passed no infos)."
            )
        return data.extras[key]

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Logs ``train()``'s metrics as ``losses/<key>``; under
        ``decision_level`` also the window's decision and component
        statistics (``train/...``) and the per-dimension ``policy/log_std``.
        Extension point, see ``BaseAlgorithm._log_update_metrics``."""
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
