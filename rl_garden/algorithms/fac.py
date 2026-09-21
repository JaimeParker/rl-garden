"""FAC (Flow Actor-Critic for Offline RL, ICLR 2026, arXiv 2602.18015), ported
from ``3rd_party/FAC/agents/fac.py`` (a structural fork of FQL, not a
subclass -- ``fac.py``'s only diff against ``3rd_party/fql/agents/fql.py`` is
an attribution comment and an unused import) and
``3rd_party/FAC/main.py:128-170``.

FAC keeps FQL's two flow-matching actor networks (``actor_bc_flow``, the
time-conditioned teacher; ``actor_onestep_flow``, the one-step student) and
twin-Q critic, but trains them through a three-phase schedule instead of
FQL's single combined loop. Phases 1-2 below run as a one-time PRELUDE
entirely inside the first ``train()`` call (a synchronous loop, not gated
per gradient step) -- matching the reference's ``main.py:128-163``, where
the BC-flow training loop and the dataset-logp pass both run to completion
BEFORE the actor-critic ``for`` loop even starts, so neither counts against
``offline_steps``. Consequently ``num_offline_steps``/``gradient_steps``
here counts ONLY actor-critic updates, exactly like the reference's
``offline_steps`` (``main.py:166-170``); ``self._global_update`` increments
once per actor-critic update, never once per BC-prelude step:

1. **BC pretrain** (prelude): only ``actor_bc_flow`` trains, via the plain
   flow-matching loss, for ``bc_pretrain_steps`` steps (frozen critic/onestep
   actor), entirely before the first actor-critic update runs.
2. **Transition** (prelude, one-time): with the now-frozen BC flow, compute
   ``log p_beta(a|s)`` for every stored transition via the reverse-ODE
   estimator (``flow_log_prob``, ``rl_garden/networks/flow_logprob.py``) and
   cache it in ``self._logp_table`` (``(per_env_buffer_size, num_envs)``, NaN
   outside ``[0, size)``); ``self._logp_min`` is the dataset-wide minimum.
3. **Actor-critic**: FQL-style critic/actor updates, except the critic loss
   gains a conservative penalty term (this port's core novelty): a
   ``fac_alpha``-weighted, behavior-likelihood-gated push-down of Q at
   one-step-actor "penalty actions" whose estimated log-density under the
   (frozen) BC flow is high relative to a ``fac_threshold``-selected
   log-density baseline. The actor loss drops FQL's own BC-flow term
   entirely (the BC flow is never updated in this phase) and renames FQL's
   distillation coefficient ``alpha`` to ``fac_lambda`` (``_init_fql_params``
   still accepts and stores ``alpha`` for constructor-signature
   compatibility with ``FQLCore``, but ``FACCore`` never reads it).

Three optimizers (matching AGENTS.md's explicit-optimizer-ownership rule, one
per network role): ``critic_optimizer`` (critic + its encoder, unchanged from
``FQLCore``), ``actor_optimizer`` (``actor_onestep_flow`` [+
``actor_extractor`` under ``encoder_sharing="separate"``]), ``bc_optimizer``
(``actor_bc_flow`` [+ ``actor_bc_flow_encoder`` under ``"separate"``]). BC and
actor-critic updates are never in the same backward call, matching the
reference's two independent ``TrainState``s (``bc_network`` vs
``ac_network``) -- this port further splits the reference's single combined
``ac_network`` optimizer (critic + ``actor_onestep`` in one Adam) into two,
matching every other rl-garden algorithm's convention (``FQLCore`` itself
already makes the analogous split for critic vs actor relative to FQL's own
single-optimizer JAX reference).

Deviations from the reference:

- ``bc_pretrain_steps``, if not given explicitly, is derived from
  ``bc_pretrain_epochs * ceil(epoch_size / bc_batch_size)`` lazily at the
  first ``train()`` call (buffer must be filled by then) and cached; the
  reference's on-disk ``pretrained_bc/`` cache is not ported -- use
  rl-garden checkpoints (``save``/``load``) to resume across runs instead.
  ``_phase_step`` (the BC-progress counter, saved/restored via
  ``_training_state_dict``) lets a resume mid-prelude replay only the
  remaining BC steps the next time ``train()`` runs its prelude -- but a
  checkpoint can only ever be written BETWEEN ``train()`` calls (the offline
  runner, ``run_offline_pretraining`` in ``rl_garden/algorithms/offline.py``,
  never checkpoints mid-call), so in practice a checkpoint only ever lands
  after the whole prelude (BC pretrain + logp cache) has finished, never
  mid-prelude.
- The per-sample threshold modes (``batch_adaptive``, ``batch_wide_constant``)
  need sample indices to gather from ``self._logp_table``, so they require
  ``offline_sampling="with_replace"`` (``FACCore`` draws its own indices, the
  same distribution as ``ReplayBuffer.sample``,
  ``rl_garden/buffers/_sampling.py:91-95``). ``dataset_wide_constant`` needs
  no per-sample indices and works under either sampling mode.
  ``_init_fac_params`` raises ``ValueError`` for the unsupported combination.
- Eval during the BC-pretrain phase reports an untrained one-step actor (the
  same caveat ``BPPO``'s own Phase A carries).
- No ``use_cp`` toggle: the reference's ``critic_loss(..., use_cp=True)``
  flag (skip the penalty entirely) has no constructor knob here -- the
  penalty pipeline always runs; setting ``fac_alpha=0.0`` reduces
  ``critic_loss`` to the plain TD loss numerically (multiplying a finite
  penalty term by zero), which this port's tests rely on instead.
- Online fine-tuning (``main.py``'s ``online_finetuning=True`` branch,
  ``fac.py:54-65``) is wired up via ``Off2OnFAC`` at the bottom of this
  module: after ``switch_to_online_mode``, ``FACCore.train()`` skips the
  BC-prelude/dataset-logp machinery entirely and, per gradient step, runs
  ``_bc_update`` on the same sampled batch before the critic/actor updates,
  with ``logp_beta`` recomputed on the fly by ``_logp_beta_online`` instead
  of gathered from ``_logp_table`` (online transitions never populate that
  cache). The reference's ``balanced_sampling`` (half offline dataset, half
  replay buffer, ``main.py:212-216``) maps to this mixin's existing
  ``online_replay_mode="mixed", offline_data_ratio=0.5`` combination rather
  than a new mechanism; neither ``Off2OnFAC`` nor ``Off2OnFINO``/``Off2OnFloQ``
  override ``Off2OnCommonArgs``'s default (``online_replay_mode="empty"``,
  ``offline_data_ratio=0.0``), so balanced sampling is opt-in via CLI flags,
  not the default. ``_bc_update`` keeps stepping ``self._lr_schedulers[2]``
  (the BC optimizer's scheduler) on every online gradient step, same as
  offline -- a no-op under the default ``lr_schedule="constant"``
  (``make_lr_scheduler`` returns ``None``), but under a non-default schedule
  sized for ``bc_pretrain_steps`` (e.g. ``warmup_cosine`` with
  ``lr_decay_steps`` set to the prelude length) the schedule keeps advancing
  indefinitely online and floors at ``min_lr_ratio * bc_lr`` once ``step``
  exceeds ``lr_decay_steps``, rather than resetting or freezing at the
  prelude's final value. The reference has no BC learning-rate schedule at
  all (plain ``optax.adam(learning_rate=config['lr_bc'])``); this only
  matters for callers who explicitly opt into a non-constant ``lr_schedule``.
- Tensor-shape adaptation: the reference's critic output is squeezed to
  ``(n_critics, batch)`` before any penalty-weight broadcast; rl-garden's
  ``EnsembleQCritic.forward_all`` keeps a trailing size-1 action-value dim
  (``(n_critics, batch, 1)``, the same convention ``FQLCore._critic_loss``
  already relies on), so the per-sample weight is reshaped to
  ``(1, batch, 1)`` before multiplying ``q_pen`` -- purely a broadcast-shape
  adaptation, not a formula change.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F

from rl_garden.algorithms.fql import FQLCore
from rl_garden.algorithms.off2on import Off2OnReplayMixin
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.common.logger import Logger
from rl_garden.common.optim import make_lr_scheduler, make_optimizer
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.factory import build_observation_encoder
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.networks.actor_vector_field import flow_onestep_distill_loss
from rl_garden.networks.flow_logprob import LogpMethod, flow_log_prob
from rl_garden.observations import ObsGroups, resolve_obs_groups
from rl_garden.policies.fql_policy import EncoderSharing, FQLPolicy

FacThreshold = Literal["batch_adaptive", "batch_wide_constant", "dataset_wide_constant"]
WeightType = Literal["linear", "logarithmic", "convex", "concave"]

_WEIGHT_TEMPERATURES: dict[str, float] = {
    "logarithmic": 1.0,
    "convex": 0.5,
    "concave": 2.0,
}


class FACCore(FQLCore):
    """FAC's BC-pretrain -> logp-cache -> conservative-actor-critic schedule
    on top of ``FQLCore``. See module docstring."""

    # The reference builds three independent encoders (critic, bc_flow,
    # actor_onestep -- fac.py:426-431, facts doc section A); rl-garden's
    # "separate" sharing mode is exactly that layout, so it is the default
    # here (overriding FQLCore's own inherited "shared_critic_grad", which
    # comes from ObservationEncoderMixin's base default,
    # rl_garden/algorithms/_observation.py:147). This is not just fidelity:
    # under a shared mode the critic keeps training the shared encoder
    # throughout Phase B while the (frozen) bc_flow's cached `_logp_table`
    # was built from features taken at the transition step -- the table
    # goes stale as the shared encoder's features drift under further
    # critic training. An explicit encoder_sharing="shared_critic_grad" is
    # still accepted (encoder_sharing_choices is unchanged from FQLCore),
    # but the staleness caveat above applies to it.
    encoder_sharing: EncoderSharing = "separate"

    def _init_fac_params(
        self,
        *,
        fac_alpha: float = 1.0,
        fac_lambda: float = 1.0,
        fac_threshold: FacThreshold = "batch_adaptive",
        logp_method: LogpMethod = "exact",
        logp_hutch_probes: int = 8,
        weight_type: WeightType = "linear",
        bc_lr: float = 3e-4,
        bc_batch_size: Optional[int] = None,
        bc_pretrain_epochs: int = 250,
        bc_pretrain_steps: Optional[int] = None,
    ) -> None:
        if fac_threshold not in ("batch_adaptive", "batch_wide_constant", "dataset_wide_constant"):
            raise ValueError(
                "fac_threshold must be 'batch_adaptive', 'batch_wide_constant', or "
                f"'dataset_wide_constant', got {fac_threshold!r}."
            )
        if logp_method not in ("exact", "hutch-rade", "hutch-gaus"):
            raise ValueError(
                f"logp_method must be 'exact', 'hutch-rade', or 'hutch-gaus', got {logp_method!r}."
            )
        if weight_type not in ("linear", "logarithmic", "convex", "concave"):
            raise ValueError(
                "weight_type must be 'linear', 'logarithmic', 'convex', or 'concave', got "
                f"{weight_type!r}."
            )
        if fac_alpha < 0:
            raise ValueError(f"fac_alpha must be >= 0, got {fac_alpha}.")
        if fac_lambda < 0:
            raise ValueError(f"fac_lambda must be >= 0, got {fac_lambda}.")
        if logp_hutch_probes < 1:
            raise ValueError(f"logp_hutch_probes must be >= 1, got {logp_hutch_probes}.")
        if bc_lr <= 0:
            raise ValueError(f"bc_lr must be positive, got {bc_lr}.")
        if bc_batch_size is not None and bc_batch_size <= 0:
            raise ValueError(f"bc_batch_size must be positive or None, got {bc_batch_size}.")
        if bc_pretrain_epochs <= 0:
            raise ValueError(f"bc_pretrain_epochs must be positive, got {bc_pretrain_epochs}.")
        if bc_pretrain_steps is not None and bc_pretrain_steps < 0:
            raise ValueError(f"bc_pretrain_steps must be >= 0 or None, got {bc_pretrain_steps}.")
        # Per-sample threshold modes gather cached logp by (batch_inds,
        # env_inds), so they need real sample indices -- only "with_replace"
        # sampling produces those (see module docstring's Deviations list).
        if self.offline_sampling != "with_replace" and fac_threshold != "dataset_wide_constant":
            raise ValueError(
                f"fac_threshold={fac_threshold!r} requires per-sample cached logp, which needs "
                "offline_sampling='with_replace' (index-based batch fetch); use "
                "fac_threshold='dataset_wide_constant' under offline_sampling='without_replace'."
            )

        self.fac_alpha = fac_alpha
        self.fac_lambda = fac_lambda
        self.fac_threshold = fac_threshold
        self.logp_method = logp_method
        self.logp_hutch_probes = logp_hutch_probes
        self.weight_type = weight_type
        self.bc_lr = bc_lr
        self.bc_batch_size = bc_batch_size
        self.bc_pretrain_epochs = bc_pretrain_epochs
        self.bc_pretrain_steps = bc_pretrain_steps

        self._phase_step = 0
        self._logp_table: Optional[torch.Tensor] = None
        self._logp_min: Optional[float] = None
        self._bc_pretrain_steps_resolved: Optional[int] = None
        self._bc_batch_size_resolved: Optional[int] = None
        # Flipped True by Off2OnFAC's rollout shell (only class that mixes
        # in Off2OnReplayMixin) via _apply_online_regularizer_override, at
        # the offline->online switch. Always False here so plain offline
        # FAC's train() is unaffected. See module docstring's Deviations.
        self._online_finetuning = False

    # --- phase-gate helpers ---

    @property
    def phase(self) -> str:
        """``"bc_pretrain"`` before the actor-critic phase starts (including
        before ``bc_pretrain_steps`` has ever been resolved), else
        ``"actor_critic"``."""
        if self._bc_pretrain_steps_resolved is None:
            return "bc_pretrain"
        if self._phase_step < self._bc_pretrain_steps_resolved:
            return "bc_pretrain"
        return "actor_critic"

    def _resolve_bc_batch_size(self) -> int:
        if self._bc_batch_size_resolved is not None:
            return self._bc_batch_size_resolved
        if self.bc_batch_size is not None:
            resolved = int(self.bc_batch_size)
        else:
            # main.py:130-134's dataset-size heuristic.
            epoch_size = self.replay_buffer.size * self.replay_buffer.num_envs
            if epoch_size < 100_000:
                multiplier = 1
            elif epoch_size < 500_000:
                multiplier = 4
            else:
                multiplier = 16
            resolved = self.batch_size * multiplier
        self._bc_batch_size_resolved = resolved
        return resolved

    def _resolve_bc_pretrain_steps(self) -> int:
        if self._bc_pretrain_steps_resolved is not None:
            return self._bc_pretrain_steps_resolved
        if self.bc_pretrain_steps is not None:
            resolved = int(self.bc_pretrain_steps)
        else:
            bc_batch_size = self._resolve_bc_batch_size()
            epoch_size = self.replay_buffer.size * self.replay_buffer.num_envs
            steps_per_epoch = math.ceil(epoch_size / bc_batch_size) if bc_batch_size > 0 else 0
            resolved = self.bc_pretrain_epochs * steps_per_epoch
        self._bc_pretrain_steps_resolved = resolved
        return resolved

    # --- parameter groups (disjoint across the three optimizers) ---

    def _actor_only_parameters(self):
        yield from self.policy.actor_onestep_flow.parameters()
        if self.encoder_sharing == "separate":
            yield from self.policy.actor_extractor.parameters()

    def _bc_only_parameters(self):
        yield from self.policy.actor_bc_flow.parameters()
        if self.encoder_sharing == "separate":
            yield from self.policy.actor_bc_flow_encoder.parameters()

    def _bc_features(self, obs) -> torch.Tensor:
        """Features ``actor_bc_flow`` consumes -- reproduces
        ``FQLPolicy.extract_actor_loss_features``'s bc branch (see
        ``rl_garden/policies/fql_policy.py:198-232``) as a standalone call,
        since FAC calls it in phases (BC pretrain, dataset-logp caching,
        critic-penalty logp) that never have a pre-computed
        ``critic_features`` tensor from the same step available to reuse."""
        if self.encoder_sharing == "separate":
            return self.policy.actor_bc_flow_encoder.extract(obs, stop_gradient=False)
        return self.policy.extract_critic_features(obs).detach()

    def _penalty_weight(self, diff: torch.Tensor) -> torch.Tensor:
        """``w(diff)``, clipped to ``[0, 1]`` (``fac.py:78-90``)."""
        if self.weight_type == "linear":
            return (1.0 - diff.exp()).clamp(0.0, 1.0)
        temp = _WEIGHT_TEMPERATURES[self.weight_type]
        # exp(diff)**temp == exp(diff*temp); computed this way to avoid a
        # separate (numerically riskier) power-of-an-exponential step.
        return (1.0 - torch.log1p(torch.exp(diff * temp)) / math.log(2.0)).clamp(0.0, 1.0)

    # --- model setup ---

    def _setup_model(self) -> None:
        observation_space = self.env.single_observation_space
        extractor_kwargs = self._policy_extractor_kwargs(observation_space)
        if self.encoder_sharing == "separate":
            actor_keys = resolve_obs_groups(
                self.observation_encoders.schema, self.obs_groups
            )["actor"].keys
            actor_bc_flow_encoder = build_observation_encoder(
                observation_space, self.encoder_config, keys=actor_keys
            )
        else:
            actor_bc_flow_encoder = None
        self.policy = FQLPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            actor_use_group_norm=self.actor_use_group_norm,
            critic_use_group_norm=self.critic_use_group_norm,
            num_groups=self.num_groups,
            critic_dropout_rate=self.critic_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            activation_fn=self.activation_fn,
            actor_bc_flow_encoder=actor_bc_flow_encoder,
            **extractor_kwargs,
        ).to(self.device)

        self.critic_optimizer = make_optimizer(
            list(self.policy.critic_and_encoder_parameters()),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.actor_optimizer = make_optimizer(
            list(self._actor_only_parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.bc_optimizer = make_optimizer(
            list(self._bc_only_parameters()),
            lr=self.bc_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.replay_buffer = self._build_replay_buffer()
        self._lr_schedulers = [
            make_lr_scheduler(
                opt,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
            for opt in (self.critic_optimizer, self.actor_optimizer, self.bc_optimizer)
        ]

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("critic_optimizer", "actor_optimizer", "bc_optimizer")

    # --- Phase A: BC-flow pretrain ---

    def _bc_update(self, data) -> dict[str, float]:
        bc_features = self._bc_features(data.obs)
        batch_size = data.actions.shape[0]
        action_dim = data.actions.shape[-1]
        device, dtype = bc_features.device, bc_features.dtype

        x_0 = torch.randn(batch_size, action_dim, device=device, dtype=dtype)
        t = torch.rand(batch_size, 1, device=device, dtype=dtype)
        x_t = (1 - t) * x_0 + t * data.actions
        vel_target = data.actions - x_0
        pred_vel = self.policy.actor_bc_flow(bc_features, x_t, t)
        bc_flow_loss = F.mse_loss(pred_vel, vel_target)

        self.bc_optimizer.zero_grad(set_to_none=True)
        bc_flow_loss.backward()
        self._clip_grad_norm(self._bc_only_parameters())
        self.bc_optimizer.step()
        if self._lr_schedulers[2] is not None:
            self._lr_schedulers[2].step()

        return {"bc_flow_loss": float(bc_flow_loss.detach().item())}

    # --- transition: cache dataset-wide log-density under the frozen BC flow ---

    def _compute_dataset_logp(self) -> None:
        buffer = self.replay_buffer
        size = buffer.size
        num_envs = buffer.num_envs
        bc_batch_size = self._resolve_bc_batch_size()

        self._logp_table = torch.full(
            (buffer.per_env_buffer_size, num_envs),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )

        was_training = self.policy.training
        self.policy.eval()
        total = size * num_envs
        if total > 0:
            flat = torch.arange(total)
            with torch.no_grad():
                for start in range(0, total, bc_batch_size):
                    chunk = flat[start : start + bc_batch_size]
                    batch_inds = torch.div(chunk, num_envs, rounding_mode="floor")
                    env_inds = chunk % num_envs
                    sample = buffer._index_batch(batch_inds, env_inds)
                    bc_features = self._bc_features(sample.obs)
                    logp = flow_log_prob(
                        self.policy.actor_bc_flow,
                        bc_features,
                        sample.actions,
                        num_steps=self.flow_steps,
                        method=self.logp_method,
                        num_probes=self.logp_hutch_probes,
                    )
                    self._logp_table[batch_inds, env_inds] = logp.to(self._logp_table.dtype)
        if was_training:
            self.policy.train()

        finite = self._logp_table[torch.isfinite(self._logp_table)]
        if finite.numel() > 0:
            mean_logp = float(finite.mean().item())
            self._logp_min = float(finite.min().item())
        else:
            mean_logp = float("nan")
            self._logp_min = float("nan")
        if self.logger is not None:
            self.logger.add_scalar("logp/dataset_mean", mean_logp, self._global_step)
            self.logger.add_scalar("logp/dataset_min", self._logp_min, self._global_step)

    # --- Phase B: actor-critic ---

    def _sample_train_batch_with_indices(self, batch_size: int):
        """Returns ``(sample, batch_inds, env_inds)``; ``batch_inds``/
        ``env_inds`` are ``None`` for the one combination that needs no
        per-sample gather (``dataset_wide_constant`` + ``without_replace``,
        see module docstring's Deviations list)."""
        if self.fac_threshold == "dataset_wide_constant" and self.offline_sampling == "without_replace":
            return self._sample_train_batch(batch_size), None, None
        upper = self.replay_buffer.size
        batch_inds = torch.randint(0, upper, size=(batch_size,))
        env_inds = torch.randint(0, self.replay_buffer.num_envs, size=(batch_size,))
        return self.replay_buffer._index_batch(batch_inds, env_inds), batch_inds, env_inds

    def _logp_beta(self, batch_inds, env_inds, batch_size: int) -> torch.Tensor:
        if self.fac_threshold == "dataset_wide_constant":
            return torch.full(
                (batch_size,), self._logp_min, device=self.device, dtype=torch.float32
            )
        per_sample = self._logp_table[batch_inds, env_inds].to(self.device)
        if self.fac_threshold == "batch_adaptive":
            return per_sample
        # batch_wide_constant: min WITHIN this minibatch, broadcast.
        return per_sample.min().expand(batch_size)

    def _logp_beta_online(self, data) -> torch.Tensor:
        """Online-finetuning counterpart of ``_logp_beta`` (``fac.py:54-65``'s
        ``online_finetuning=True`` branch): recomputes ``estimated_logp_beta``
        on the fly from THIS batch's own actions under the BC flow (already
        ``_bc_update``-d this step, matching ``main.py:220-223``'s
        ``train_bc`` -> ``train_ac`` order) instead of gathering
        ``_logp_table`` -- online transitions never populate that cache.
        ``batch_wide_constant`` AND ``dataset_wide_constant`` both collapse to
        the in-batch minimum online (``fac.py:61-65``: the dataset-wide cache
        `dataset_wide_constant` used offline no longer applies); only
        ``batch_adaptive`` keeps per-sample values. No explicit
        ``.detach()``/stop-gradient call is needed -- the whole computation
        runs under ``torch.no_grad()``, matching the reference's
        ``jax.lax.stop_gradient``."""
        batch_size = data.actions.shape[0]
        with torch.no_grad():
            bc_features = self._bc_features(data.obs)
            logp_beta = flow_log_prob(
                self.policy.actor_bc_flow,
                bc_features,
                data.actions,
                num_steps=self.flow_steps,
                method=self.logp_method,
                num_probes=self.logp_hutch_probes,
            ).to(device=self.device, dtype=torch.float32)
            if self.fac_threshold != "batch_adaptive":
                logp_beta = logp_beta.min().expand(batch_size)
        return logp_beta

    def _critic_update(
        self, data, obs_features: torch.Tensor, logp_beta: torch.Tensor
    ) -> dict[str, float]:
        with torch.no_grad():
            next_features_critic = self.policy.extract_critic_features(data.next_obs)
            if self.encoder_sharing == "separate":
                next_features_actor = self.policy.extract_actor_onestep_features(
                    data.next_obs
                )
            else:
                next_features_actor = next_features_critic
            next_noise = self.policy.sample_noise(
                next_features_actor.shape[0],
                device=next_features_actor.device,
                dtype=next_features_actor.dtype,
            )
            next_action = self.policy.actor_onestep_flow(next_features_actor, next_noise)
            next_action = next_action.clamp(
                self.policy.action_low, self.policy.action_high
            )
            target_q_all = self.policy.q_values_all(
                next_features_critic, next_action, target=True
            )
            next_q = self._aggregate_target_q(target_q_all)
            target_q = data.rewards.unsqueeze(-1) + self.gamma * (
                1.0 - data.dones.unsqueeze(-1)
            ) * next_q

            # Conservative-penalty ingredients (fac.py:67-93): penalty action
            # from the frozen one-step actor, its estimated log-density under
            # the frozen BC flow, and the resulting weight -- all no_grad,
            # matching the reference's stop_gradient on
            # estimated_logp_pi/estimated_logp_beta and on the weight itself
            # (fac.py:56,71,93).
            if self.encoder_sharing == "separate":
                pen_features_actor = self.policy.extract_actor_onestep_features(data.obs)
            else:
                pen_features_actor = obs_features.detach()
            pen_noise = self.policy.sample_noise(
                pen_features_actor.shape[0],
                device=pen_features_actor.device,
                dtype=pen_features_actor.dtype,
            )
            a_pen = self.policy.actor_onestep_flow(pen_features_actor, pen_noise)
            a_pen = a_pen.clamp(self.policy.action_low, self.policy.action_high)
            bc_features = self._bc_features(data.obs)
            logp_pi = flow_log_prob(
                self.policy.actor_bc_flow,
                bc_features,
                a_pen,
                num_steps=self.flow_steps,
                method=self.logp_method,
                num_probes=self.logp_hutch_probes,
            )
            diff = logp_pi - logp_beta
            weight = self._penalty_weight(diff).detach()

        q_all = self.policy.q_values_all(obs_features, data.actions, target=False)
        td_loss = self._critic_loss(q_all, target_q)

        # q_pen: (n_critics, batch, 1) -- see module docstring's shape-
        # adaptation note. DOES receive gradient (unlike q_all's TD target
        # side, the critic forward at a_pen is live).
        q_pen = self.policy.q_values_all(obs_features, a_pen, target=False)
        penalty = self.fac_alpha * (weight.view(1, -1, 1) * q_pen).mean()

        critic_loss = td_loss + penalty

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
        self.critic_optimizer.step()
        if self._lr_schedulers[0] is not None:
            self._lr_schedulers[0].step()

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "td_loss": float(td_loss.detach().item()),
            "critic_penalty": float(penalty.detach().item()),
            "est_logpi_mean": float(logp_pi.mean().item()),
            "est_logpi_max": float(logp_pi.max().item()),
            "est_logpi_min": float(logp_pi.min().item()),
            "est_logbeta_mean": float(logp_beta.mean().item()),
            "est_logbeta_max": float(logp_beta.max().item()),
            "est_logbeta_min": float(logp_beta.min().item()),
            "penalty_weight_mean": float(weight.mean().item()),
        }

    def _actor_update(self, data, obs_features: torch.Tensor) -> dict[str, float]:
        # Same shape as FQLCore._actor_update, minus the bc_flow_loss term
        # (the BC flow is frozen in this phase) and alpha -> fac_lambda.
        bc_features, onestep_features, q_features = self.policy.extract_actor_loss_features(
            data.obs, critic_features=obs_features
        )
        batch_size = data.actions.shape[0]
        action_dim = data.actions.shape[-1]
        device, dtype = onestep_features.device, onestep_features.dtype

        noises = torch.randn(batch_size, action_dim, device=device, dtype=dtype)
        actor_actions = self.policy.actor_onestep_flow(onestep_features, noises)
        distill_loss = flow_onestep_distill_loss(
            self.policy.actor_bc_flow,
            actor_actions,
            bc_features,
            noises,
            self.flow_steps,
            low=self.policy.action_low,
            high=self.policy.action_high,
        )

        clipped_actions = actor_actions.clamp(
            self.policy.action_low, self.policy.action_high
        )
        q_all_pi = self.policy.q_values_all(q_features, clipped_actions, target=False)
        q_pi = q_all_pi.mean(dim=0)  # actor loss always averages the ensemble
        q_loss = -q_pi.mean()
        if self.normalize_q_loss:
            lam = (1.0 / q_pi.abs().mean()).detach()
            q_loss = lam * q_loss

        actor_loss = self.fac_lambda * distill_loss + q_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip_grad_norm(self._actor_only_parameters())
        self.actor_optimizer.step()
        if self._lr_schedulers[1] is not None:
            self._lr_schedulers[1].step()

        return {
            "actor_loss": float(actor_loss.detach().item()),
            "distill_loss": float(distill_loss.detach().item()),
            "q_loss": float(q_loss.detach().item()),
        }

    # --- training loop ---

    def _bc_prelude_log_freq(self) -> int:
        log_freq = getattr(self, "log_freq", None)
        if log_freq is None or log_freq <= 0:
            return 1000
        return int(log_freq)

    def _run_bc_prelude(self, bc_pretrain_steps: int, bc_batch_size: int) -> None:
        """Runs the ENTIRE remaining BC-flow pretrain to completion, matching
        the reference's ``main.py:139-144`` loop, which runs before the
        actor-critic loop ever starts (see module docstring). Only called
        once, from the start of ``train()``, while ``self._phase_step <
        bc_pretrain_steps``."""
        remaining = bc_pretrain_steps - self._phase_step
        if self.std_log:
            print(
                f"[fac] bc_prelude start: phase_step={self._phase_step}/"
                f"{bc_pretrain_steps} ({remaining} steps remaining)",
                flush=True,
            )
        log_freq = self._bc_prelude_log_freq()
        while self._phase_step < bc_pretrain_steps:
            data = self._sample_train_batch(bc_batch_size)
            metrics = self._bc_update(data)
            self._phase_step += 1
            if self.logger is not None and self._phase_step % log_freq == 0:
                self.logger.add_scalar(
                    "train/bc_flow_loss", metrics["bc_flow_loss"], self._phase_step
                )
        if self.std_log:
            print(f"[fac] bc_prelude done: phase_step={self._phase_step}", flush=True)

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        """FAC update loop matching the reference's phase order
        (``main.py:128-170``): on entry, if the BC-flow prelude (pretrain +
        dataset-logp cache) has not finished yet, it runs to completion
        first -- unconditionally, regardless of ``gradient_steps`` -- then
        exactly ``gradient_steps`` actor-critic updates run. ``gradient_steps``
        therefore always counts actor-critic updates only, and
        ``self._global_update`` increments only for those.

        Once ``self._online_finetuning`` is True (``Off2OnFAC``, after
        ``switch_to_online_mode``), the BC-prelude/dataset-logp machinery
        above is skipped entirely -- online transitions never populate
        ``_logp_table`` -- and each gradient step instead reproduces the
        reference's ``main.py:220-223`` online branch: sample once (the
        mixin's offline/online mixed batch), run ``_bc_update`` on that same
        batch, then the critic/actor updates with ``logp_beta`` recomputed by
        ``_logp_beta_online`` instead of gathered from the cache. See module
        docstring."""
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        self.policy.train()

        if not self._online_finetuning:
            bc_pretrain_steps = self._resolve_bc_pretrain_steps()
            bc_batch_size = self._resolve_bc_batch_size()

            if self._phase_step < bc_pretrain_steps:
                self._run_bc_prelude(bc_pretrain_steps, bc_batch_size)

            if self._logp_table is None:
                self._compute_dataset_logp()  # restores policy.train() itself

        metrics_sum: dict[str, float] = {}
        counts: dict[str, int] = {}
        for _ in range(gradient_steps):
            self._global_update += 1
            if self._online_finetuning:
                data = self._sample_train_batch(self.batch_size)
                bc_metrics = self._bc_update(data)
                obs_features = self.policy.extract_critic_features(data.obs)
                logp_beta = self._logp_beta_online(data)
            else:
                bc_metrics = {}
                data, batch_inds, env_inds = self._sample_train_batch_with_indices(
                    self.batch_size
                )
                obs_features = self.policy.extract_critic_features(data.obs)
                logp_beta = self._logp_beta(batch_inds, env_inds, self.batch_size)
            critic_metrics = self._critic_update(data, obs_features, logp_beta)
            actor_metrics = self._actor_update(data, obs_features)
            self._update_targets()
            metrics = {**bc_metrics, **critic_metrics, **actor_metrics}

            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

        if not compute_info:
            return {}
        return {key: metrics_sum[key] / counts[key] for key in metrics_sum}

    # --- checkpointing ---

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "fac_alpha": self.fac_alpha,
            "fac_lambda": self.fac_lambda,
            "fac_threshold": self.fac_threshold,
            "logp_method": self.logp_method,
            "logp_hutch_probes": self.logp_hutch_probes,
            "weight_type": self.weight_type,
            "bc_lr": self.bc_lr,
            "bc_batch_size": self.bc_batch_size,
            "bc_pretrain_epochs": self.bc_pretrain_epochs,
            "bc_pretrain_steps": self._bc_pretrain_steps_resolved,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            **super()._extra_checkpoint_state(),
            "logp_table": self._logp_table.cpu() if self._logp_table is not None else None,
            "logp_min": self._logp_min,
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        logp_table = state.get("logp_table")
        self._logp_table = logp_table.to(self.device) if logp_table is not None else None
        self._logp_min = state.get("logp_min")

    def _training_state_dict(self) -> dict[str, Any]:
        # Chains to super() (bug fix, this port's first override of the
        # hook): previously returned a fresh dict with no super() call,
        # which silently dropped OffPolicyAlgorithm's
        # `initial_phase_start_step` from every Off2OnFAC checkpoint (plain
        # offline FAC was unaffected -- BaseAlgorithm's own default is {}).
        return {
            **super()._training_state_dict(),
            "phase_step": self._phase_step,
            "bc_pretrain_steps_resolved": self._bc_pretrain_steps_resolved,
            # Off2OnReplayMixin's own `_online_start_step` persistence
            # (_extra_checkpoint_state) does not cover this flag: it is set
            # by `_apply_online_regularizer_override`, which only fires from
            # `switch_to_online_mode` -- a resumed already-online checkpoint
            # would never re-run it, so the flag needs its own round trip.
            "online_finetuning": self._online_finetuning,
        }

    def _load_training_state_dict(self, state: dict[str, Any]) -> None:
        super()._load_training_state_dict(state)
        self._phase_step = int(state.get("phase_step", 0))
        resolved = state.get("bc_pretrain_steps_resolved")
        self._bc_pretrain_steps_resolved = int(resolved) if resolved is not None else None
        self._online_finetuning = bool(state.get("online_finetuning", False))


class FAC(FACCore, OfflineRLAlgorithm):
    """Offline FAC: twin-Q critic with a conservative penalty + two-network
    flow-matching actor, trained through a BC-pretrain -> logp-cache ->
    actor-critic schedule. See module docstring."""

    _compatible_checkpoint_algorithms = ("FAC",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        gamma: float = 0.99,
        offline_sampling: str = "with_replace",
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        alpha: float = 10.0,  # unused by FAC; kept only for FQLCore-signature compatibility
        flow_steps: int = 10,
        q_agg: Literal["mean", "min"] = "mean",
        normalize_q_loss: bool = False,
        net_arch: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = "xavier_uniform",
        backbone_type: BackboneType = "mlp",
        activation_fn: Optional[Activation] = "gelu",
        encoder_sharing: Optional[EncoderSharing] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        fac_alpha: float = 1.0,
        fac_lambda: float = 1.0,
        fac_threshold: FacThreshold = "batch_adaptive",
        logp_method: LogpMethod = "exact",
        logp_hutch_probes: int = 8,
        weight_type: WeightType = "linear",
        bc_lr: float = 3e-4,
        bc_batch_size: Optional[int] = None,
        bc_pretrain_epochs: int = 250,
        bc_pretrain_steps: Optional[int] = None,
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
        self._init_fql_params(
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            alpha=alpha,
            flow_steps=flow_steps,
            q_agg=q_agg,
            normalize_q_loss=normalize_q_loss,
            net_arch=net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            activation_fn=activation_fn,
            encoder_sharing=encoder_sharing,
            encoder_config=encoder_config,
            obs_groups=obs_groups,
            critic_encoder_config=critic_encoder_config,
        )
        self._init_fac_params(
            fac_alpha=fac_alpha,
            fac_lambda=fac_lambda,
            fac_threshold=fac_threshold,
            logp_method=logp_method,
            logp_hutch_probes=logp_hutch_probes,
            weight_type=weight_type,
            bc_lr=bc_lr,
            bc_batch_size=bc_batch_size,
            bc_pretrain_epochs=bc_pretrain_epochs,
            bc_pretrain_steps=bc_pretrain_steps,
        )

        self._setup_model()


class _FACRolloutTrainingShell(Off2OnReplayMixin, FACCore, OffPolicyAlgorithm):
    """Internal rollout/eval shell wiring ``FACCore`` into ``OffPolicyAlgorithm``.

    .. warning::
       **Do not instantiate this class directly.** Use :class:`Off2OnFAC`.
       Mirrors ``_FINORolloutTrainingShell``'s precedent for this internal
       extension point.
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
        training_freq: int = 64,
        utd: float = 1.0,
        bootstrap_at_done: str = "truncated",
        online_episodes_per_iteration: Optional[int] = None,
        stats_window_size: Optional[int] = None,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        alpha: float = 10.0,  # unused by FAC; kept only for FQLCore-signature compatibility
        flow_steps: int = 10,
        q_agg: Literal["mean", "min"] = "mean",
        normalize_q_loss: bool = False,
        net_arch: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = "xavier_uniform",
        backbone_type: BackboneType = "mlp",
        activation_fn: Optional[Activation] = "gelu",
        encoder_sharing: Optional[EncoderSharing] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        fac_alpha: float = 1.0,
        fac_lambda: float = 1.0,
        fac_threshold: FacThreshold = "batch_adaptive",
        logp_method: LogpMethod = "exact",
        logp_hutch_probes: int = 8,
        weight_type: WeightType = "linear",
        bc_lr: float = 3e-4,
        bc_batch_size: Optional[int] = None,
        bc_pretrain_epochs: int = 250,
        bc_pretrain_steps: Optional[int] = None,
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
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
            online_episodes_per_iteration=online_episodes_per_iteration,
            stats_window_size=stats_window_size,
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
        self._init_fql_params(
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            alpha=alpha,
            flow_steps=flow_steps,
            q_agg=q_agg,
            normalize_q_loss=normalize_q_loss,
            net_arch=net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            activation_fn=activation_fn,
            encoder_sharing=encoder_sharing,
            encoder_config=encoder_config,
            obs_groups=obs_groups,
            critic_encoder_config=critic_encoder_config,
        )
        # offline_sampling must be set (via _init_off2on_params) BEFORE
        # _init_fac_params runs -- unlike FINO's shell, FACCore's own
        # _init_fac_params validates fac_threshold against
        # self.offline_sampling (module docstring's Deviations list), and
        # OffPolicyAlgorithm.__init__ (unlike OfflineRLAlgorithm.__init__)
        # never sets that attribute itself.
        self._init_off2on_params(offline_sampling=offline_sampling)
        self._init_fac_params(
            fac_alpha=fac_alpha,
            fac_lambda=fac_lambda,
            fac_threshold=fac_threshold,
            logp_method=logp_method,
            logp_hutch_probes=logp_hutch_probes,
            weight_type=weight_type,
            bc_lr=bc_lr,
            bc_batch_size=bc_batch_size,
            bc_pretrain_epochs=bc_pretrain_epochs,
            bc_pretrain_steps=bc_pretrain_steps,
        )
        self._setup_model()

    def _apply_online_regularizer_override(self, online_replay_mode: str) -> None:
        """Flip to the reference's ``online_finetuning=True`` critic-penalty
        path (``main.py:223``, ``fac.py:54-65``) at the offline->online
        switch: every subsequent ``train()`` gradient step runs a BC-flow
        update on the same sampled batch first, then recomputes ``logp_beta``
        from that batch's own actions under the just-updated BC flow instead
        of gathering the offline ``_logp_table`` cache. Precedent: the same
        single-flag-flip shape as ``ValueFlowsCore``'s
        ``_apply_online_regularizer_override`` (``value_flows.py``), the
        mixin hook documented for algorithm-specific regularizer-state
        changes at the switch (``rl_garden/algorithms/off2on.py``'s module
        docstring). See ``FACCore.train()`` and ``_logp_beta_online``."""
        del online_replay_mode
        self._online_finetuning = True


class Off2OnFAC(_FACRolloutTrainingShell):
    """Offline-to-online FAC: unchanged ``FACCore.train()`` offline
    (BC-pretrain -> logp-cache -> conservative actor-critic schedule), then
    -- after ``switch_to_online_mode`` -- the reference's
    ``online_finetuning=True`` branch each gradient step (BC update on the
    live sampled batch, then critic/actor updates with an on-the-fly
    recomputed ``logp_beta``). See module docstring."""

    _compatible_checkpoint_algorithms = ("Off2OnFAC", "FAC")
