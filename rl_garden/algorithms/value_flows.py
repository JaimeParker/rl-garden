"""Value Flows (Dong et al., "Value Flows: Bringing Distributional
Reinforcement Learning to Flow Matching Policies", arXiv 2510.07650), ported
from ``3rd_party/value-flows/agents/value_flows.py``, an independent fork of
the official FQL code (``3rd_party/fql``) benchmarked on OGBench and D4RL.

The actor side (BC flow + one-step distillation + Q-loss) is FQL's, unchanged
except the Q term. The critic is a flow-matching model of the *full return
distribution*: twin velocity networks over the scalar-return axis
(``ValueFlowVectorField``, ``rl_garden/networks/value_flow_field.py``),
trained with (a) BCFM, a rectified-flow regression whose endpoint is the TD
target, and (b) DCFM, a Bellman-consistency loss on the vector field at a
partially integrated point, both per-sample weighted by a confidence weight
derived from a JVP of the return flow. There is no separate scalar critic --
Q is a one-Euler-step estimate off the same flow nets
(``ValueFlowsPolicy.one_step_q``). It shares nothing with FloQ except the FQL
ancestor, so it subclasses ``FQLCore`` directly and reuses the FloQ port only
as a structural template.

Built as ``ValueFlowsCore(FQLCore)``: ``_critic_update``/``_actor_update``/
``_update_targets``/``_setup_model``/``_checkpoint_metadata`` are overridden.
``_actor_update`` is a near-copy of ``FQLCore._actor_update`` with the Q term
replaced by ``policy.one_step_q`` (``q_agg`` aggregation, not
``.mean(dim=0)``).

**Rejection-sampling policy extraction (``ValueFlowsPolicy.sample_actions_rs``,
``rl_garden/policies/value_flows_policy.py``)**: ports the reference's
``sample_actions``, ``policy_extraction='rs'`` branch (its own default,
``agents/value_flows.py:343-396``) -- draws ``num_samples`` candidates from
the multi-step BC-flow teacher (``compute_flow_actions``/``actor_flow``, *not*
the one-step student), scores each with ``one_step_q`` off the (non-target)
critic flows, and takes the per-observation argmax. ``ValueFlowsCore
._critic_update``'s TD-target next-action always uses ``sample_actions_rs``,
matching the reference's ``critic_loss`` (``agents/value_flows.py:28``, which
calls ``self.sample_actions(batch['next_observations'], actor_rng)`` *without*
passing ``policy_extraction`` and so gets its ``'rs'`` default) --
independent of ``self.policy_extraction``, which only governs rollout/eval
action selection (``ValueFlowsPolicy.predict``). ``policy_extraction``
defaults to ``'rs'`` and is switched to ``'rpg'`` (the plain one-step actor,
``actor_onestep_flow``) by ``_ValueFlowsRolloutTrainingShell
._apply_online_regularizer_override`` at the offline->online transition,
matching ``main.py``'s own switch: online exploration always uses ``'rpg'``
(``main.py:140``), and eval uses ``'rs'`` through the offline phase and
``'rpg'`` once online (``main.py:200-204``).

**Deviation (plan pseudocode vs. reference, reference followed)**: the
plan's draft pseudocode computes the confidence-weight JVP
(``integrate_returns_with_jvp``) on the *next* transition
(``next_features``/``next_action``). The actual reference
(``agents/value_flows.py::critic_loss``, lines 30-39) computes it on the
*current* transition instead -- ``batch['observations']``/``batch['actions']``,
not ``batch['next_observations']``/the sampled ``next_actions`` used
everywhere else in the same loss. This port follows the reference: the
confidence-weight JVP below runs on ``obs_features``/``data.actions``.

``min_reward``/``max_reward`` (the reference reads them from dataset
statistics) are explicit constructor/CLI args here, defaults ``-1.0``/``0.0``
-- the same simplification as FloQ's ``r_min``/``r_max``.

``_critic_update`` reference facts (verified against
``3rd_party/value-flows/agents/value_flows.py::critic_loss`` and
``compute_flow_returns``), ``B`` = batch, ``K`` = ``n_critics``:

- Confidence weights: for each of the ``K`` *target* flows, integrate the
  full 0->1 return unroll from a shared ``ret_noise`` with JVP tracking
  (tangent initialized to ones), evaluated at ``(obs_features, data.actions)``
  under ``torch.no_grad()``. ``std = agg(|tangent_k|, q_agg)``;
  ``weights = sigmoid(-confidence_weight_temp / std) + 0.5``. This reuses
  ``q_agg`` for a second purpose beyond aggregating Q (matches the
  reference), per ``FQLCore.q_agg``.
- Next action: the plain one-step actor on ``next_obs``, clamped to the
  action bounds (not the reference's default rejection sampling -- see
  above).
- BCFM: ``noise``/``t`` shared by BCFM's ``x_0``/``x_t`` and DCFM's ``x_0``/
  end time. Full 0->1 target-flow unroll from ``noise`` on
  ``(next_features, next_action)``, aggregated by ``ret_agg`` ->
  ``next_returns``; ``td_returns = r + gamma*(1-done)*next_returns``, all
  under ``torch.no_grad()``. ``x_t = t*td_returns + (1-t)*noise``;
  ``vel_target = td_returns - noise``; each of the ``K`` (non-target) flows
  is queried at ``(obs_features, data.actions, x_t, t)``; ``bcfm_loss =
  sum_k (flow_k(...) - vel_target)**2`` (matches the reference's
  ``((vf1-target)**2 + (vf2-target)**2)``, generalized to ``K``).
- DCFM: partial 0->t target-flow unroll from the *same* ``noise`` on
  ``(next_features, next_action)`` -> ``noisy_next_returns`` (``ret_agg``);
  ``noisy_td_returns = r + gamma*(1-done)*noisy_next_returns``; ``target_vf =
  agg(target_flow_k(next_features, next_action, noisy_next_returns, t),
  ret_agg)`` -- all under ``torch.no_grad()``. ``dcfm_loss = sum_k
  (flow_k(obs_features, data.actions, noisy_td_returns, t) - target_vf)**2``.
- ``critic_loss = (weights * (bcfm_lambda*bcfm_loss +
  dcfm_lambda*dcfm_loss)).mean()``.
- Logging-only ``q``: ``policy.one_step_q`` on ``(obs_features,
  data.actions)`` with ``q_agg``, under ``torch.no_grad()`` (never
  backpropagated in the critic update -- only the actor update's Q term uses
  a grad-enabled call).

Every ``torch.no_grad()`` boundary above is explicit in ``_critic_update``,
mirroring the reference's implicit stop-gradient (the JAX reference never
passes ``params=grad_params`` to target-network calls or to the
next-action/next-returns computation; PyTorch has no such implicit
convention).

Targets: Polyak every flow net -> its target with ``tau``, matching
``target_update``'s per-module tree_map in the reference.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.fql import FQLCore
from rl_garden.algorithms.off2on import Off2OnReplayMixin
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.common.logger import Logger
from rl_garden.common.optim import make_lr_scheduler, make_optimizer
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.combined import ImageEncoderFactory
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.networks.actor_vector_field import flow_onestep_distill_loss
from rl_garden.networks.value_flow_field import integrate_returns, integrate_returns_with_jvp
from rl_garden.policies.fql_policy import EncoderSharing
from rl_garden.policies.value_flows_policy import ValueFlowsPolicy, aggregate


class ValueFlowsCore(FQLCore):
    """Value Flows' distributional flow-matching critic on top of
    ``FQLCore``. See module docstring."""

    def _init_value_flows_params(
        self,
        *,
        min_reward: float = -1.0,
        max_reward: float = 0.0,
        ret_agg: Literal["mean", "min"] = "mean",
        confidence_weight_temp: float = 0.3,
        dcfm_lambda: float = 1.0,
        bcfm_lambda: float = 1.0,
        clip_flow_returns: bool = True,
        num_samples: int = 16,
        policy_extraction: Literal["rs", "rpg"] = "rs",
    ) -> None:
        if ret_agg not in ("mean", "min"):
            raise ValueError(f"ret_agg must be 'mean' or 'min', got {ret_agg!r}.")
        if confidence_weight_temp <= 0:
            raise ValueError(
                f"confidence_weight_temp must be positive, got {confidence_weight_temp}."
            )
        if policy_extraction not in ("rs", "rpg"):
            raise ValueError(
                f"policy_extraction must be 'rs' or 'rpg', got {policy_extraction!r}."
            )

        self.min_reward = min_reward
        self.max_reward = max_reward
        self.ret_agg = ret_agg
        self.confidence_weight_temp = confidence_weight_temp
        self.dcfm_lambda = dcfm_lambda
        self.bcfm_lambda = bcfm_lambda
        self.clip_flow_returns = clip_flow_returns
        self.num_samples = num_samples
        self.policy_extraction = policy_extraction

        self.return_clip_range: Optional[tuple[float, float]] = (
            (min_reward / (1.0 - self.gamma), max_reward / (1.0 - self.gamma))
            if clip_flow_returns
            else None
        )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "ret_agg": self.ret_agg,
            "confidence_weight_temp": self.confidence_weight_temp,
            "dcfm_lambda": self.dcfm_lambda,
            "bcfm_lambda": self.bcfm_lambda,
            "clip_flow_returns": self.clip_flow_returns,
            "num_samples": self.num_samples,
            "policy_extraction": self.policy_extraction,
        }

    def _setup_model(self) -> None:
        features_extractor = self._build_features_extractor()
        if self.encoder_sharing == "separate":
            actor_bc_flow_encoder = self._build_features_extractor()
            actor_onestep_flow_encoder = self._build_features_extractor()
        else:
            actor_bc_flow_encoder = None
            actor_onestep_flow_encoder = None
        self.policy = ValueFlowsPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
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
            encoder_sharing=self.encoder_sharing,
            actor_bc_flow_encoder=actor_bc_flow_encoder,
            actor_onestep_flow_encoder=actor_onestep_flow_encoder,
            num_samples=self.num_samples,
            policy_extraction=self.policy_extraction,
            num_flow_steps=self.flow_steps,
            q_agg=self.q_agg,
            return_clip_range=self.return_clip_range,
        ).to(self.device)

        self.critic_optimizer = make_optimizer(
            list(self.policy.critic_and_encoder_parameters()),
            lr=self.critic_lr,
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
                opt,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
            for opt in (self.critic_optimizer, self.actor_optimizer)
        ]

    def _critic_update(self, data, obs_features: torch.Tensor) -> dict[str, float]:
        batch_size = data.actions.shape[0]
        device, dtype = obs_features.device, obs_features.dtype

        with torch.no_grad():
            next_features_critic = self.policy.extract_features(data.next_obs)
            if self.encoder_sharing == "separate":
                next_features_actor = self.policy.extract_actor_onestep_features(
                    data.next_obs
                )
            else:
                next_features_actor = next_features_critic
            # TD-target next action: always rejection sampling, matching the
            # reference's `critic_loss` (`agents/value_flows.py:28`), which
            # calls `self.sample_actions(...)` without `policy_extraction`
            # and so defaults to `'rs'` -- regardless of
            # `self.policy_extraction`, which only governs rollout/eval
            # action selection (see `_ValueFlowsRolloutTrainingShell`'s
            # `_apply_online_regularizer_override`). `sample_actions_rs`
            # already clamps its candidates (via `compute_flow_actions`),
            # so no extra clamp is needed here.
            next_action = self.policy.sample_actions_rs(
                next_features_actor,
                next_features_critic,
                num_samples=self.num_samples,
                clip_range=self.return_clip_range,
                agg=self.q_agg,
            )

            # Confidence weights: JVP of the full 0->1 target unroll, evaluated
            # at the CURRENT (obs, action) pair -- see module docstring's
            # "Deviation from the port plan" note. `obs_features` is already
            # detached-in-effect here (no_grad context), so it's reused
            # directly rather than re-encoded.
            ret_noise = torch.randn(batch_size, 1, device=device, dtype=dtype)
            jac_list = []
            for flow_target in self.policy.critic_flows_target:
                _, jac_k = integrate_returns_with_jvp(
                    flow_target,
                    obs_features,
                    data.actions,
                    ret_noise,
                    steps=self.flow_steps,
                    clip_range=self.return_clip_range,
                )
                jac_list.append(jac_k.abs())
            std = aggregate(torch.stack(jac_list, dim=0), self.q_agg).squeeze(-1)  # (B,)
            weights = torch.sigmoid(-self.confidence_weight_temp / std) + 0.5

        # Shared by BCFM's x_0/x_t and DCFM's x_0/end-time -- this sharing is
        # load-bearing, not incidental (matches the reference's single
        # `noises`/`times` draw reused across both loss terms).
        noise = torch.randn(batch_size, 1, device=device, dtype=dtype)
        t = torch.rand(batch_size, 1, device=device, dtype=dtype)

        with torch.no_grad():
            next_returns_list = [
                integrate_returns(
                    flow_target,
                    next_features_critic,
                    next_action,
                    noise,
                    steps=self.flow_steps,
                    clip_range=self.return_clip_range,
                )
                for flow_target in self.policy.critic_flows_target
            ]
            next_returns = aggregate(torch.stack(next_returns_list, dim=0), self.ret_agg)
            td_returns = data.rewards.unsqueeze(-1) + self.gamma * (
                1.0 - data.dones.unsqueeze(-1)
            ) * next_returns

        x_t = t * td_returns + (1.0 - t) * noise
        vel_target = td_returns - noise
        bcfm_terms = [
            F.mse_loss(
                flow(obs_features, data.actions, x_t, t), vel_target, reduction="none"
            )
            for flow in self.policy.critic_flows
        ]
        bcfm_loss = torch.stack(bcfm_terms, dim=0).sum(dim=0).squeeze(-1)  # (B,)

        with torch.no_grad():
            noisy_next_returns_list = [
                integrate_returns(
                    flow_target,
                    next_features_critic,
                    next_action,
                    noise,
                    steps=self.flow_steps,
                    end_times=t,
                    clip_range=self.return_clip_range,
                )
                for flow_target in self.policy.critic_flows_target
            ]
            noisy_next_returns = aggregate(
                torch.stack(noisy_next_returns_list, dim=0), self.ret_agg
            )
            noisy_td_returns = data.rewards.unsqueeze(-1) + self.gamma * (
                1.0 - data.dones.unsqueeze(-1)
            ) * noisy_next_returns
            target_vf_list = [
                flow_target(next_features_critic, next_action, noisy_next_returns, t)
                for flow_target in self.policy.critic_flows_target
            ]
            target_vf = aggregate(torch.stack(target_vf_list, dim=0), self.ret_agg)

        dcfm_terms = [
            F.mse_loss(
                flow(obs_features, data.actions, noisy_td_returns, t),
                target_vf,
                reduction="none",
            )
            for flow in self.policy.critic_flows
        ]
        dcfm_loss = torch.stack(dcfm_terms, dim=0).sum(dim=0).squeeze(-1)  # (B,)

        critic_loss = (
            weights * (self.bcfm_lambda * bcfm_loss + self.dcfm_lambda * dcfm_loss)
        ).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
        self.critic_optimizer.step()
        if self._lr_schedulers[0] is not None:
            self._lr_schedulers[0].step()

        with torch.no_grad():
            q = self.policy.one_step_q(
                obs_features, data.actions, clip_range=self.return_clip_range, agg=self.q_agg
            )

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "bcfm_loss": float(bcfm_loss.mean().detach().item()),
            "dcfm_loss": float(dcfm_loss.mean().detach().item()),
            "weight": float(weights.mean().detach().item()),
            "q": float(q.mean().detach().item()),
        }

    def _actor_update(self, data, obs_features: torch.Tensor) -> dict[str, float]:
        bc_features, onestep_features, q_features = self.policy.extract_actor_loss_features(
            data.obs, critic_features=obs_features
        )
        batch_size = data.actions.shape[0]
        action_dim = data.actions.shape[-1]
        device, dtype = bc_features.device, bc_features.dtype

        x_0 = torch.randn(batch_size, action_dim, device=device, dtype=dtype)
        t = torch.rand(batch_size, 1, device=device, dtype=dtype)
        x_t = (1 - t) * x_0 + t * data.actions
        vel_target = data.actions - x_0
        pred_vel = self.policy.actor_bc_flow(bc_features, x_t, t)
        bc_flow_loss = F.mse_loss(pred_vel, vel_target)

        noises = torch.randn(batch_size, action_dim, device=device, dtype=dtype)
        actor_actions = self.policy.actor_onestep_flow(onestep_features, noises)
        actor_actions = actor_actions.clamp(self.policy.action_low, self.policy.action_high)
        distill_loss = flow_onestep_distill_loss(
            self.policy.actor_bc_flow,
            actor_actions,
            bc_features,
            noises,
            self.flow_steps,
            low=self.policy.action_low,
            high=self.policy.action_high,
        )

        # Q term: one-Euler-step estimate off the flow critics, aggregated by
        # q_agg (not FQLCore's `.mean(dim=0)` -- Value Flows has no
        # per-critic-ensemble tensor to mean over the same way).
        q_pi = self.policy.one_step_q(
            q_features, actor_actions, clip_range=self.return_clip_range, agg=self.q_agg
        )
        q_loss = -q_pi.mean()
        if self.normalize_q_loss:
            lam = (1.0 / q_pi.abs().mean()).detach()
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.alpha * distill_loss + q_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip_grad_norm(self.policy.actor_parameters())
        self.actor_optimizer.step()
        if self._lr_schedulers[1] is not None:
            self._lr_schedulers[1].step()

        return {
            "actor_loss": float(actor_loss.detach().item()),
            "bc_flow_loss": float(bc_flow_loss.detach().item()),
            "distill_loss": float(distill_loss.detach().item()),
            "q_loss": float(q_loss.detach().item()),
        }

    def _update_targets(self) -> None:
        for flow, flow_target in zip(
            self.policy.critic_flows, self.policy.critic_flows_target
        ):
            polyak_update(flow.parameters(), flow_target.parameters(), self.tau)


class ValueFlows(ValueFlowsCore, OfflineRLAlgorithm):
    """Offline Value Flows: twin flow-matching critics over the return
    distribution + two-network flow-matching actor."""

    _compatible_checkpoint_algorithms = ("ValueFlows",)

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
        alpha: float = 10.0,
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
        encoder_sharing: EncoderSharing = "shared",
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        min_reward: float = -1.0,
        max_reward: float = 0.0,
        ret_agg: Literal["mean", "min"] = "mean",
        confidence_weight_temp: float = 0.3,
        dcfm_lambda: float = 1.0,
        bcfm_lambda: float = 1.0,
        clip_flow_returns: bool = True,
        num_samples: int = 16,
        policy_extraction: Literal["rs", "rpg"] = "rs",
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
            image_encoder_factory=image_encoder_factory,
        )
        self._init_value_flows_params(
            min_reward=min_reward,
            max_reward=max_reward,
            ret_agg=ret_agg,
            confidence_weight_temp=confidence_weight_temp,
            dcfm_lambda=dcfm_lambda,
            bcfm_lambda=bcfm_lambda,
            clip_flow_returns=clip_flow_returns,
            num_samples=num_samples,
            policy_extraction=policy_extraction,
        )

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, (spaces.Box, spaces.Dict)):
            raise TypeError(
                f"ValueFlows supports only Box or Dict observation spaces, got {type(obs_space)}"
            )

        self._setup_model()


class _ValueFlowsRolloutTrainingShell(Off2OnReplayMixin, ValueFlowsCore, OffPolicyAlgorithm):
    """Internal rollout/eval shell wiring ``ValueFlowsCore`` into
    ``OffPolicyAlgorithm``.

    .. warning::
       **Do not instantiate this class directly.** Use
       :class:`Off2OnValueFlows`. Mirrors ``_FloQRolloutTrainingShell``'s
       precedent for this internal extension point.
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
        alpha: float = 10.0,
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
        encoder_sharing: EncoderSharing = "shared",
        min_reward: float = -1.0,
        max_reward: float = 0.0,
        ret_agg: Literal["mean", "min"] = "mean",
        confidence_weight_temp: float = 0.3,
        dcfm_lambda: float = 1.0,
        bcfm_lambda: float = 1.0,
        clip_flow_returns: bool = True,
        num_samples: int = 16,
        policy_extraction: Literal["rs", "rpg"] = "rs",
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
        # Dict observation encoding (see Off2OnReplayMixin._configure_observation_kwargs)
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: Optional[tuple[str, ...]] = None,
        state_key: Optional[str] = None,
        use_proprio: Optional[bool] = None,
        proprio_latent_dim: Optional[int] = None,
        image_fusion_mode: Optional[str] = None,
        enable_stacking: Optional[bool] = None,
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
        self._configure_observation_kwargs(
            env,
            image_encoder_factory=image_encoder_factory,
            image_keys=image_keys,
            state_key=state_key,
            use_proprio=use_proprio,
            proprio_latent_dim=proprio_latent_dim,
            image_fusion_mode=image_fusion_mode,
            enable_stacking=enable_stacking,
        )
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
            image_encoder_factory=image_encoder_factory,
        )
        self._init_value_flows_params(
            min_reward=min_reward,
            max_reward=max_reward,
            ret_agg=ret_agg,
            confidence_weight_temp=confidence_weight_temp,
            dcfm_lambda=dcfm_lambda,
            bcfm_lambda=bcfm_lambda,
            clip_flow_returns=clip_flow_returns,
            num_samples=num_samples,
            policy_extraction=policy_extraction,
        )
        self._init_off2on_params(offline_sampling=offline_sampling)
        self._setup_model()

    def _apply_online_regularizer_override(self, online_replay_mode: str) -> None:
        """Switch rollout/eval action selection to `'rpg'` at the
        offline->online transition, matching `main.py`'s own switch (online
        exploration always passes `policy_extraction='rpg'`, `main.py:140`;
        eval uses `'rs'` through the offline phase and `'rpg'` once
        `i > FLAGS.offline_steps`, `main.py:200-204`). The TD-target next
        action in `_critic_update` is unaffected -- it is always `'rs'`
        regardless of this switch (see `ValueFlowsCore._critic_update`)."""
        del online_replay_mode
        self.policy_extraction = "rpg"
        self.policy.policy_extraction = "rpg"
        if self.logger:
            self.logger.add_summary("off2on/policy_extraction", "rpg")


class Off2OnValueFlows(_ValueFlowsRolloutTrainingShell):
    """Offline-to-online Value Flows (FQL/FloQ's off2on pattern, no action
    chunking). See module docstring."""

    _compatible_checkpoint_algorithms = ("Off2OnValueFlows", "ValueFlows")
