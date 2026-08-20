"""QGF (Q-Guided Flow): test-time gradient-guided flow-matching RL
(``3rd_party/qgf/agents/qgf.py``, arXiv 2606.11087).

Unlike FQL/ACFQL, QGF never trains its actor with an RL objective: the actor
is pure BC flow matching, and the critic/value are trained IQL-style
(expectile-regression ``ValueNetwork`` bootstrap, no actor sampling needed
for the critic's TD target) -- fully decoupled from the actor. All "policy
improvement" happens at inference time via ``QGFPolicy``'s guided denoising
(see that module's docstring). ``QGFPolicy.sampling_mode`` also covers all
four other test-time-guidance baselines from the same reference repo: GradStep
(post-hoc gradient ascent in clean action space), IFQL (best-of-N rejection
sampling, no gradient at all), BPTT (full backprop through a remaining-steps
rollout at every denoising step), and RobustQ (guidance from a separate
noise-conditioned critic) -- one algorithm, one settable mode, rather than
five separate registrations, mirroring ``ACFQLPolicy``'s existing
``best-of-n`` vs ``distill-ddpg`` precedent. BPTT and RobustQ are
**reconstructions** of code broken in the reference itself
(``agents/bptt.py:48`` calls a ``self._bc_flow_from`` that does not exist
anywhere in the tree; ``agents/robust_q.py:148`` references an undefined
``cfg``) -- see ``QGFPolicy``'s docstring for exactly what was reconstructed.

Built directly on ``OfflineRLAlgorithm`` (no rollout shell at all, following
``FQL``'s own precedent) since QGF's training never touches an env, even
with ``horizon_length > 1`` -- action chunking here only changes what a
"macro-action" means to the offline dataset/critic, not how training data is
collected. (Live-env eval with ``horizon_length > 1`` would additionally
need per-env chunk-queue rollout logic akin to ``ChunkedRolloutMixin``, which
``OfflineRLAlgorithm`` does not have; out of scope for v1, which targets
OGBench-free validation against existing offline datasets.)

Formulas verified against ``3rd_party/qgf/agents/qgf.py`` directly:

- **Policy loss** (``qgf.py:56-81``): BC flow matching, ``t`` sampled on a
  **discrete grid** by default (``t_sampling="grid"``:
  ``randint(0, denoise_steps+1) / denoise_steps``) -- NOT the continuous
  ``torch.rand`` rl-garden's ``FQL``/``ACFQL`` use. ``t_sampling="uniform"``
  reproduces IFQL's own actor loss (``agents/ifql.py:72``,
  ``jax.random.uniform``) instead. Deliberately **no ``valid`` masking**
  here, matching the reference exactly (``qgf.py:80`` has none) -- this is a
  genuine difference from ``ACFQLCore``'s BC-flow loss, which *does* mask
  per-position (its own reference, ``acfql.py:73-79``, does too). Each port
  faithfully reproduces its own reference's actual formula.
- **Critic loss** (``qgf.py:83-94``): ``target_q = rewards + discount**H *
  masks * next_v`` where ``next_v = value(next_obs)`` -- a straight
  V-network bootstrap, not a Q-ensemble one. Maps directly onto
  ``ChunkedTensorReplayBuffer``'s ``rewards``/``discounts`` fields (which
  already fold ``discount**H * masks[...,-1]`` together for both
  ``horizon_length=1`` and ``>1``).
- **Value loss** (``qgf.py:96-106``): expectile regression of ``V(obs)``
  toward ``aggregate_q(target_critic(obs, batch_actions))`` -- **``q_agg``
  defaults to ``"min"``** here (not ``"mean"`` like FQL), and gates both this
  bootstrap and the inference-time guidance gradient (``qgf.py:207``).
- **``valid`` masking omitted in critic/value loss**: the reference masks
  both with ``valid[...,-1]`` (whole-window discard for windows that ran
  past a terminal). This port omits it, for the same reason
  ``ACRLPDCore``/``ACFQLCore`` already omit the analogous masking on this
  buffer: ``ChunkedTensorReplayBuffer`` stops reward/discount accumulation
  exactly at the first true terminal (see its module docstring), so the
  target it produces is already correct/unbiased for every sampled window --
  QC's own ``valid[...,-1]``-discard convention is redundant on top of that,
  not a fidelity gap. The critic's/value's LHS input (``batch_actions``, the
  full action-chunk window) can still contain garbage past-terminal tail
  positions in the rare case a window runs into the next episode -- exactly
  the same acknowledged tradeoff ``ACFQLCore``'s critic loss already makes
  with its own ``flat_actions`` input, extended here to ``value_loss`` too.
- **Combined backward, two optimizers**: unlike the reference's three
  separate per-network ``optax`` states (``qgf.py:128-151``), this port
  follows rl-garden ``IQLCore``'s existing convention -- one
  ``critic_value_optimizer`` (critic + value + encoder) and one
  ``actor_optimizer``, one combined ``total_loss.backward()``. Valid because
  gradients are additive and disjoint per network (each loss only touches
  its own network's params; cross terms are explicitly detached).

Deliberately duplicates rl-garden ``IQL``'s critic/value loss shape rather
than extracting a shared mixin -- see this module's design note in the
project plan: the two losses are not actually the same formula (IQL has no
H-step/valid-masking/configurable-aggregation), so extracting one would mean
generalizing shipped, tested ``IQL`` to support features it doesn't need.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.chunked_replay_buffer import ChunkedTensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.policies.qgf_policy import (
    DenoisedActionApprox,
    QGFPolicy,
    SamplingMode,
)


class QGFCore:
    """Shared QGF loss/network logic. See module docstring."""

    def _init_qgf_params(
        self,
        *,
        horizon_length: int = 1,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_value_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        expectile: float = 0.9,
        q_agg: Literal["mean", "min"] = "min",
        n_critics: int = 2,
        denoise_steps: int = 10,
        t_sampling: Literal["grid", "uniform"] = "grid",
        sampling_mode: SamplingMode = "guided",
        guidance_weight: float = 1.0,
        denoised_action_approx: DenoisedActionApprox = "one_euler_step_approx",
        qgrad_step_size: float = 0.1,
        qgrad_steps: int = 1,
        use_sign_gradient: bool = False,
        actor_num_samples: int = 32,
        robust_critic_lr: float = 3e-4,
        robust_critic_t_emb_size: int = 16,
        net_arch: Optional[Sequence[int]] = None,
        actor_use_layer_norm: bool = True,
        critic_use_layer_norm: bool = True,
        value_use_layer_norm: bool = True,
        kernel_init: Optional[KernelInit] = "xavier_uniform",
        backbone_type: BackboneType = "mlp",
        activation_fn: Optional[Activation] = "gelu",
    ) -> None:
        if horizon_length < 1:
            raise ValueError(f"horizon_length must be >= 1, got {horizon_length}")
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {tau}.")
        if not (0.0 < expectile < 1.0):
            raise ValueError(f"expectile must be in (0, 1), got {expectile}.")
        if q_agg not in ("mean", "min"):
            raise ValueError(f"q_agg must be 'mean' or 'min', got {q_agg!r}.")
        if denoise_steps < 1:
            raise ValueError(f"denoise_steps must be >= 1, got {denoise_steps}.")
        if t_sampling not in ("grid", "uniform"):
            raise ValueError(f"t_sampling must be 'grid' or 'uniform', got {t_sampling!r}.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )

        self.horizon_length = horizon_length
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_value_lr = critic_value_lr
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip_norm = grad_clip_norm
        self.expectile = expectile
        self.q_agg = q_agg
        self.n_critics = n_critics
        self.denoise_steps = denoise_steps
        self.t_sampling = t_sampling
        self.sampling_mode = sampling_mode
        self.guidance_weight = guidance_weight
        self.denoised_action_approx = denoised_action_approx
        self.qgrad_step_size = qgrad_step_size
        self.qgrad_steps = qgrad_steps
        self.use_sign_gradient = use_sign_gradient
        self.actor_num_samples = actor_num_samples
        self.robust_critic_lr = robust_critic_lr
        self.robust_critic_t_emb_size = robust_critic_t_emb_size
        self.net_arch: list[int] = (
            list(net_arch) if net_arch is not None else [512, 512, 512, 512]
        )
        self.actor_use_layer_norm = actor_use_layer_norm
        self.critic_use_layer_norm = critic_use_layer_norm
        self.value_use_layer_norm = value_use_layer_norm
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.activation_fn = activation_fn

    def _optimizer_names(self) -> tuple[str, ...]:
        names = ("critic_value_optimizer", "actor_optimizer")
        if self.sampling_mode == "robust_q":
            names = names + ("robust_critic_optimizer",)
        return names

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "horizon_length": self.horizon_length,
            "tau": self.tau,
            "actor_lr": self.actor_lr,
            "critic_value_lr": self.critic_value_lr,
            "weight_decay": self.weight_decay,
            "use_adamw": self.use_adamw,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "grad_clip_norm": self.grad_clip_norm,
            "expectile": self.expectile,
            "q_agg": self.q_agg,
            "n_critics": self.n_critics,
            "denoise_steps": self.denoise_steps,
            "t_sampling": self.t_sampling,
            "sampling_mode": self.sampling_mode,
            "guidance_weight": self.guidance_weight,
            "denoised_action_approx": self.denoised_action_approx,
            "qgrad_step_size": self.qgrad_step_size,
            "qgrad_steps": self.qgrad_steps,
            "use_sign_gradient": self.use_sign_gradient,
            "actor_num_samples": self.actor_num_samples,
            "robust_critic_lr": self.robust_critic_lr,
            "robust_critic_t_emb_size": self.robust_critic_t_emb_size,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
        }

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

    def _policy_action_space(self) -> spaces.Box:
        raw = self.env.single_action_space
        assert isinstance(raw, spaces.Box), "QGF requires a flat Box action space."
        low = np.tile(np.asarray(raw.low, dtype=np.float32).reshape(-1), self.horizon_length)
        high = np.tile(np.asarray(raw.high, dtype=np.float32).reshape(-1), self.horizon_length)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _build_replay_buffer(self) -> ChunkedTensorReplayBuffer:
        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("QGF is state-only (Box observations); vision is out of scope.")
        return ChunkedTensorReplayBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            horizon_length=self.horizon_length,
            gamma=self.gamma,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _setup_model(self) -> None:
        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("QGF is state-only (Box observations); vision is out of scope.")
        features_extractor = FlattenExtractor(observation_space=obs_space)
        self.policy = QGFPolicy(
            observation_space=obs_space,
            action_space=self._policy_action_space(),
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            value_use_layer_norm=self.value_use_layer_norm,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            activation_fn=self.activation_fn,
            q_agg=self.q_agg,
            denoise_steps=self.denoise_steps,
            sampling_mode=self.sampling_mode,
            guidance_weight=self.guidance_weight,
            denoised_action_approx=self.denoised_action_approx,
            qgrad_step_size=self.qgrad_step_size,
            qgrad_steps=self.qgrad_steps,
            use_sign_gradient=self.use_sign_gradient,
            actor_num_samples=self.actor_num_samples,
            robust_critic_lr=self.robust_critic_lr,
            robust_critic_t_emb_size=self.robust_critic_t_emb_size,
        ).to(self.device)

        self.critic_value_optimizer = make_optimizer(
            list(self.policy.critic_and_value_parameters()),
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
        if self.policy.robust_critic is not None:
            self.robust_critic_optimizer = make_optimizer(
                list(self.policy.robust_critic_parameters()),
                lr=self.robust_critic_lr,
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
            for opt in (self.critic_value_optimizer, self.actor_optimizer)
        ]

    def _sample_train_batch(self, batch_size: int):
        if self.offline_sampling == "with_replace":
            return self.replay_buffer.sample(batch_size)
        if self.offline_sampling == "without_replace":
            sample = getattr(self.replay_buffer, "sample_without_replace", None)
            if sample is None:
                raise ValueError(
                    "offline_sampling='without_replace' requires a replay buffer "
                    "with sample_without_replace()."
                )
            return sample(batch_size)
        raise ValueError(f"Unknown offline_sampling: {self.offline_sampling!r}")

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        return weight * diff.pow(2)

    def _sample_flow_time(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self.t_sampling == "grid":
            t_idx = torch.randint(0, self.denoise_steps + 1, (batch_size,), device=device)
            t = t_idx.to(dtype) / self.denoise_steps
        else:  # "uniform" (IFQL-faithful actor training)
            t = torch.rand(batch_size, device=device, dtype=dtype)
        return t.unsqueeze(-1)

    def _compute_losses(self, data) -> tuple[torch.Tensor, dict[str, float]]:
        flat_actions = data.actions.reshape(data.actions.shape[0], -1)
        features = self.policy.extract_features(data.obs, stop_gradient=False)

        # --- critic: Q(obs, action_chunk) -> rewards + discounts * next_v ---
        # (next_v from the LIVE value net, not a target one -- QGF has none;
        # matches rl-garden IQL's own convention, iql.py:483-493.)
        with torch.no_grad():
            next_features = self.policy.extract_features(data.next_obs, stop_gradient=False)
            next_v = self.policy.value(next_features)
            target_q = data.rewards.unsqueeze(-1) + data.discounts.unsqueeze(-1) * next_v
        q_all = self.policy.q_values_all(features, flat_actions, target=False)
        critic_loss = F.mse_loss(q_all, target_q.unsqueeze(0).expand_as(q_all))

        # --- value: expectile regression onto aggregate_q(target_critic) ---
        with torch.no_grad():
            target_qs = self.policy.q_values_all(features.detach(), flat_actions, target=True)
            target_q_for_value = self.policy._aggregate_q(target_qs)
        values = self.policy.value(features)
        value_loss = self._expectile_loss(target_q_for_value - values).mean()

        # --- policy: BC flow matching, unmasked (matches qgf.py:56-81) ---
        batch_size = flat_actions.shape[0]
        a0 = torch.randn_like(flat_actions)
        t = self._sample_flow_time(
            batch_size, device=flat_actions.device, dtype=flat_actions.dtype
        )
        a_t = (1 - t) * a0 + t * flat_actions
        vel_target = flat_actions - a0
        pred_vel = self.policy.actor(features, a_t, t)
        bc_loss = F.mse_loss(pred_vel, vel_target)

        total_loss = critic_loss + value_loss + bc_loss
        metrics = {
            "loss": float(total_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "bc_loss": float(bc_loss.detach().item()),
            "q": float(q_all.detach().mean().item()),
            "target_q": float(target_q.detach().mean().item()),
            "v": float(values.detach().mean().item()),
        }
        return total_loss, metrics

    def _robust_critic_loss(self, data) -> tuple[torch.Tensor, dict[str, float]]:
        """RobustQ's noise-conditioned critic loss (``robust_q.py:30-62``):
        regress ``Q_robust(s, a_t, t)`` onto the *clean* target-critic Q,
        for ``a_t`` noised along the flow path at a **continuous-uniform**
        ``t`` -- always continuous, independent of ``t_sampling`` (RobustQ's
        own reference never references that knob, it's a QGF-policy-loss-
        only setting; matches IFQL's own convention of always using
        ``jax.random.uniform`` too)."""
        flat_actions = data.actions.reshape(data.actions.shape[0], -1)
        features = self.policy.extract_features(data.obs, stop_gradient=False)
        with torch.no_grad():
            target_qs = self.policy.q_values_all(features.detach(), flat_actions, target=True)
            target_q = self.policy._aggregate_q(target_qs)

        x0 = torch.randn_like(flat_actions)
        t = torch.rand(flat_actions.shape[0], 1, device=flat_actions.device, dtype=flat_actions.dtype)
        a_t = x0 * (1 - t) + flat_actions * t
        robust_q = self.policy._robust_q_value(features.detach(), a_t, t)
        robust_critic_loss = F.mse_loss(robust_q, target_q.unsqueeze(0).expand_as(robust_q))
        return robust_critic_loss, {
            "robust_critic_loss": float(robust_critic_loss.detach().item()),
            "robust_q_mean": float(robust_q.detach().mean().item()),
        }

    def _step_schedulers(self) -> None:
        for sched in self._lr_schedulers:
            if sched is not None:
                sched.step()

    def _clip_grad_norm(self) -> None:
        if self.grad_clip_norm is None:
            return
        params = list(self.policy.critic_and_value_parameters()) + list(
            self.policy.actor_parameters()
        )
        torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
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

            polyak_update(
                self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau
            )

            if self.sampling_mode == "robust_q":
                robust_loss, robust_info = self._robust_critic_loss(data)
                self.robust_critic_optimizer.zero_grad(set_to_none=True)
                robust_loss.backward()
                self.robust_critic_optimizer.step()
                metrics.update(robust_info)

            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        if not compute_info:
            return {}
        return {key: value / gradient_steps for key, value in metrics_sum.items()}


class QGF(QGFCore, OfflineRLAlgorithm):
    """Offline QGF: BC flow-matching actor + IQL critic/value, guided
    denoising at inference. See module docstring."""

    _compatible_checkpoint_algorithms = ("QGF",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        gamma: float = 0.99,
        offline_sampling: str = "with_replace",
        horizon_length: int = 1,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_value_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        expectile: float = 0.9,
        q_agg: Literal["mean", "min"] = "min",
        n_critics: int = 2,
        denoise_steps: int = 10,
        t_sampling: Literal["grid", "uniform"] = "grid",
        sampling_mode: SamplingMode = "guided",
        guidance_weight: float = 1.0,
        denoised_action_approx: DenoisedActionApprox = "one_euler_step_approx",
        qgrad_step_size: float = 0.1,
        qgrad_steps: int = 1,
        use_sign_gradient: bool = False,
        actor_num_samples: int = 32,
        robust_critic_lr: float = 3e-4,
        robust_critic_t_emb_size: int = 16,
        net_arch: Optional[Sequence[int]] = None,
        actor_use_layer_norm: bool = True,
        critic_use_layer_norm: bool = True,
        value_use_layer_norm: bool = True,
        kernel_init: Optional[KernelInit] = "xavier_uniform",
        backbone_type: BackboneType = "mlp",
        activation_fn: Optional[Activation] = "gelu",
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
        self._init_qgf_params(
            horizon_length=horizon_length,
            tau=tau,
            actor_lr=actor_lr,
            critic_value_lr=critic_value_lr,
            weight_decay=weight_decay,
            use_adamw=use_adamw,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            expectile=expectile,
            q_agg=q_agg,
            n_critics=n_critics,
            denoise_steps=denoise_steps,
            t_sampling=t_sampling,
            sampling_mode=sampling_mode,
            guidance_weight=guidance_weight,
            denoised_action_approx=denoised_action_approx,
            qgrad_step_size=qgrad_step_size,
            qgrad_steps=qgrad_steps,
            use_sign_gradient=use_sign_gradient,
            actor_num_samples=actor_num_samples,
            robust_critic_lr=robust_critic_lr,
            robust_critic_t_emb_size=robust_critic_t_emb_size,
            net_arch=net_arch,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            value_use_layer_norm=value_use_layer_norm,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            activation_fn=activation_fn,
        )

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(f"QGF supports only Box observation spaces, got {type(obs_space)}")

        self._setup_model()
