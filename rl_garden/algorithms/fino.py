"""FINO (Shin et al., "Flow Matching with Injected Noise for Offline-to-Online
RL", ICLR 2026), ported from ``3rd_party/FINO/agents/fino.py``, a fork of the
official FQL code (``3rd_party/fql``). ``networks.py``/``datasets.py``/
``evaluation.py`` are identical to FQL; the critic loss, distill loss, Q
loss, and target update are all identical to FQL too -- FINO needs **no
critic override at all**, unlike FloQ/Value Flows.

The only loss change is in the BC-flow branch of ``actor_loss``: the teacher
network's input point gets Gaussian noise injected, scaled by a schedule
that fades to ~0 at flow-time ``t=0`` and is full-strength at ``t=1``; the
regression target (``x_1 - x_0``) is untouched. Built as
``FINOCore(FQLCore)``: only ``_actor_update``/``_setup_model``/
``_checkpoint_metadata`` are overridden; ``_critic_update``/``_update_targets``
are inherited unchanged from ``FQLCore`` (the reference's critic TD-target
next action, ``sample_batch_actions``, is a plain one-step-flow-plus-clip,
identical to ``FQLCore._critic_update``'s existing ``next_action``
computation).

Inference (``sample_actions``/``predict``, see ``FINOPolicy``) is also new:
instead of a plain one-step draw, FINO samples ``K`` one-step candidates per
observation from the live (non-target) critic, then either argmaxes (eval)
or Boltzmann-samples (online exploration) over them, controlled by a
``beta`` temperature-like scale.

Deviations from the reference:

- ``beta`` is a fixed constructor/CLI arg (default ``10.0``), not adapted
  online. The reference adapts ``beta`` after every eval via
  ``update_entropy`` (fino.py:215-232, a scikit-learn ``GaussianMixture``
  entropy estimate over sampled actions,
  ``beta <- max(0, beta - 0.1*(-action_dim - entropy))``); rl-garden has no
  post-eval algorithm hook today, so this port fixes ``beta``.
- The reference's eval-time ``actions[jnp.argmax(q)]`` (fino.py:188-190) is a
  bug: ``q`` has shape ``(K, B)`` at that point and ``jnp.argmax(q)`` with no
  ``axis`` returns a single flat index into the whole ``K*B`` array, used to
  index ``actions``' *first* axis only -- for ``B > 1`` this silently picks
  the wrong candidate for most observations. This port implements the
  intended per-observation argmax instead (see ``FINOPolicy.sample_actions``).
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
from rl_garden.encoders.combined import ImageEncoderFactory
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.networks.actor_vector_field import flow_onestep_distill_loss
from rl_garden.policies.fino_policy import EncoderSharing, FINOPolicy


class FINOCore(FQLCore):
    """FINO's noise-injected BC-flow actor training on top of ``FQLCore``.
    See module docstring."""

    def _init_fino_params(
        self,
        *,
        noise_scale: float = 0.1,
        beta: float = 10.0,
        num_samples: Optional[int] = None,
    ) -> None:
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0, got {noise_scale}.")
        self.noise_scale = noise_scale
        self.beta = beta  # policy owns the live value after _setup_model
        self.num_samples = num_samples  # possibly None until _setup_model resolves it

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "noise_scale": self.noise_scale,
            "beta": self.policy.beta,
            "num_samples": self.num_samples,
        }

    def _setup_model(self) -> None:
        features_extractor = self._build_features_extractor()
        if self.encoder_sharing == "separate":
            actor_bc_flow_encoder = self._build_features_extractor()
            actor_onestep_flow_encoder = self._build_features_extractor()
        else:
            actor_bc_flow_encoder = None
            actor_onestep_flow_encoder = None
        self.policy = FINOPolicy(
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
            beta=self.beta,
            num_samples=self.num_samples,
            q_agg=self.q_agg,
        ).to(self.device)
        # Resolve None -> concrete int so _checkpoint_metadata() reports the
        # value actually used, independent of the action-space arithmetic
        # that produced it.
        self.num_samples = self.policy.num_samples

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

    def _actor_update(self, data, obs_features: torch.Tensor) -> dict[str, float]:
        # In "shared" mode bc/onestep/q features all alias one detached
        # forward through the shared encoder (today's behavior). In
        # "separate" mode each is a fresh, grad-enabled forward through
        # that network's own encoder -- q_features in particular cannot
        # reuse `obs_features` above: its graph was already consumed by
        # critic_loss.backward(), so PyTorch would raise on a second
        # backward through it. See FQLPolicy.extract_actor_loss_features.
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

        # FINO-only change: the teacher network sees a noise-injected input
        # point, not x_t directly. The injected noise fades to ~0 at t=0 and
        # is full-strength at t=1 (see module docstring).
        noise_scale_t = torch.exp(10.0 * (t - 1.0)) * self.noise_scale
        injected = x_t + noise_scale_t * torch.randn_like(x_0)
        pred_vel = self.policy.actor_bc_flow(bc_features, injected, t)
        bc_flow_loss = F.mse_loss(pred_vel, vel_target)

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
        q_all_pi = self.policy.q_values_all(
            q_features, clipped_actions, target=False
        )
        q_pi = q_all_pi.mean(dim=0)  # actor loss always averages the ensemble
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


class FINO(FINOCore, OfflineRLAlgorithm):
    """Offline FINO: twin-Q critic + two-network flow-matching actor with
    noise-injected BC-flow training."""

    _compatible_checkpoint_algorithms = ("FINO",)

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
        noise_scale: float = 0.1,
        beta: float = 10.0,
        num_samples: Optional[int] = None,
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
        self._init_fino_params(
            noise_scale=noise_scale,
            beta=beta,
            num_samples=num_samples,
        )

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, (spaces.Box, spaces.Dict)):
            raise TypeError(
                f"FINO supports only Box or Dict observation spaces, got {type(obs_space)}"
            )

        self._setup_model()


class _FINORolloutTrainingShell(Off2OnReplayMixin, FINOCore, OffPolicyAlgorithm):
    """Internal rollout/eval shell wiring ``FINOCore`` into ``OffPolicyAlgorithm``.

    .. warning::
       **Do not instantiate this class directly.** Use :class:`Off2OnFINO`.
       Mirrors ``_FloQRolloutTrainingShell``'s precedent for this internal
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
        noise_scale: float = 0.1,
        beta: float = 10.0,
        num_samples: Optional[int] = None,
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
        self._init_fino_params(
            noise_scale=noise_scale,
            beta=beta,
            num_samples=num_samples,
        )
        self._init_off2on_params(offline_sampling=offline_sampling)
        self._setup_model()


class Off2OnFINO(_FINORolloutTrainingShell):
    """Offline-to-online FINO (FloQ's off2on pattern, no critic-flow
    machinery). See module docstring."""

    _compatible_checkpoint_algorithms = ("Off2OnFINO", "FINO")
