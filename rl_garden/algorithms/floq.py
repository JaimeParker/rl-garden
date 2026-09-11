"""FloQ (Farebrother et al., "floq: Training Critics Via Flow-Matching For
Scaling Compute In Value-Based RL", arXiv 2509.06863), ported from
``3rd_party/floq/agents/floq.py``, a fork of the official FQL code
(``3rd_party/fql``). Diffing the two repos shows the actor side (BC flow +
one-step distillation + Q-loss) is unchanged FQL; the only substantive
addition is the critic: instead of a plain scalar Q-network trained by TD
MSE, floq trains a velocity field over the scalar-return axis
(``CriticVectorField``, ``rl_garden/networks/critic_vector_field.py``) with a
flow-matching TD loss, bootstrapped by Euler-integrating a Polyak target
velocity field (``floq_target``). A cheap plain scalar critic (FQL's own
``critic``, unchanged) is distilled from the flow critic's integrated
returns and used for the actor's Q-loss, so the actor never has to
Euler-integrate at rollout/update time.

Built as ``FloQCore(FQLCore)``: only ``_critic_update``/``_update_targets``/
``_setup_model``/``_checkpoint_metadata`` are overridden; ``_actor_update`` is
inherited unchanged from ``FQLCore`` (it only ever queries the plain
``critic``, never ``floq``/``critic_target``, so it needs no floq awareness).

``_critic_update`` reference facts (verified line-by-line against
``3rd_party/floq/agents/floq.py::critic_loss``):

- Three random draws, all ``(B*noise_samples, 1)`` (``B`` = batch, `R` =
  ``noise_samples``): ``next_noise_ratios`` (target unroll start),
  ``noise_ratios`` (shared by BOTH the current-returns unroll start AND the
  flow-matching ``x_0`` -- this sharing is load-bearing, not incidental),
  and ``t`` (``= 0`` everywhere when ``train_at_zero_only``).
- ``B -> B*R`` repetition uses ``repeat_interleave`` (matches the reference's
  ``x[:,None].repeat(n,1).reshape(B*n,...)``), never ``.repeat(R,1)`` --
  those interleave batch elements differently and would silently corrupt the
  reshape-back-to-``(B,R)`` step below.
- Target: ``floq_target`` integrated from ``next_noise_ratios`` -> ``(E,B*R,1)``
  -> aggregated over the ensemble dim (``_aggregate_target_q``, mean or min
  per ``q_agg``) -> reshaped ``(B,R)`` -> ``target = (r + gamma*(1-done)*next_returns)
  .mean(-1)`` -> ``(B,)``. Computed fully inside ``torch.no_grad()``.
- Current returns (the plain critic's distillation target): ``floq``
  integrated from ``noise_ratios`` on the dataset action -> ``(E,B*R,1)`` ->
  reshaped ``(E,B,R)`` -> mean over ``R`` then mean over ``E`` -> ``(B,)``.
  Also fully inside ``torch.no_grad()`` -- ``mean_current_returns`` must carry
  no gradient at all, it is a pure TD-style target for the distilled critic.
  In shared-encoder mode this unroll runs on ``obs_features.detach()``
  (repeated), an intentional simplification of the official reference's
  separate ``target_floq_encoder`` -- floq has only one encoder
  (``features_extractor``, shared with the plain critic) in this port, so
  there is no separate encoder to route the no-grad current-returns unroll
  through; detaching the shared encoder's features here is the equivalent.
- Flow-matching loss: ``x_0 = noise_ratios*noise_max + (1-noise_ratios)*noise_min``
  ``(B*R,1)``; ``x_1`` = the (no-grad) target returns broadcast to
  ``(E,B*R,1)``; ``x_t = (1-t)*x_0 + t*x_1``; ``pred = floq(features, actions,
  x_t, t)`` (grad-enabled: this is the term that trains both ``floq`` and,
  via the shared encoder, ``features_extractor``); ``floq_loss =
  ((pred-(x_1-x_0))**2)`` reshaped ``(E,B,R)`` and summed over ``R`` ->
  ``(E,B)``.
- Distilled loss: the plain critic's ``(q_all - mean_current_returns)**2``
  via ``FQLCore._critic_loss`` (mean over the critic ensemble, matching the
  official ``(floq_loss + distilled).mean()`` up to the ``n_critics ==
  flow_num_ensembles`` case; well-defined for any ``n_critics`` otherwise).
- ``critic_loss = floq_loss.mean() + distilled_loss`` (``distilled_loss`` is
  already a scalar out of ``_critic_loss``).

Only ``floq`` has a Polyak target; the plain distilled critic has none
(``_update_targets`` therefore does not call ``super()`` -- ``critic_target``
no longer exists on ``FloQPolicy``).

``r_min``/``r_max`` (OGBench singletask sparse-reward defaults ``-1.0``/``0.0``)
are explicit constructor/CLI args, not read from dataset statistics -- a
deliberate simplification of the reference's config, per this port's design
decisions.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
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
from rl_garden.networks.critic_vector_field import integrate_returns
from rl_garden.policies.floq_policy import EncoderSharing, FloQPolicy


class FloQCore(FQLCore):
    """FloQ's flow-matching critic on top of ``FQLCore``. See module docstring."""

    def _init_floq_params(
        self,
        *,
        r_min: float = -1.0,
        r_max: float = 0.0,
        flow_num_ensembles: int = 2,
        noise_samples: int = 8,
        noise_coverage: float = 0.1,
        critic_flow_steps: int = 8,
        train_at_zero_only: bool = False,
        embed_time: bool = True,
        time_embed_dim: int = 64,
        use_prob_embed: bool = True,
        num_bins: int = 51,
        sigma: float = 16.0,
        reward_offset: float = 0.01,
        critic_flow_net_arch: Optional[Sequence[int]] = None,
    ) -> None:
        if flow_num_ensembles < 1:
            raise ValueError(f"flow_num_ensembles must be >= 1, got {flow_num_ensembles}.")
        if noise_samples < 1:
            raise ValueError(f"noise_samples must be >= 1, got {noise_samples}.")
        if critic_flow_steps < 1:
            raise ValueError(f"critic_flow_steps must be >= 1, got {critic_flow_steps}.")
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}.")

        self.r_min = r_min
        self.r_max = r_max
        self.flow_num_ensembles = flow_num_ensembles
        self.noise_samples = noise_samples
        self.noise_coverage = noise_coverage
        self.critic_flow_steps = critic_flow_steps
        self.train_at_zero_only = train_at_zero_only
        self.embed_time = embed_time
        self.time_embed_dim = time_embed_dim
        self.use_prob_embed = use_prob_embed
        self.num_bins = num_bins
        self.sigma = sigma
        self.reward_offset = reward_offset
        self.critic_flow_net_arch = list(
            critic_flow_net_arch if critic_flow_net_arch is not None else self.net_arch
        )

        self.q_min = (r_min - reward_offset) / (1.0 - self.gamma)
        self.q_max = (r_max + reward_offset) / (1.0 - self.gamma)
        self.noise_min = noise_coverage * (r_min / (1.0 - self.gamma))
        self.noise_max = noise_coverage * (r_max / (1.0 - self.gamma))

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "r_min": self.r_min,
            "r_max": self.r_max,
            "flow_num_ensembles": self.flow_num_ensembles,
            "noise_samples": self.noise_samples,
            "noise_coverage": self.noise_coverage,
            "critic_flow_steps": self.critic_flow_steps,
            "train_at_zero_only": self.train_at_zero_only,
            "embed_time": self.embed_time,
            "time_embed_dim": self.time_embed_dim,
            "use_prob_embed": self.use_prob_embed,
            "num_bins": self.num_bins,
            "sigma": self.sigma,
            "reward_offset": self.reward_offset,
            "critic_flow_net_arch": self.critic_flow_net_arch,
        }

    def _setup_model(self) -> None:
        features_extractor = self._build_features_extractor()
        if self.encoder_sharing == "separate":
            actor_bc_flow_encoder = self._build_features_extractor()
            actor_onestep_flow_encoder = self._build_features_extractor()
        else:
            actor_bc_flow_encoder = None
            actor_onestep_flow_encoder = None
        self.policy = FloQPolicy(
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
            flow_num_ensembles=self.flow_num_ensembles,
            embed_time=self.embed_time,
            time_embed_dim=self.time_embed_dim,
            use_prob_embed=self.use_prob_embed,
            q_min=self.q_min,
            q_max=self.q_max,
            num_bins=self.num_bins,
            sigma=self.sigma,
            critic_flow_net_arch=self.critic_flow_net_arch,
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
        repeat = self.noise_samples
        device, dtype = obs_features.device, obs_features.dtype

        with torch.no_grad():
            next_features_critic = self.policy.extract_features(data.next_obs)
            if self.encoder_sharing == "separate":
                next_features_actor = self.policy.extract_actor_onestep_features(
                    data.next_obs
                )
            else:
                next_features_actor = next_features_critic

            # Repeat B->B*R before sampling the next action, not after: the
            # reference draws one `next_action` per (batch, noise-sample) pair
            # (`sample_actions` on the already-repeated `next_observations`,
            # floq.py:41-42), not one action per batch element broadcast to
            # every noise sample -- repeating a single sampled action here
            # would collapse the target's Monte Carlo average over next
            # actions down to an average over noise ratios only.
            next_features_r = next_features_critic.repeat_interleave(repeat, dim=0)
            next_features_actor_r = next_features_actor.repeat_interleave(repeat, dim=0)
            next_noise = self.policy.sample_noise(
                batch_size * repeat, device=device, dtype=dtype
            )
            next_action_r = self.policy.actor_onestep_flow(next_features_actor_r, next_noise)
            next_action_r = next_action_r.clamp(
                self.policy.action_low, self.policy.action_high
            )

            next_noise_ratios = torch.rand(
                batch_size * repeat, 1, device=device, dtype=dtype
            )
            next_returns = integrate_returns(
                self.policy.floq_target,
                next_features_r,
                next_action_r,
                next_noise_ratios,
                noise_min=self.noise_min,
                noise_max=self.noise_max,
                steps=self.critic_flow_steps,
            )  # (E, B*R, 1)
            next_q = self._aggregate_target_q(next_returns).reshape(batch_size, repeat)
            target_returns = (
                data.rewards.unsqueeze(-1)
                + self.gamma * (1.0 - data.dones.unsqueeze(-1)) * next_q
            ).mean(-1)  # (B,)

            # Shared by the current-returns unroll below AND the flow-matching
            # x_0 outside this no_grad block -- this sharing is intentional,
            # not a coincidence (see module docstring).
            noise_ratios = torch.rand(batch_size * repeat, 1, device=device, dtype=dtype)
            actions_r = data.actions.repeat_interleave(repeat, dim=0)

            # This current-returns unroll runs entirely inside the enclosing
            # torch.no_grad() block. Shared-encoder-mode simplification: the
            # reference routes it through a separate `target_floq_encoder`;
            # this port has only one encoder (features_extractor), so it
            # reuses the critic's own features in place of that target encoder.
            current_features_r = obs_features.repeat_interleave(repeat, dim=0)
            current_returns = integrate_returns(
                self.policy.floq,
                current_features_r,
                actions_r,
                noise_ratios,
                noise_min=self.noise_min,
                noise_max=self.noise_max,
                steps=self.critic_flow_steps,
            )  # (E, B*R, 1)
            current_returns = current_returns.reshape(
                self.flow_num_ensembles, batch_size, repeat
            )
            mean_current_returns = current_returns.mean(dim=-1).mean(dim=0)  # (B,)

        x_0 = noise_ratios * self.noise_max + (1.0 - noise_ratios) * self.noise_min  # (B*R,1)
        x_1 = (
            target_returns.reshape(1, batch_size, 1)
            .repeat(self.flow_num_ensembles, 1, repeat)
            .reshape(self.flow_num_ensembles, batch_size * repeat, 1)
        )  # (E, B*R, 1), no grad (target_returns computed under no_grad above)

        if self.train_at_zero_only:
            t = torch.zeros(batch_size * repeat, 1, device=device, dtype=dtype)
        else:
            t = torch.rand(batch_size * repeat, 1, device=device, dtype=dtype)

        x_0_e = x_0.unsqueeze(0)
        t_e = t.unsqueeze(0)
        x_t = (1.0 - t_e) * x_0_e + t_e * x_1  # (E, B*R, 1)
        vel_target = x_1 - x_0_e  # (E, B*R, 1)

        obs_features_r = obs_features.repeat_interleave(repeat, dim=0)
        pred = self.policy.floq(obs_features_r, actions_r, x_t, t)  # (E, B*R, 1)

        pred = pred.reshape(self.flow_num_ensembles, batch_size, repeat)
        vel_target = vel_target.reshape(self.flow_num_ensembles, batch_size, repeat)
        floq_loss = ((pred - vel_target) ** 2).sum(dim=-1)  # (E, B)

        q_all = self.policy.q_values_all(obs_features, data.actions, target=False)
        distilled_loss = self._critic_loss(q_all, mean_current_returns.unsqueeze(-1))

        critic_loss = floq_loss.mean() + distilled_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
        self.critic_optimizer.step()
        if self._lr_schedulers[0] is not None:
            self._lr_schedulers[0].step()

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "floq_loss": float(floq_loss.mean().detach().item()),
            "distilled_critic_loss": float(distilled_loss.detach().item()),
        }

    def _update_targets(self) -> None:
        polyak_update(
            self.policy.floq.parameters(),
            self.policy.floq_target.parameters(),
            self.tau,
        )


class FloQ(FloQCore, OfflineRLAlgorithm):
    """Offline FloQ: flow-matching TD critic + plain distilled critic +
    two-network flow-matching actor."""

    _compatible_checkpoint_algorithms = ("FloQ",)

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
        r_min: float = -1.0,
        r_max: float = 0.0,
        flow_num_ensembles: int = 2,
        noise_samples: int = 8,
        noise_coverage: float = 0.1,
        critic_flow_steps: int = 8,
        train_at_zero_only: bool = False,
        embed_time: bool = True,
        time_embed_dim: int = 64,
        use_prob_embed: bool = True,
        num_bins: int = 51,
        sigma: float = 16.0,
        reward_offset: float = 0.01,
        critic_flow_net_arch: Optional[Sequence[int]] = None,
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
        self._init_floq_params(
            r_min=r_min,
            r_max=r_max,
            flow_num_ensembles=flow_num_ensembles,
            noise_samples=noise_samples,
            noise_coverage=noise_coverage,
            critic_flow_steps=critic_flow_steps,
            train_at_zero_only=train_at_zero_only,
            embed_time=embed_time,
            time_embed_dim=time_embed_dim,
            use_prob_embed=use_prob_embed,
            num_bins=num_bins,
            sigma=sigma,
            reward_offset=reward_offset,
            critic_flow_net_arch=critic_flow_net_arch,
        )

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, (spaces.Box, spaces.Dict)):
            raise TypeError(
                f"FloQ supports only Box or Dict observation spaces, got {type(obs_space)}"
            )

        self._setup_model()


class _FloQRolloutTrainingShell(Off2OnReplayMixin, FloQCore, OffPolicyAlgorithm):
    """Internal rollout/eval shell wiring ``FloQCore`` into ``OffPolicyAlgorithm``.

    .. warning::
       **Do not instantiate this class directly.** Use :class:`Off2OnFloQ`.
       Mirrors ``_ACFQLRolloutTrainingShell``'s precedent for this internal
       extension point, minus chunking (FloQ has no action-chunking).
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
        r_min: float = -1.0,
        r_max: float = 0.0,
        flow_num_ensembles: int = 2,
        noise_samples: int = 8,
        noise_coverage: float = 0.1,
        critic_flow_steps: int = 8,
        train_at_zero_only: bool = False,
        embed_time: bool = True,
        time_embed_dim: int = 64,
        use_prob_embed: bool = True,
        num_bins: int = 51,
        sigma: float = 16.0,
        reward_offset: float = 0.01,
        critic_flow_net_arch: Optional[Sequence[int]] = None,
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
        self._init_floq_params(
            r_min=r_min,
            r_max=r_max,
            flow_num_ensembles=flow_num_ensembles,
            noise_samples=noise_samples,
            noise_coverage=noise_coverage,
            critic_flow_steps=critic_flow_steps,
            train_at_zero_only=train_at_zero_only,
            embed_time=embed_time,
            time_embed_dim=time_embed_dim,
            use_prob_embed=use_prob_embed,
            num_bins=num_bins,
            sigma=sigma,
            reward_offset=reward_offset,
            critic_flow_net_arch=critic_flow_net_arch,
        )
        self._init_off2on_params(offline_sampling=offline_sampling)
        self._setup_model()


class Off2OnFloQ(_FloQRolloutTrainingShell):
    """Offline-to-online FloQ (ACFQL's off2on pattern, no action chunking).
    See module docstring."""

    _compatible_checkpoint_algorithms = ("Off2OnFloQ", "FloQ")
