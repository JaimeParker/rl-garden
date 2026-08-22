"""SPOT: TD3-BC with a VAE-based "support constraint" replacing the BC term.

Ported from ``3rd_party/CORL/algorithms/finetune/spot.py``
(arXiv:2202.06239). A genuine literature-subtyping relationship to the
already-shipped ``TD3BC`` (``rl_garden/algorithms/td3_bc.py``) -- SPOT's own
paper frames itself as a refinement of the TD3+BC lineage that replaces the
BC-MSE regularizer with an explicit behavior-density estimate -- so
``SPOTCore(TD3BCCore)`` reuses that class's ``_critic_loss`` static method
and ``policy_freq``-delayed actor/target-update structure directly.
``TD3BCCore.train()`` has no hook seam for swapping the actor regularizer
(monolithic, like ``ReBRACCore``'s parent), so ``train()`` below is a full
override, matching the precedent ``ReBRACCore.train()`` already set: one
method, shared unmodified by both the offline class and the off2on rollout
shell (the "default template" per ``.agents/rules/adding-algorithm.md``,
just not literally calling ``super().train()``).

Formulas verified against ``spot.py`` directly:

- **Critic** (``SPOT.train``, ``spot.py:586-614``): plain TD3 target, no BC
  term -- identical to ``TD3BCCore``'s own critic block, reused verbatim via
  ``TD3BCCore._critic_loss``.
- **Actor** (``spot.py:616-646``), delayed by ``policy_freq``: replaces
  TD3BC's ``bc_loss = F.mse_loss(pi_action, data.actions)`` with a VAE-based
  "support constraint" -- the negative ELBO (or IWAE bound, if
  ``iwae=True``) of the actor's own sampled action under a frozen,
  pretrained-offline ``BehaviorVAE``:
  ``norm_q = 1/|Q(s,pi(s))|.mean().detach()`` (no ``alpha`` numerator --
  unlike ``TD3BCCore``'s ``lmbda = alpha/|q|.mean()``), ``actor_loss =
  -norm_q*Q(s,pi(s)).mean() + lambd*neg_log_beta.mean()``.
- **VAE pretraining** (``SPOT.vae_train``, ``spot.py:556-575``): a one-time
  offline phase run *before* the main TD3 loop starts (see ``pretrain_vae``
  below), using its own step counter, not ``self._global_update``.
- **Online switch** (``spot.py:814-828``): CORL resets its three optimizers
  (two critics + actor) and swaps ``discount`` to ``online_discount``.
  rl-garden's ``EnsembleQCritic`` is one module with one optimizer, so only
  two optimizers need rebuilding here; see
  ``_SPOTRolloutTrainingShell._apply_online_regularizer_override``.
- **``lambd`` cooling** (``spot.py:627-632``): ``online_it``/``max_online_steps``
  are **gradient-step** counts in CORL (``trainer.online_it`` increments once
  per ``train()`` call), matching rl-garden's ``self._global_update`` --
  see ``_current_lambd``.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.algorithms.off2on import Off2OnReplayMixin
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.td3_bc import TD3BCCore
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.common.utils import polyak_update
from rl_garden.networks import KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.policies.spot_policy import SPOTPolicy


class SPOTCore(TD3BCCore):
    """Shared SPOT loss/network logic. See module docstring."""

    def _init_spot_params(
        self,
        *,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        net_arch: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        vae_lr: float = 1e-3,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
        vae_iterations: int = 100_000,
        beta: float = 0.5,
        lambd: float = 1.0,
        num_samples: int = 1,
        iwae: bool = False,
        lambd_cool: bool = False,
        lambd_end: float = 0.2,
        expl_noise: float = 0.1,
    ) -> None:
        # TD3BCCore._init_td3bc_params owns tau/lrs/net_arch/n_critics/layer
        # norms/etc.; its `alpha` field is unused here (SPOT's actor loss has
        # no BC-MSE term to weight) -- left at its default, never read.
        self._init_td3bc_params(
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
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            net_arch=net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        if vae_iterations < 0:
            raise ValueError(f"vae_iterations must be >= 0, got {vae_iterations}.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self.vae_lr = vae_lr
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_iterations = vae_iterations
        self.beta = beta
        self.lambd = lambd
        self.num_samples = num_samples
        self.iwae = iwae
        self.lambd_cool = lambd_cool
        self.lambd_end = lambd_end
        self.expl_noise = expl_noise
        self._vae_pretrained = False

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = super()._checkpoint_metadata()
        meta.pop("alpha", None)
        return {
            **meta,
            "vae_lr": self.vae_lr,
            "vae_hidden_dim": self.vae_hidden_dim,
            "vae_latent_dim": self.vae_latent_dim,
            "vae_iterations": self.vae_iterations,
            "beta": self.beta,
            "lambd": self.lambd,
            "num_samples": self.num_samples,
            "iwae": self.iwae,
            "lambd_cool": self.lambd_cool,
            "lambd_end": self.lambd_end,
            "expl_noise": self.expl_noise,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        state = super()._extra_checkpoint_state()
        state["vae_pretrained"] = self._vae_pretrained
        return state

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        self._vae_pretrained = bool(state.get("vae_pretrained", False))
        if self._vae_pretrained:
            # policy.load_state_dict() (called separately by load()) restores
            # parameter values but not requires_grad -- reapply the freeze.
            for p in self.policy.vae.parameters():
                p.requires_grad_(False)

    def _setup_model(self) -> None:
        # Cannot call super()._setup_model(): TD3BCCore hardcodes TD3BCPolicy.
        features_extractor = self._build_features_extractor()
        self.policy = SPOTPolicy(
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
            actor_dropout_rate=self.actor_dropout_rate,
            critic_dropout_rate=self.critic_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            vae_hidden_dim=self.vae_hidden_dim,
            vae_latent_dim=self.vae_latent_dim,
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
        self.vae_optimizer = make_optimizer(
            list(self.policy.vae_parameters()), lr=self.vae_lr
        )
        self.replay_buffer = self._build_replay_buffer()
        self._build_lr_schedulers()

        low = torch.as_tensor(
            self.env.single_action_space.low, dtype=torch.float32, device=self.device
        )
        high = torch.as_tensor(
            self.env.single_action_space.high, dtype=torch.float32, device=self.device
        )
        self._action_low = low
        self._action_high = high

    def _build_lr_schedulers(self) -> None:
        # VAE has no LR schedule (fixed vae_lr, matching CORL) -- keep
        # indices [0]=critic, [1]=actor unchanged from TD3BCCore's convention.
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

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("critic_optimizer", "actor_optimizer", "vae_optimizer")

    def pretrain_vae(self) -> dict[str, float]:
        """Pretrain ``self.policy.vae`` on the replay buffer, then freeze it.

        No-op (with a one-line note) if already pretrained -- e.g. a second
        call after a checkpoint load that already restored a pretrained VAE.
        Uses its own step counter, not ``self._global_update`` (which stays
        reserved for the main TD3 loop -- CORL's own shared ``total_it``
        counter across VAE pretraining and the main loop has no faithfulness
        value worth preserving here).
        """
        if self._vae_pretrained:
            return {}
        vae = self.policy.vae
        last_loss = {}
        for _ in range(self.vae_iterations):
            data = self.replay_buffer.sample(self.batch_size)
            features = self.policy.extract_features(data.obs)
            losses = vae.loss(features, data.actions, self.beta)

            self.vae_optimizer.zero_grad(set_to_none=True)
            losses["vae_loss"].backward()
            self.vae_optimizer.step()
            last_loss = {
                "vae/reconstruction_loss": float(losses["recon_loss"].detach().item()),
                "vae/kl_loss": float(losses["kl_loss"].detach().item()),
                "vae/vae_loss": float(losses["vae_loss"].detach().item()),
            }

        for p in vae.parameters():
            p.requires_grad_(False)
        vae.eval()
        self._vae_pretrained = True
        if self.logger is not None:
            for key, value in last_loss.items():
                self.logger.add_summary(key, value)
        return last_loss

    def _current_lambd(self) -> float:
        """SPOT's ``lambd`` regularizer weight. Offline: constant. The off2on
        rollout shell overrides this to add cooling during online fine-tuning.
        """
        return self.lambd

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        counts: dict[str, int] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch(self.batch_size)

            obs_features = self.policy.extract_features(data.obs)
            with torch.no_grad():
                next_features = self.policy.extract_features(data.next_obs)
                noise = (torch.randn_like(data.actions) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                next_action = (self.policy.actor_target(next_features) + noise).clamp(
                    self._action_low, self._action_high
                )
                target_q_all = self.policy.q_values_all(
                    next_features, next_action, target=True
                )
                target_q = data.rewards.unsqueeze(-1) + self.gamma * (
                    1.0 - data.dones.unsqueeze(-1)
                ) * target_q_all.min(dim=0).values

            q_all = self.policy.q_values_all(obs_features, data.actions, target=False)
            critic_loss = self._critic_loss(q_all, target_q)

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
            self.critic_optimizer.step()
            if self._lr_schedulers[0] is not None:
                self._lr_schedulers[0].step()

            metrics_sum["critic_loss"] = metrics_sum.get("critic_loss", 0.0) + float(
                critic_loss.detach().item()
            )
            counts["critic_loss"] = counts.get("critic_loss", 0) + 1

            if self._global_update % self.policy_freq == 0:
                features_detached = obs_features.detach()
                pi_action = self.policy.actor(features_detached)
                q_pi = self.policy.q_values_all(
                    features_detached, pi_action, target=False
                )[0]
                # NOTE: no alpha numerator here (unlike TD3BCCore's
                # `lmbda = self.alpha / q_pi.abs().mean()`) -- spot.py:634.
                norm_q = (1.0 / q_pi.abs().mean()).detach()

                if self.iwae:
                    neg_log_beta = -self.policy.vae.iwae_ll(
                        features_detached, pi_action, self.beta, self.num_samples
                    )
                else:
                    neg_log_beta = self.policy.vae.elbo_loss(
                        features_detached, pi_action, self.beta, self.num_samples
                    )
                lambd = self._current_lambd()
                actor_loss = -norm_q * q_pi.mean() + lambd * neg_log_beta.mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self._clip_grad_norm(self.policy.actor_parameters())
                self.actor_optimizer.step()
                if self._lr_schedulers[1] is not None:
                    self._lr_schedulers[1].step()

                polyak_update(
                    self.policy.critic.parameters(),
                    self.policy.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.policy.actor.parameters(),
                    self.policy.actor_target.parameters(),
                    self.tau,
                )

                for key, value in (
                    ("actor_loss", float(actor_loss.detach().item())),
                    ("neg_log_beta", float(neg_log_beta.detach().mean().item())),
                    ("lambd", float(lambd)),
                ):
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1

        if not compute_info:
            return {}
        return {key: metrics_sum[key] / counts[key] for key in metrics_sum}


class SPOT(SPOTCore, OfflineRLAlgorithm):
    """Offline SPOT: TD3-BC + a VAE-based support constraint. See module docstring."""

    _compatible_checkpoint_algorithms = ("SPOT",)

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
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        net_arch: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        vae_lr: float = 1e-3,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
        vae_iterations: int = 100_000,
        beta: float = 0.5,
        lambd: float = 1.0,
        num_samples: int = 1,
        iwae: bool = False,
        lambd_cool: bool = False,
        lambd_end: float = 0.2,
        expl_noise: float = 0.1,
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
        self._init_spot_params(
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
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            net_arch=net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            vae_lr=vae_lr,
            vae_hidden_dim=vae_hidden_dim,
            vae_latent_dim=vae_latent_dim,
            vae_iterations=vae_iterations,
            beta=beta,
            lambd=lambd,
            num_samples=num_samples,
            iwae=iwae,
            lambd_cool=lambd_cool,
            lambd_end=lambd_end,
            expl_noise=expl_noise,
        )

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                f"SPOT supports only Box observation spaces, got {type(obs_space)}"
            )

        self._setup_model()


class _SPOTRolloutTrainingShell(Off2OnReplayMixin, SPOTCore, OffPolicyAlgorithm):
    """Internal rollout/eval shell that wires ``SPOTCore`` into ``OffPolicyAlgorithm``.

    Generic offline->online transition mechanics (replay-buffer switching,
    mixed-batch sampling, checkpoint/probe/logging plumbing) are inherited
    from ``Off2OnReplayMixin``. Unlike IQL/AWAC, SPOT overrides
    ``_apply_online_regularizer_override`` (the second algorithm to do so,
    after Cal-QL) to reproduce CORL's online-switch optimizer reset and
    discount swap, and ``_rollout_action`` to add TD3-style exploration noise
    (``SPOTPolicy.predict()`` is always deterministic, like ``TD3BCPolicy``'s,
    so the default rollout path -- which calls ``predict()`` -- would
    otherwise collect data with a fully greedy actor; see ``DDPG._rollout_action``
    for the precedent this mirrors).

    .. warning::
       **Do not instantiate this class directly.** It exists only to back
       :class:`~rl_garden.algorithms.Off2OnSPOT`. For standalone offline SPOT
       pretraining use :class:`SPOT`. The shape and arguments of this shell
       may change without notice.
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
        bootstrap_at_done: str = "always",
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: ScheduleType = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        net_arch: Optional[Sequence[int]] = None,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        vae_lr: float = 1e-3,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
        vae_iterations: int = 100_000,
        beta: float = 0.5,
        lambd: float = 1.0,
        num_samples: int = 1,
        iwae: bool = False,
        lambd_cool: bool = False,
        lambd_end: float = 0.2,
        expl_noise: float = 0.1,
        online_discount: float = 0.995,
        max_online_updates: int = 1_000_000,
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
        self._is_dict_obs = False  # SPOT is Box-only, no image support.
        self.online_discount = online_discount
        self.max_online_updates = max_online_updates
        self._spot_online_update_start: Optional[int] = None
        super().__init__(
            env=env,
            eval_env=eval_env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
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
            initial_training_phase=initial_training_phase,
        )
        self._init_spot_params(
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
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
            net_arch=net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            vae_lr=vae_lr,
            vae_hidden_dim=vae_hidden_dim,
            vae_latent_dim=vae_latent_dim,
            vae_iterations=vae_iterations,
            beta=beta,
            lambd=lambd,
            num_samples=num_samples,
            iwae=iwae,
            lambd_cool=lambd_cool,
            lambd_end=lambd_end,
            expl_noise=expl_noise,
        )
        self._setup_model()
        self._init_off2on_params(offline_sampling=offline_sampling)

    def _current_lambd(self) -> float:
        if not self.lambd_cool or self._spot_online_update_start is None:
            return self.lambd
        online_it = self._global_update - self._spot_online_update_start
        return self.lambd * max(
            self.lambd_end, 1.0 - online_it / self.max_online_updates
        )

    def _apply_online_regularizer_override(self, online_replay_mode: str) -> None:
        del online_replay_mode
        # Reproduce spot.py:819-828's "Resetting optimizers" + discount swap
        # at the offline->online transition. Rebuild the LR schedulers
        # together with their optimizers -- leaving the old schedulers
        # wrapping discarded optimizer objects would silently detach them.
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
        self._build_lr_schedulers()
        self.gamma = self.online_discount
        self._spot_online_update_start = self._global_update

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        if self._active_initial_training_phase() is not None or not learning_has_started:
            return super()._rollout_action(obs, learning_has_started)
        with torch.no_grad():
            features = self.policy.extract_features(self._obs_to_policy_device(obs))
            action = self.policy.actor.deterministic_action(features)
            noise = (torch.randn_like(action) * self.expl_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            action = (action + noise).clamp(self._action_low, self._action_high)
        return action, action, None
