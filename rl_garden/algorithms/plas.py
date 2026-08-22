"""PLAS: latent-space policy for offline RL (Zhou et al. 2020, arXiv:2011.07213).

Ported from ``Wenxuan-Zhou/PLAS/algos.py`` (the official reference -- CORL
has no PLAS implementation to cross-check against). Pure offline, no online
fine-tuning variant in the reference. Box observations only. Verified
against a full clone of the upstream repo (``algos.py``, ``main.py``,
``README.md`` read in full), not just fetched raw files.

Deliberately NOT built on ``BCQCore``: despite the shared VAE and soft
double-Q target formula (PLAS's own code reuses BCQ's target-mixture
verbatim), PLAS's VAE is pretrained-then-frozen rather than jointly trained,
its actor is single-shot/deterministic rather than a 10/100-candidate
search, and its actor input shape differs (state -> latent, not
(state, action) -> perturbed action). There is no shared seam worth
inheriting; see ``BCQCore``'s module docstring for the same reasoning
applied to its own choice not to subclass ``TD3BCCore``.

Formulas verified against ``algos.py`` directly:

- **VAE pretraining** (``main.py:105-117``): trained once, offline, for
  ``vae_iterations`` steps *before* the main actor/critic loop starts, then
  frozen (``requires_grad_(False)`` + ``eval()``) -- mirrors ``SPOTCore``'s
  ``pretrain_vae()`` pattern exactly (not shared via a common base class,
  per ``.agents/rules/adding-algorithm.md``'s "don't retrofit a working
  class for a hypothetical shared seam" guidance -- duplicating this ~20-line
  method is cheaper than coupling SPOT and PLAS together). Note: upstream's
  own ``VAEModule.train()`` never explicitly freezes the VAE after
  pretraining (a wart flagged by the official-source cross-check -- actor
  gradients would silently accumulate into unused VAE parameter ``.grad``
  fields with nothing consuming them); this port fixes that by freezing
  explicitly, the same correct behavior SPOT already has. Also note:
  ``pretrain_vae()`` samples ``self.batch_size``-sized batches, where
  upstream hardcodes ``100`` inside its own VAE trainer -- the same
  divergence SPOT's ``pretrain_vae()`` already has, kept for consistency
  rather than adding a separate ``vae_batch_size`` field.
- **Latent actor** (``algos.py`` ``Latent`` class): outputs a
  ``max_latent_action``-bounded vector in the VAE's latent space (not a raw
  action); ``vae.decode(state, z=latent)`` turns it into an action.
- **Critic target**: the identical "soft clipped double-Q" mixture BCQ uses
  (``soft_q_lambda * min(q1, q2) + (1 - soft_q_lambda) * max(q1, q2)``,
  default ``0.75``) -- but evaluated on a single actor-produced next-action,
  **no** multi-candidate-and-max step (PLAS's actor is deterministic, so
  there is exactly one candidate per state, unlike BCQ's 10-sample target).
- **Actor loss**: ``-critic.q1(state, action_from_latent(state)).mean()``,
  q1-only (live critic), gradients flowing through the frozen-but-
  differentiable VAE decoder into the latent actor's parameters only.
- **"-P" variant** (``use_perturbation=True``, default ``False``): matches
  upstream's own default -- ``main.py``'s ``--algo_name`` defaults to
  ``"Latent"``, not ``"LatentPerturbation"``, so no perturbation network
  exists at all unless -P is explicitly selected. (``--phi`` separately
  defaults to ``0.`` in the CLI, but that is only a safety net for running
  ``--algo_name LatentPerturbation`` *without* an explicit ``--phi`` --
  at ``phi=0`` the perturbation stage's own output is identically zero, so
  it degrades to a no-op rather than the mechanism that turns -P off by
  default.) When enabled, adds a second-stage ``PerturbationActor`` after
  the VAE decode, reusing BCQ's exact perturbation-network class (confirmed
  against the cloned upstream source: PLAS-P's second stage is literally
  BCQ's ``Actor`` class reused inline as ``ActorPerturbation``'s ``l4-l6``).
  ``phi=0.05`` here matches the class-level default and the README's
  documented -P invocation (``--phi 0.05``), not the CLI's inert blanket
  default. The paper's headline numbers on several tasks use this variant,
  so it is implemented here rather than silently omitted -- see
  ``PLASPolicy``'s docstring for how it composes with the latent actor.
- **Target updates**: plain polyak ``tau=0.005`` on the critic and the
  actor (latent actor, plus the perturbation net when present) every
  gradient step -- no delay.
- **Obs normalization**: neither research pass found normalization in
  upstream PLAS (raw states throughout). ``PLASPolicy`` still normalizes
  obs by mean/std (``ObsNormalizingMixin``), same rl-garden-wide convention
  as ``BCQPolicy``/``TD3BCPolicy``/``SPOTPolicy`` -- not ported from PLAS.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.networks.mlp import KernelInit
from rl_garden.policies.plas_policy import PLASPolicy


class PLASCore:
    """Shared PLAS loss/network logic. See module docstring."""

    def _init_plas_params(
        self,
        *,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        net_arch: Optional[Sequence[int]] = None,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        max_latent_action: float = 2.0,
        use_perturbation: bool = False,
        phi: float = 0.05,
        vae_lr: float = 1e-4,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
        vae_iterations: int = 500_000,
        beta: float = 0.5,
        soft_q_lambda: float = 0.75,
    ) -> None:
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {tau}.")
        if not (0.0 <= soft_q_lambda <= 1.0):
            raise ValueError(f"soft_q_lambda must be in [0, 1], got {soft_q_lambda}.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )
        if vae_iterations < 0:
            raise ValueError(f"vae_iterations must be >= 0, got {vae_iterations}.")

        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip_norm = grad_clip_norm
        self.net_arch: list[int] = list(net_arch) if net_arch is not None else [400, 300]
        self.actor_use_layer_norm = actor_use_layer_norm
        self.critic_use_layer_norm = critic_use_layer_norm
        self.actor_use_group_norm = actor_use_group_norm
        self.critic_use_group_norm = critic_use_group_norm
        self.num_groups = num_groups
        self.actor_dropout_rate = actor_dropout_rate
        self.critic_dropout_rate = critic_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.max_latent_action = max_latent_action
        self.use_perturbation = use_perturbation
        self.phi = phi
        self.vae_lr = vae_lr
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_iterations = vae_iterations
        self.beta = beta
        self.soft_q_lambda = soft_q_lambda
        self._vae_pretrained = False

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("critic_optimizer", "actor_optimizer", "vae_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "tau": self.tau,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "weight_decay": self.weight_decay,
            "use_adamw": self.use_adamw,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "grad_clip_norm": self.grad_clip_norm,
            "net_arch": self.net_arch,
            "max_latent_action": self.max_latent_action,
            "use_perturbation": self.use_perturbation,
            "phi": self.phi,
            "vae_lr": self.vae_lr,
            "vae_hidden_dim": self.vae_hidden_dim,
            "vae_latent_dim": self.vae_latent_dim,
            "vae_iterations": self.vae_iterations,
            "beta": self.beta,
            "soft_q_lambda": self.soft_q_lambda,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "vae_pretrained": self._vae_pretrained,
            "lr_scheduler_states": [
                sched.state_dict() if sched is not None else None
                for sched in self._lr_schedulers
            ],
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        self._vae_pretrained = bool(state.get("vae_pretrained", False))
        if self._vae_pretrained:
            # policy.load_state_dict() (called separately by load()) restores
            # parameter values but not requires_grad -- reapply the freeze.
            for p in self.policy.vae.parameters():
                p.requires_grad_(False)
        for sched, sched_state in zip(
            self._lr_schedulers, state.get("lr_scheduler_states", [])
        ):
            if sched is not None and sched_state is not None:
                sched.load_state_dict(sched_state)

    def _default_features_extractor_class(self) -> type[BaseFeaturesExtractor]:
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Box):
            return FlattenExtractor
        raise TypeError(
            "PLAS only supports Box observation spaces, got " + str(type(obs_space))
        )

    def _build_features_extractor(self) -> BaseFeaturesExtractor:
        cls = self._default_features_extractor_class()
        return cls(observation_space=self.env.single_observation_space)

    def _build_replay_buffer(self) -> TensorReplayBuffer:
        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                "PLAS only supports Box observation spaces, got " + str(type(obs_space))
            )
        return TensorReplayBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _setup_model(self) -> None:
        features_extractor = self._build_features_extractor()
        self.policy = PLASPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            net_arch=self.net_arch,
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            actor_use_group_norm=self.actor_use_group_norm,
            critic_use_group_norm=self.critic_use_group_norm,
            num_groups=self.num_groups,
            actor_dropout_rate=self.actor_dropout_rate,
            critic_dropout_rate=self.critic_dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            max_latent_action=self.max_latent_action,
            use_perturbation=self.use_perturbation,
            phi=self.phi,
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

    def fit_obs_normalizer(self) -> None:
        buf = self.replay_buffer
        obs = buf.obs[: buf.size].reshape(-1, buf.obs.shape[-1]).to(self.device)
        self.policy.fit_obs_normalizer(obs)

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

    def _clip_grad_norm(self, params) -> None:
        if self.grad_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(list(params), self.grad_clip_norm)

    def pretrain_vae(self) -> dict[str, float]:
        """Pretrain ``self.policy.vae`` on the replay buffer, then freeze it.

        No-op (returns ``{}``) if already pretrained -- e.g. a second call
        after a checkpoint load that already restored a pretrained VAE. See
        module docstring for the divergence from upstream this fixes.
        """
        if self._vae_pretrained:
            return {}
        vae = self.policy.vae
        last_loss: dict[str, float] = {}
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
                next_latent = self.policy.latent_actor_target(next_features)
                next_action = self.policy.action_from_latent(
                    next_features, next_latent, target=True
                )
                q1_t, q2_t = self.policy.q_values(next_features, next_action, target=True)
                mixed_q = self.soft_q_lambda * torch.min(q1_t, q2_t) + (
                    1.0 - self.soft_q_lambda
                ) * torch.max(q1_t, q2_t)
                target_q = data.rewards.unsqueeze(-1) + self.gamma * (
                    1.0 - data.dones.unsqueeze(-1)
                ) * mixed_q

            q1, q2 = self.policy.q_values(obs_features, data.actions, target=False)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
            self.critic_optimizer.step()
            if self._lr_schedulers[0] is not None:
                self._lr_schedulers[0].step()

            features_detached = obs_features.detach()
            latent = self.policy.latent_actor(features_detached)
            action = self.policy.action_from_latent(features_detached, latent, target=False)
            q1_pi, _ = self.policy.q_values(features_detached, action, target=False)
            actor_loss = -q1_pi.mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip_grad_norm(self.policy.actor_parameters())
            self.actor_optimizer.step()
            if self._lr_schedulers[1] is not None:
                self._lr_schedulers[1].step()

            polyak_update(
                self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau
            )
            polyak_update(
                self.policy.latent_actor.parameters(),
                self.policy.latent_actor_target.parameters(),
                self.tau,
            )
            if self.use_perturbation:
                polyak_update(
                    self.policy.perturbation.parameters(),
                    self.policy.perturbation_target.parameters(),
                    self.tau,
                )

            for key, value in (
                ("critic_loss", float(critic_loss.detach().item())),
                ("actor_loss", float(actor_loss.detach().item())),
            ):
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

        if not compute_info:
            return {}
        return {key: metrics_sum[key] / counts[key] for key in metrics_sum}


class PLAS(PLASCore, OfflineRLAlgorithm):
    """Offline PLAS: latent-space policy over a pretrained-frozen VAE. See module docstring."""

    _compatible_checkpoint_algorithms = ("PLAS",)

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
        critic_lr: float = 1e-3,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        net_arch: Optional[Sequence[int]] = None,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        max_latent_action: float = 2.0,
        use_perturbation: bool = False,
        phi: float = 0.05,
        vae_lr: float = 1e-4,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
        vae_iterations: int = 500_000,
        beta: float = 0.5,
        soft_q_lambda: float = 0.75,
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
        self._init_plas_params(
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
            net_arch=net_arch,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            max_latent_action=max_latent_action,
            use_perturbation=use_perturbation,
            phi=phi,
            vae_lr=vae_lr,
            vae_hidden_dim=vae_hidden_dim,
            vae_latent_dim=vae_latent_dim,
            vae_iterations=vae_iterations,
            beta=beta,
            soft_q_lambda=soft_q_lambda,
        )

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                f"PLAS supports only Box observation spaces, got {type(obs_space)}"
            )

        self._setup_model()
