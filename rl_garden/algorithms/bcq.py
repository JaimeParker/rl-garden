"""BCQ: Batch-Constrained deep Q-learning (Fujimoto et al. 2019, arXiv:1812.02900).

Ported from ``sfujim/BCQ/continuous_BCQ/BCQ.py`` (the official reference --
CORL has no BCQ implementation to cross-check against). Pure offline, no
online fine-tuning variant in the reference. Box observations only.
Verified against a full clone of the upstream repo (``BCQ.py``, ``main.py``
read in full), not just fetched raw files.

Deliberately NOT built on ``TD3BCCore``/``SPOTCore``: BCQ has no
``policy_freq``-delayed updates (actor and both target networks update every
single gradient step), no TD3-style next-action noise, and its actor takes
``(state, action) -> perturbed action`` rather than ``state -> action`` --
there is no shared seam worth inheriting from the TD3-BC family.

Formulas verified against ``BCQ.py`` directly:

- **VAE**: trained *jointly*, every gradient step (``BCQ.py:139-147``) --
  unlike SPOT's one-time offline pretraining. There is no freeze; ``self.vae``
  stays a normal trainable submodule for the algorithm's entire lifetime
  (see ``BCQPolicy``'s docstring). ``vae_lr`` is pinned explicitly to
  ``1e-3`` here; upstream's own ``vae_optimizer = Adam(self.vae.parameters())``
  omits an explicit ``lr`` and silently relies on PyTorch's Adam default
  (also ``1e-3``) -- the research cross-check flagged this as likely an
  oversight, not a deliberate choice, so this port pins the value rather than
  reproducing the omission.
- **Critic target** (``BCQ.py:150-163``): 10 VAE-sampled next-actions per
  next-state (``next_features.repeat_interleave(10, dim=0)``), each perturbed
  by the frozen ``actor_target``, scored by both target critics, combined via
  the "soft clipped double Q-learning" mixture
  ``soft_q_lambda * min(q1, q2) + (1 - soft_q_lambda) * max(q1, q2)``
  (``soft_q_lambda`` default ``0.75``), then **maxed** (not averaged) over
  the 10 candidates per original next-state -- approximates the best
  batch-constrained next action. Named ``soft_q_lambda`` here, not ``lmbda``,
  to avoid colliding with TD3-BC's ``alpha``-normalizer local or SPOT's
  ``lambd`` regularizer weight -- three unrelated "lambda"s already exist in
  this codebase's TD3-BC-family algorithms.
- **Critic loss**: sum (not mean) of both heads' MSE against the target.
- **Actor (perturbation) loss** (``BCQ.py:173-178``): a single VAE-sampled
  action per state, perturbed by the (live) actor, scored by **q1 only**
  (not ``min(q1, q2)``, unlike TD3-BC/SPOT's actor loss) --
  ``actor_loss = -q1(state, perturbed_action).mean()``.
- **Prior sampling clip**: every ``vae.decode(z=None)`` call site in BCQ
  hardcodes ``clip=0.5`` (target's 10 candidates, actor loss's 1 candidate,
  and ``BCQPolicy.predict``'s 100 eval candidates) -- ``ConditionalVAE.decode``
  defaults to ``clip=None`` (unclipped), so every call site here passes
  ``clip=0.5`` explicitly.
- **Target updates**: plain polyak ``tau=0.005`` on both actor and critic,
  every gradient step -- no delay, no separate schedule.
- **Obs normalization**: neither research pass found normalization in
  upstream BCQ (raw states throughout). ``BCQPolicy`` still normalizes obs
  by mean/std (``ObsNormalizingMixin``) -- this is rl-garden's project-wide
  offline-training convention (also applied to TD3-BC/SPOT), not something
  ported from BCQ itself.
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
from rl_garden.policies.bcq_policy import BCQPolicy

_NUM_TARGET_CANDIDATES = 10


class BCQCore:
    """Shared BCQ loss/network logic. See module docstring."""

    def _init_bcq_params(
        self,
        *,
        tau: float = 0.005,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        vae_lr: float = 1e-3,
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
        phi: float = 0.05,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
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

        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.vae_lr = vae_lr
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
        self.phi = phi
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.beta = beta
        self.soft_q_lambda = soft_q_lambda

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("critic_optimizer", "actor_optimizer", "vae_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "tau": self.tau,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "vae_lr": self.vae_lr,
            "weight_decay": self.weight_decay,
            "use_adamw": self.use_adamw,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "grad_clip_norm": self.grad_clip_norm,
            "net_arch": self.net_arch,
            "phi": self.phi,
            "vae_hidden_dim": self.vae_hidden_dim,
            "vae_latent_dim": self.vae_latent_dim,
            "beta": self.beta,
            "soft_q_lambda": self.soft_q_lambda,
            "num_target_candidates": _NUM_TARGET_CANDIDATES,
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

    def _default_features_extractor_class(self) -> type[BaseFeaturesExtractor]:
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Box):
            return FlattenExtractor
        raise TypeError(
            "BCQ only supports Box observation spaces, got " + str(type(obs_space))
        )

    def _build_features_extractor(self) -> BaseFeaturesExtractor:
        cls = self._default_features_extractor_class()
        return cls(observation_space=self.env.single_observation_space)

    def _build_replay_buffer(self) -> TensorReplayBuffer:
        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                "BCQ only supports Box observation spaces, got " + str(type(obs_space))
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
        self.policy = BCQPolicy(
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
            features_detached = obs_features.detach()

            vae_losses = self.policy.vae.loss(features_detached, data.actions, self.beta)
            self.vae_optimizer.zero_grad(set_to_none=True)
            vae_losses["vae_loss"].backward()
            self.vae_optimizer.step()

            with torch.no_grad():
                next_features = self.policy.extract_features(data.next_obs)
                tiled_next_features = next_features.repeat_interleave(
                    _NUM_TARGET_CANDIDATES, dim=0
                )
                sampled_next_actions = self.policy.vae.decode(tiled_next_features, clip=0.5)
                perturbed_next_actions = self.policy.actor_target(
                    tiled_next_features, sampled_next_actions
                )
                q1_t, q2_t = self.policy.q_values(
                    tiled_next_features, perturbed_next_actions, target=True
                )
                mixed_q = self.soft_q_lambda * torch.min(q1_t, q2_t) + (
                    1.0 - self.soft_q_lambda
                ) * torch.max(q1_t, q2_t)
                mixed_q = mixed_q.reshape(-1, _NUM_TARGET_CANDIDATES).max(
                    dim=1, keepdim=True
                ).values
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

            sampled_actions = self.policy.vae.decode(features_detached, clip=0.5)
            perturbed_actions = self.policy.actor(features_detached, sampled_actions)
            q1_pi, _ = self.policy.q_values(features_detached, perturbed_actions, target=False)
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
                self.policy.actor.parameters(), self.policy.actor_target.parameters(), self.tau
            )

            for key, value in (
                ("critic_loss", float(critic_loss.detach().item())),
                ("actor_loss", float(actor_loss.detach().item())),
                ("vae_loss", float(vae_losses["vae_loss"].detach().item())),
                ("vae_recon_loss", float(vae_losses["recon_loss"].detach().item())),
                ("vae_kl_loss", float(vae_losses["kl_loss"].detach().item())),
            ):
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

        if not compute_info:
            return {}
        return {key: metrics_sum[key] / counts[key] for key in metrics_sum}


class BCQ(BCQCore, OfflineRLAlgorithm):
    """Offline BCQ: VAE-constrained perturbation actor + soft double-Q critic. See module docstring."""

    _compatible_checkpoint_algorithms = ("BCQ",)

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
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        vae_lr: float = 1e-3,
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
        phi: float = 0.05,
        vae_hidden_dim: int = 750,
        vae_latent_dim: Optional[int] = None,
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
        self._init_bcq_params(
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            vae_lr=vae_lr,
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
            phi=phi,
            vae_hidden_dim=vae_hidden_dim,
            vae_latent_dim=vae_latent_dim,
            beta=beta,
            soft_q_lambda=soft_q_lambda,
        )

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                f"BCQ supports only Box observation spaces, got {type(obs_space)}"
            )

        self._setup_model()
