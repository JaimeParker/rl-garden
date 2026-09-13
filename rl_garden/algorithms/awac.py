"""AWAC: advantage-weighted actor-critic.

Ported from ``3rd_party/CORL/algorithms/offline/awac.py``. ``AWACCore`` holds
the loss/network/optimizer logic shared by the pure offline ``AWAC`` (built
on ``OfflineRLAlgorithm``) and the rollout-capable
``_AWACRolloutTrainingShell`` (built on ``OffPolicyAlgorithm``, backing
``Off2OnAWAC``) -- mirrors ``IQLCore``/``IQL``/``_IQLRolloutTrainingShell`` in
``rl_garden/algorithms/iql.py``. State observations only (no images;
``_setup_model`` rejects ``schema.has_images``), matching CORL's D4RL MuJoCo
scope -- ``state_<name>`` keys and an asymmetric ``obs_groups``/
``critic_encoder_config``/``encoder_sharing="separate"`` critic are
supported (see ``AWACPolicy``).

Two deliberate deviations from every other actor-critic algorithm in
rl-garden, both faithful to CORL's actual numerics (not bugs):

* **No actor target.** The critic backup samples ``next_action`` from the
  *current* actor (``self._actor(next_states)`` in CORL), not a
  Polyak-averaged target actor. Only the twin-Q critic has a target network.
* **Unsquashed actor.** ``UnsquashedGaussianActor`` hard-clamps to the action
  bounds instead of tanh-squashing, so its actor loss is not directly
  comparable in scale to IQL's tanh-squashed AWR loss.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional, Sequence

import torch

from rl_garden.algorithms._observation import EncoderSharing
from rl_garden.algorithms.off2on import Off2OnReplayMixin
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks import KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.observations import ObservationContractError, ObsGroups
from rl_garden.policies.awac_policy import AWACPolicy


class AWACCore:
    """Shared AWAC loss/network logic."""

    def _init_awac_params(
        self,
        *,
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
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
        std_parameterization: Literal["exp", "uniform"] = "exp",
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
        image_augmentation_seed: Optional[int] = None,
    ) -> None:
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {tau}.")
        if awac_lambda <= 0:
            raise ValueError(f"awac_lambda must be positive, got {awac_lambda}.")
        if exp_adv_max <= 0:
            raise ValueError(f"exp_adv_max must be positive, got {exp_adv_max}.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )

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
        self.awac_lambda = awac_lambda
        self.exp_adv_max = exp_adv_max
        self.net_arch: list[int] = (
            list(net_arch) if net_arch is not None else [256, 256, 256]
        )
        self.n_critics = n_critics
        self.actor_use_layer_norm = actor_use_layer_norm
        self.critic_use_layer_norm = critic_use_layer_norm
        self.actor_use_group_norm = actor_use_group_norm
        self.critic_use_group_norm = critic_use_group_norm
        self.num_groups = num_groups
        self.actor_dropout_rate = actor_dropout_rate
        self.critic_dropout_rate = critic_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.std_parameterization = std_parameterization
        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self.critic_encoder_config = critic_encoder_config
        self.encoder_sharing = encoder_sharing
        self._image_augmentation_seed = image_augmentation_seed

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("critic_optimizer", "actor_optimizer")

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
            "awac_lambda": self.awac_lambda,
            "exp_adv_max": self.exp_adv_max,
            "net_arch": self.net_arch,
            "n_critics": self.n_critics,
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
            "image_augmentation_seed": self._image_augmentation_seed,
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

    def _build_replay_buffer(self):
        # obs_space is always Dict (boundary normalization is unconditional).
        return ReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _setup_model(self) -> None:
        extractor_kwargs = self._policy_extractor_kwargs(
            self.env.single_observation_space,
            augmentation_seed=self._image_augmentation_seed,
        )
        if self.observation_encoders.schema.has_images:
            raise ObservationContractError(
                "AWAC only supports state observations (no images); got image "
                f"keys {self.observation_encoders.schema.image_keys}."
            )
        self.policy = AWACPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
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
            std_parameterization=self.std_parameterization,
            **extractor_kwargs,
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

    def fit_obs_normalizer(self) -> None:
        buf = self.replay_buffer
        # buf.obs is always a DictArray (boundary normalization is
        # unconditional); hand every stored state key to both extractors --
        # each's own FlattenExtractor picks out only the key(s) it was built
        # from (see FlattenExtractor._flatten_obs), so this stays correct
        # even when obs_groups makes actor/critic state keys asymmetric.
        obs = {
            key: buf.obs[key][: buf.size].reshape(-1, *buf.obs[key].shape[2:]).to(self.device)
            for key in buf.obs.keys()
        }
        with torch.no_grad():
            self.policy.fit_obs_normalizer(self.policy.actor_extractor(obs))
            # Under encoder_sharing="separate" AWACPolicy also owns a second,
            # suffix="critic" obs-normalizer buffer pair sized to
            # critic_extractor's own output features (see
            # rl_garden/policies/awac_policy.py's docstring) -- fit it too,
            # or it stays at its __init__-time zeros/ones default (a no-op
            # normalization) forever. Dedup by identity
            # (BasePolicy.update_normalizer's convention).
            if (
                self.policy.critic_extractor is not None
                and self.policy.critic_extractor is not self.policy.actor_extractor
            ):
                self.policy.fit_obs_normalizer(
                    self.policy.critic_extractor(obs), suffix="critic"
                )

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

    def _step_schedulers(self) -> None:
        for sched in self._lr_schedulers:
            if sched is not None:
                sched.step()

    def _compute_critic_loss(self, data) -> torch.Tensor:
        critic_features = self.policy.extract_critic_features(data.obs)
        with torch.no_grad():
            next_critic_features = self.policy.extract_critic_features(data.next_obs)
            next_actor_features = self.policy.extract_actor_features(data.next_obs)
            # Faithful to CORL: next_action is sampled from the CURRENT actor,
            # not a target actor (AWAC has no actor target -- see module docstring).
            next_action, _ = self.policy.actor.action_log_prob(next_actor_features)
            q_next = self.policy.q_values_all(
                next_critic_features, next_action, target=True
            ).min(dim=0).values
            target_q = (
                data.rewards.unsqueeze(-1)
                + self.gamma * (1.0 - data.dones.unsqueeze(-1)) * q_next
            )
        q_all = self.policy.q_values_all(critic_features, data.actions, target=False)
        expanded_target = target_q.unsqueeze(0).expand_as(q_all)
        critic_loss = sum(
            torch.nn.functional.mse_loss(q_pred, q_target)
            for q_pred, q_target in zip(q_all, expanded_target)
        )
        return critic_loss

    def _compute_actor_loss(self, data) -> torch.Tensor:
        # actor_features carries the encoder_sharing-governed gradient
        # (BasePolicy.extract_actor_features): detached under the default
        # "shared_critic_grad" sharing (numerically identical to the old
        # unconditional features.detach()), full gradient to actor_extractor
        # under "separate" (actor_extractor is then actor-loss-exclusive --
        # see AWACPolicy.actor_parameters).
        actor_features = self.policy.extract_actor_features(data.obs)
        with torch.no_grad():
            critic_features = self.policy.extract_critic_features(data.obs)
            pi_action, _ = self.policy.actor.action_log_prob(actor_features)
            v = self.policy.q_values_all(
                critic_features, pi_action, target=False
            ).min(dim=0).values
            q = self.policy.q_values_all(
                critic_features, data.actions, target=False
            ).min(dim=0).values
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self.awac_lambda), self.exp_adv_max
            )
        log_prob = self.policy.actor.evaluate_action_log_prob(
            actor_features, data.actions
        )
        return -(log_prob * weights).mean()

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch(self.batch_size)

            critic_loss = self._compute_critic_loss(data)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
            self.critic_optimizer.step()
            if self._lr_schedulers[0] is not None:
                self._lr_schedulers[0].step()

            actor_loss = self._compute_actor_loss(data)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip_grad_norm(self.policy.actor_parameters())
            self.actor_optimizer.step()
            if self._lr_schedulers[1] is not None:
                self._lr_schedulers[1].step()

            # No delayed update -- CORL's AWAC Polyak-updates the critic
            # target every step (unlike TD3-BC's policy_freq-gated update).
            polyak_update(
                self.policy.critic.parameters(),
                self.policy.critic_target.parameters(),
                self.tau,
            )

            if compute_info:
                for key, value in (
                    ("critic_loss", float(critic_loss.detach().item())),
                    ("actor_loss", float(actor_loss.detach().item())),
                ):
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        if not compute_info:
            return {}
        return {key: value / gradient_steps for key, value in metrics_sum.items()}


class _AWACRolloutTrainingShell(Off2OnReplayMixin, AWACCore, OffPolicyAlgorithm):
    """Internal rollout/eval shell that wires ``AWACCore`` into ``OffPolicyAlgorithm``.

    .. warning::
       **Do not instantiate this class directly.** It exists only to back
       :class:`~rl_garden.algorithms.Off2OnAWAC`. For standalone offline AWAC
       pretraining use :class:`AWAC`.
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
        offline_sampling: Literal["with_replace", "without_replace"] = "with_replace",
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
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
        std_parameterization: Literal["exp", "uniform"] = "exp",
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
        image_augmentation_seed: Optional[int] = None,
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
        )
        self._init_awac_params(
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
            awac_lambda=awac_lambda,
            exp_adv_max=exp_adv_max,
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
            std_parameterization=std_parameterization,
            encoder_config=encoder_config,
            obs_groups=obs_groups,
            critic_encoder_config=critic_encoder_config,
            encoder_sharing=encoder_sharing,
            image_augmentation_seed=image_augmentation_seed,
        )
        self._setup_model()
        self._init_off2on_params(offline_sampling=offline_sampling)


class AWAC(AWACCore, OfflineRLAlgorithm):
    """Offline AWAC: advantage-weighted regression actor + twin-Q critic."""

    _compatible_checkpoint_algorithms = ("AWAC",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        gamma: float = 0.99,
        offline_sampling: str = "with_replace",
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
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
        std_parameterization: Literal["exp", "uniform"] = "exp",
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
        image_augmentation_seed: Optional[int] = None,
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
        self._init_awac_params(
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
            awac_lambda=awac_lambda,
            exp_adv_max=exp_adv_max,
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
            std_parameterization=std_parameterization,
            encoder_config=encoder_config,
            obs_groups=obs_groups,
            critic_encoder_config=critic_encoder_config,
            encoder_sharing=encoder_sharing,
            image_augmentation_seed=image_augmentation_seed,
        )
        self._setup_model()
