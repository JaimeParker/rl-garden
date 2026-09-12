"""DDPG algorithm (DrQ-v2): DDPG + data augmentation + n-step returns.

Ported from ``3rd_party/drqv2``.  Inherits ``OffPolicyAlgorithm`` for the
rollout/eval/checkpoint loop and implements DrQ-v2's update mechanics:
deterministic actor, n-step TD targets, exploration noise schedule, and
no entropy regularisation.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.buffers.nstep_buffer import (
    LazyNextNStepReplayBuffer,
    NStepReplayBuffer,
)
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.common.schedules import schedule
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObsGroups
from rl_garden.observations.schema import ObservationContractError
from rl_garden.policies.ddpg_policy import DDPGPolicy


class DDPG(OffPolicyAlgorithm):
    """DrQ-v2: DDPG with data augmentation and n-step returns.

    Key differences from SAC
    ------------------------
    * **DDPG**: deterministic actor, no entropy, no log-prob.
    * **External noise schedule**: stddev is a function of step, not learned.
    * **N-step returns**: TD targets computed over ``nstep`` transitions.

    Parameters
    ----------
    policy_lr : float
        Learning rate for actor (encoder is trained by critic loss).
    q_lr : float
        Learning rate for critic + encoder.
    feature_dim : int
        Trunk dimension for actor/critic (DrQ-v2 default: 50).
    hidden_dim : int
        Hidden dimension for actor MLP (DrQ-v2 default: 1024).
    nstep : int
        Number of steps for n-step returns (DrQ-v2 default: 3).
    stddev_schedule : str
        Exploration noise schedule string (DrQ-v2 default: ``"linear(1.0,0.1,500000)"``).
    stddev_clip : float
        Clamp exploration noise magnitude (DrQ-v2 default: 0.3).
    num_expl_steps : int
        Number of initial steps with uniform random actions (DrQ-v2 default: 2000).
    """

    _compatible_checkpoint_algorithms = ("DDPG",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        # --- OffPolicyAlgorithm ---
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        learning_starts: int = 4_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.01,
        training_freq: int = 32,
        utd: float = 0.5,
        bootstrap_at_done: str = "truncated",
        # --- DDPG-specific ---
        policy_lr: float = 1e-4,
        q_lr: float = 1e-4,
        feature_dim: int = 50,
        hidden_dim: int = 1024,
        nstep: int = 3,
        stddev_schedule: str = "linear(1.0,0.1,500000)",
        stddev_clip: float = 0.3,
        num_expl_steps: int = 2000,
        # --- Optimizer ---
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        lr_schedule: ScheduleType = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        # --- Vision ---
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        image_augmentation_seed: Optional[int] = None,
        # --- Misc ---
        policy_kwargs: Optional[dict[str, Any]] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 10_000,
        num_eval_steps: int = 50,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_replay_buffer: bool = False,
        save_final_checkpoint: bool = True,
        mmap_dir: Optional[str | Path] = None,
        mmap_mode: Literal["create", "open"] = "create",
        replay_lazy_next_obs: bool = False,
        replay_pin_sampled_batch: bool = False,
    ) -> None:
        if mmap_dir is not None and save_replay_buffer:
            raise ValueError(
                "mmap replay buffers cannot be embedded in replay checkpoints; "
                "set save_replay_buffer=False"
            )
        if replay_lazy_next_obs and mmap_dir is not None:
            raise ValueError("lazy next_obs replay is not supported with mmap_dir")
        if replay_lazy_next_obs and save_replay_buffer:
            raise ValueError(
                "lazy next_obs replay cannot be embedded in replay checkpoints; "
                "set save_replay_buffer=False"
            )
        if replay_pin_sampled_batch and not replay_lazy_next_obs:
            raise ValueError(
                "replay_pin_sampled_batch currently requires replay_lazy_next_obs"
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

        self.mmap_dir = mmap_dir
        self.mmap_mode = mmap_mode
        self.replay_lazy_next_obs = replay_lazy_next_obs
        self.replay_pin_sampled_batch = replay_pin_sampled_batch
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.nstep = nstep
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.num_expl_steps = num_expl_steps
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip_norm = grad_clip_norm
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive or None, got {grad_clip_norm}.")

        # DrQ-v2's validated default: DrQv2Encoder + random-shift augmentation.
        # A caller-supplied encoder_config overrides every field wholesale.
        self.encoder_config = (
            encoder_config
            if encoder_config is not None
            else EncoderConfig(backbone="drqv2_conv", image_augmentation="random_shift")
        )
        self.obs_groups = obs_groups
        self._image_augmentation_seed = image_augmentation_seed

        self.policy_kwargs = dict(policy_kwargs or {})
        self._global_update: int = 0

        self._setup_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_features_extractor(self) -> BaseFeaturesExtractor:
        """``policy_kwargs`` explicit override (raw escape hatch, unchanged),
        else the schema-driven extractor resolved onto
        ``self.observation_encoders`` by ``_resolve_observation_encoders``
        (see ``_setup_model``, which resolves it eagerly so the "at least
        one image key" check below can run regardless of any override)."""
        features_extractor_class = self.policy_kwargs.get("features_extractor_class")
        if features_extractor_class is None:
            return self.observation_encoders.actor
        features_extractor_kwargs = self.policy_kwargs.get("features_extractor_kwargs") or {}
        return features_extractor_class(
            observation_space=self.env.single_observation_space,
            **features_extractor_kwargs,
        )

    def _setup_model(self) -> None:
        self._resolve_observation_encoders(
            self.env.single_observation_space,
            augmentation_seed=self._image_augmentation_seed,
        )
        if not self.observation_encoders.schema.has_images:
            raise ObservationContractError(
                "DDPG requires at least one image observation key; observation "
                f"space has keys {self.observation_encoders.schema.keys!r}."
            )
        features_extractor = self._build_features_extractor()
        self.policy = DDPGPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.q_optimizer = make_optimizer(
            list(self.policy.critic_and_encoder_parameters()),
            lr=self.q_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.actor_optimizer = make_optimizer(
            list(self.policy.actor_parameters()),
            lr=self.policy_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )

        self._lr_schedulers: list[Optional[Any]] = []
        for opt in (self.q_optimizer, self.actor_optimizer):
            self._lr_schedulers.append(
                make_lr_scheduler(
                    opt,
                    schedule_type=self.lr_schedule,
                    warmup_steps=self.lr_warmup_steps,
                    decay_steps=self.lr_decay_steps,
                    min_lr_ratio=self.lr_min_ratio,
                )
            )

        self.replay_buffer = self._build_replay_buffer()

    def _build_replay_buffer(self):
        # _setup_model already rejects an image-less observation space before
        # this is called, so obs_space is always a genuine Dict with at least
        # one image key here (only a Dict space can have rgb*/depth* keys --
        # see ObservationSchema.from_space).
        obs_space = self.env.single_observation_space
        buffer_cls = (
            LazyNextNStepReplayBuffer
            if self.replay_lazy_next_obs
            else NStepReplayBuffer
        )
        lazy_kwargs = (
            {"pin_sampled_batch": self.replay_pin_sampled_batch}
            if self.replay_lazy_next_obs
            else {}
        )
        return buffer_cls(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            nstep=self.nstep,
            gamma=self.gamma,
            storage_device=self.buffer_device,
            sample_device=self.device,
            mmap_dir=self.mmap_dir,
            mmap_mode=self.mmap_mode,
            **lazy_kwargs,
        )

    def _replay_buffer_step_kwargs(
        self,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
    ) -> dict[str, Any]:
        return {"episode_end": terminations | truncations}

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        if not learning_has_started:
            actions = self._explore_action(obs)
            return actions, actions, None

        with torch.no_grad():
            obs_device = self._obs_to_policy_device(obs)
            stddev = schedule(self.stddev_schedule, self._global_step)
            if self._global_step < self.num_expl_steps:
                actions = self._explore_action(obs)
            else:
                features = self.policy.extract_features(obs_device)
                dist = self.policy.actor(features, stddev)
                actions = dist.sample(clip=None)
        return actions, actions, None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _current_stddev(self) -> float:
        return schedule(self.stddev_schedule, self._global_step)

    def _target_action_noise(self) -> tuple[float, float]:
        """(std, clip) for the noise added when sampling the target action.

        Default: reuse the exploration noise schedule (current DDPG/DrQ-v2
        behavior). TD3 overrides this with a fixed pair decoupled from the
        exploration schedule (target policy smoothing).
        """
        return self._current_stddev(), self.stddev_clip

    def _should_update_actor_and_target(self) -> bool:
        """Whether to update the actor and polyak-average the target network
        this gradient step. Default: always (current DDPG behavior). TD3
        overrides this to delay updates by ``policy_freq`` steps.
        """
        return True

    def _actor_q_value(self, q_actor_all: torch.Tensor) -> torch.Tensor:
        """Reduce twin-Q values to the scalar used for the actor loss.

        Default: min over both critics (current DDPG/DrQ-v2 behavior). TD3
        overrides this to use the first critic only (canonical TD3 actor
        loss).
        """
        return q_actor_all.min(dim=0).values

    def train(
        self, gradient_steps: int, compute_info: bool = False
    ) -> dict[str, float]:
        critic_losses: list[torch.Tensor] = []
        actor_losses: list[torch.Tensor] = []
        info_accum: dict[str, list[torch.Tensor]] = {}

        for _ in range(gradient_steps):
            self._global_update += 1
            data = self.replay_buffer.sample(self.batch_size)
            self.policy.features_extractor.prepare_batch(data.obs, data.next_obs)

            # --- Critic update ---
            obs_features = self.policy.extract_features(data.obs)
            with torch.no_grad():
                next_features = self.policy.extract_features(data.next_obs)
                target_std, target_clip = self._target_action_noise()
                dist = self.policy.actor(next_features, target_std)
                next_action = dist.sample(clip=target_clip)
                target_q_all = self.policy.q_values_all(
                    next_features, next_action, target=True
                )
                target_q = (
                    data.rewards.reshape(-1, 1)
                    + data.discounts.reshape(-1, 1) * target_q_all.min(dim=0).values
                )

            q_all = self.policy.q_values_all(obs_features, data.actions, target=False)
            critic_loss = self._critic_loss(q_all, target_q)

            self.q_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.critic_and_encoder_parameters()),
                    self.grad_clip_norm,
                )
            self.q_optimizer.step()
            if self._lr_schedulers and self._lr_schedulers[0] is not None:
                self._lr_schedulers[0].step()

            if compute_info:
                critic_losses.append(critic_loss.detach())
                info_accum.setdefault("target_q", []).append(target_q.mean().detach())
                info_accum.setdefault("predicted_q", []).append(q_all.mean().detach())

            if self._should_update_actor_and_target():
                # --- Actor update ---
                stddev = self._current_stddev()
                features_detached = obs_features.detach()
                action = self.policy.actor_action_from_features(
                    features_detached,
                    stddev,
                    noise_clip=self.stddev_clip,
                )
                q_actor_all = self.policy.q_values_all(
                    features_detached, action, target=False
                )
                actor_loss = -self._actor_q_value(q_actor_all).mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy.actor_parameters()),
                        self.grad_clip_norm,
                    )
                self.actor_optimizer.step()
                if len(self._lr_schedulers) >= 2 and self._lr_schedulers[1] is not None:
                    self._lr_schedulers[1].step()

                # --- Target update ---
                polyak_update(
                    self.policy.critic.parameters(),
                    self.policy.critic_target.parameters(),
                    self.tau,
                )

                if compute_info:
                    actor_losses.append(actor_loss.detach())
                    info_accum.setdefault("stddev", []).append(
                        torch.tensor(stddev, device=self.device)
                    )

        if not compute_info:
            return {}

        def _mean(vals: list[torch.Tensor]) -> float:
            return float(torch.stack(vals).mean().item())

        out: dict[str, float] = {"critic_loss": _mean(critic_losses)}
        if actor_losses:
            out["actor_loss"] = _mean(actor_losses)
        for key in ("target_q", "predicted_q", "stddev"):
            if key in info_accum:
                out[key] = _mean(info_accum[key])
        return out

    @staticmethod
    def _critic_loss(
        q_all: torch.Tensor, target_q: torch.Tensor
    ) -> torch.Tensor:
        expanded_target = target_q.unsqueeze(0).expand_as(q_all)
        return sum(
            F.mse_loss(q_pred, q_target)
            for q_pred, q_target in zip(q_all, expanded_target)
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint_metadata(self) -> dict[str, Any]:
        meta = {
            **super()._checkpoint_metadata(),
            "policy_lr": self.policy_lr,
            "q_lr": self.q_lr,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "nstep": self.nstep,
            "stddev_schedule": self.stddev_schedule,
            "stddev_clip": self.stddev_clip,
            "num_expl_steps": self.num_expl_steps,
            "weight_decay": self.weight_decay,
            "use_adamw": self.use_adamw,
            "lr_schedule": self.lr_schedule,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_min_ratio": self.lr_min_ratio,
            "grad_clip_norm": self.grad_clip_norm,
            "replay_lazy_next_obs": self.replay_lazy_next_obs,
            "replay_pin_sampled_batch": self.replay_pin_sampled_batch,
            "encoder_sharing": self.encoder_sharing,
            "encoder_config": (
                dataclasses.asdict(self.encoder_config) if self.encoder_config is not None else None
            ),
            "obs_groups": (
                dataclasses.asdict(self.obs_groups) if self.obs_groups is not None else None
            ),
            "image_augmentation_seed": self._image_augmentation_seed,
        }
        return meta

    def load(
        self,
        path: str | Path,
        strict: bool = True,
        load_replay_buffer: bool = True,
        load_optimizers: bool = True,
    ) -> "DDPG":
        if self.mmap_dir is not None and load_replay_buffer:
            raise ValueError(
                "replay checkpoint loading is not supported with mmap buffers; "
                "use mmap_mode='open' and load_replay_buffer=False"
            )
        return super().load(
            path,
            strict=strict,
            load_replay_buffer=load_replay_buffer,
            load_optimizers=load_optimizers,
        )

    def load_replay_buffer(self, path: str | Path, strict: bool = True) -> None:
        if self.mmap_dir is not None:
            raise ValueError(
                "replay checkpoint loading is not supported with mmap buffers; "
                "use mmap_mode='open'"
            )
        super().load_replay_buffer(path, strict=strict)
