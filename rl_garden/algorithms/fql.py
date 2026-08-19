"""FQL (Flow Q-Learning): offline RL with a two-network flow-matching actor.

Ported from FQL's ``agents/fql.py``. Mirrors TD3BCCore/TD3BC's shape
(mixin + OfflineRLAlgorithm, two optimizers, no shared base-class changes)
rather than SAC/SACCore's -- FQL has no entropy term, no log_prob, and a
critic target with no soft-value term, the same shape as TD3-BC's target,
not SAC's.

Unlike the reference (one combined JAX optimizer over critic + both actor
networks, one combined backward pass), this port uses two optimizers
(critic_optimizer, actor_optimizer) matching every other rl-garden
algorithm's convention -- actor_optimizer covers both `actor_bc_flow` and
`actor_onestep_flow` (via `FQLPolicy.actor_parameters()`), so the summed
three-term actor loss still backprops through one `.step()` call, the same
mechanism `SACCore._actor_loss`/`TD3BC.train()` already rely on.

Gradient-path note (the one place needing an explicit `torch.no_grad()`):
the reference stop-gradients its distill-loss target (`compute_flow_actions`,
the teacher's multi-step Euler unroll) implicitly, by never passing
`params=grad_params` to that call. PyTorch has no such implicit convention,
so `target_flow_actions` below is computed inside `torch.no_grad()`
explicitly -- omitting it would let the distill-loss term contribute an
extra, unintended gradient into `actor_bc_flow` on top of `bc_flow_loss`'s
own gradient in the same step. Every other loss term already follows an
existing rl-garden pattern (critic weights are never in `actor_optimizer`,
so the q_loss term's plain differentiable critic forward needs no
`no_grad` -- same as `SACCore._actor_loss`/`TD3BC.train()`).

FQL has no delayed/policy_freq-style actor update: the reference updates
critic and actor on every gradient step (one combined backward covers both
losses), so this port does too.

``kernel_init`` defaults to ``"xavier_uniform"`` here (unlike TD3-BC/AWAC's
``None``): the reference's ``default_init()`` (Xavier-uniform, zero bias) is
applied unconditionally to every ``nn.Dense`` in both actors and the critic,
not a tunable choice like TD3-BC/AWAC's PyTorch-native CORL references.

Likewise ``activation_fn`` defaults to ``"gelu"`` here (unlike every other
rl-garden algorithm's ``None`` -> ReLU): the reference's MLP hardcodes
``nn.gelu`` unconditionally (``3rd_party/fql/utils/networks.py``).

``encoder_sharing`` (vision/Dict obs only -- Box obs uses a parameterless
``FlattenExtractor`` either way): ``"shared"`` (default) follows AGENTS.md's
project convention and every other vision-capable algorithm (``SACPolicy``),
one encoder trained by critic loss, actor path detached. ``"separate"``
matches FQL's own JAX reference exactly (three independent encoder
instances). See ``FQLPolicy``'s docstring for the full gradient-isolation
argument per mode.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.dict_buffer import DictReplayBuffer
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.combined import (
    CombinedExtractor,
    ImageEncoderFactory,
    default_image_encoder_factory,
)
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.policies.fql_policy import EncoderSharing, FQLPolicy


class FQLCore:
    """Shared FQL loss/network logic."""

    _SUPPORTED_POLICY_KWARGS = frozenset(
        {"features_extractor_class", "features_extractor_kwargs"}
    )

    def _init_fql_params(
        self,
        *,
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
    ) -> None:
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {tau}.")
        if flow_steps < 1:
            raise ValueError(f"flow_steps must be >= 1, got {flow_steps}.")
        if q_agg not in ("mean", "min"):
            raise ValueError(f"q_agg must be 'mean' or 'min', got {q_agg!r}.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )
        if encoder_sharing not in ("shared", "separate"):
            raise ValueError(
                f"encoder_sharing must be 'shared' or 'separate', got {encoder_sharing!r}."
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
        self.alpha = alpha
        self.flow_steps = flow_steps
        self.q_agg = q_agg
        self.normalize_q_loss = normalize_q_loss
        self.net_arch: list[int] = (
            list(net_arch) if net_arch is not None else [512, 512, 512, 512]
        )
        self.n_critics = n_critics
        self.actor_use_layer_norm = actor_use_layer_norm
        self.critic_use_layer_norm = critic_use_layer_norm
        self.actor_use_group_norm = actor_use_group_norm
        self.critic_use_group_norm = critic_use_group_norm
        self.num_groups = num_groups
        self.critic_dropout_rate = critic_dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.activation_fn = activation_fn
        self.encoder_sharing = encoder_sharing
        self.image_encoder_factory = image_encoder_factory

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
            "alpha": self.alpha,
            "flow_steps": self.flow_steps,
            "q_agg": self.q_agg,
            "normalize_q_loss": self.normalize_q_loss,
            "net_arch": self.net_arch,
            "n_critics": self.n_critics,
            "activation_fn": self.activation_fn,
            "encoder_sharing": self.encoder_sharing,
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
        if isinstance(obs_space, spaces.Dict):
            return CombinedExtractor
        raise TypeError(
            "FQL only supports Box or Dict observation spaces, got " + str(type(obs_space))
        )

    def _default_features_extractor_kwargs(self) -> dict[str, Any]:
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Dict):
            return {
                "image_encoder_factory": (
                    self.image_encoder_factory or default_image_encoder_factory()
                ),
            }
        return {}

    def _build_features_extractor(self) -> BaseFeaturesExtractor:
        cls = self._default_features_extractor_class()
        return cls(
            observation_space=self.env.single_observation_space,
            **self._default_features_extractor_kwargs(),
        )

    def _build_replay_buffer(self):
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Dict):
            return DictReplayBuffer(
                observation_space=obs_space,
                action_space=self.env.single_action_space,
                num_envs=self.num_envs,
                buffer_size=self.buffer_size,
                storage_device=self.buffer_device,
                sample_device=self.device,
            )
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                "FQL only supports Box or Dict observation spaces, got " + str(type(obs_space))
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
        if self.encoder_sharing == "separate":
            actor_bc_flow_encoder = self._build_features_extractor()
            actor_onestep_flow_encoder = self._build_features_extractor()
        else:
            actor_bc_flow_encoder = None
            actor_onestep_flow_encoder = None
        self.policy = FQLPolicy(
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

    @staticmethod
    def _critic_loss(q_all: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        expanded_target = target_q.unsqueeze(0).expand_as(q_all)
        return sum(
            F.mse_loss(q_pred, q_target)
            for q_pred, q_target in zip(q_all, expanded_target)
        )

    def _aggregate_target_q(self, q_all: torch.Tensor) -> torch.Tensor:
        if self.q_agg == "min":
            return q_all.min(dim=0).values
        return q_all.mean(dim=0)

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

            # Critic's own encoding of obs -- grad-enabled, backprops into
            # the encoder via critic_loss (the encoder IS self.features_extractor
            # in both encoder_sharing modes; only the actor's encoder(s) differ).
            obs_features = self.policy.extract_features(data.obs)

            # --- critic ---
            with torch.no_grad():
                next_features_critic = self.policy.extract_features(data.next_obs)
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

            q_all = self.policy.q_values_all(obs_features, data.actions, target=False)
            critic_loss = self._critic_loss(q_all, target_q)

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_encoder_parameters())
            self.critic_optimizer.step()
            if self._lr_schedulers[0] is not None:
                self._lr_schedulers[0].step()

            # --- actor (every step -- FQL has no policy_freq-style delay) ---
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
            pred_vel = self.policy.actor_bc_flow(bc_features, x_t, t)
            bc_flow_loss = F.mse_loss(pred_vel, vel_target)

            noises = torch.randn(batch_size, action_dim, device=device, dtype=dtype)
            with torch.no_grad():
                target_flow_actions = self.policy.compute_flow_actions(
                    bc_features, noises, self.flow_steps
                )
            actor_actions = self.policy.actor_onestep_flow(onestep_features, noises)
            distill_loss = F.mse_loss(actor_actions, target_flow_actions)

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

            polyak_update(
                self.policy.critic.parameters(),
                self.policy.critic_target.parameters(),
                self.tau,
            )

            for key, value in (
                ("critic_loss", float(critic_loss.detach().item())),
                ("actor_loss", float(actor_loss.detach().item())),
                ("bc_flow_loss", float(bc_flow_loss.detach().item())),
                ("distill_loss", float(distill_loss.detach().item())),
                ("q_loss", float(q_loss.detach().item())),
            ):
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

        if not compute_info:
            return {}
        return {key: metrics_sum[key] / counts[key] for key in metrics_sum}


class FQL(FQLCore, OfflineRLAlgorithm):
    """Offline FQL: twin-Q critic + two-network flow-matching actor."""

    _compatible_checkpoint_algorithms = ("FQL",)

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

        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, (spaces.Box, spaces.Dict)):
            raise TypeError(
                f"FQL supports only Box or Dict observation spaces, got {type(obs_space)}"
            )

        self._setup_model()
