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
``FlattenExtractor`` either way): ``"shared_critic_grad"`` (default) follows
AGENTS.md's project convention and every other vision-capable algorithm
(``SACPolicy``), one encoder trained by critic loss, actor path detached.
``"separate"`` matches FQL's own JAX reference exactly (three independent
encoder instances). See ``FQLPolicy``'s docstring for the full
gradient-isolation argument per mode.

``_critic_loss`` averages (not sums) the per-critic MSE across the ensemble,
matching the reference's ``mean((q - target_q)**2)`` over the full
``(n_critics, batch)`` array -- summing would scale the critic gradient by
``n_critics`` relative to the reference.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.common.logger import Logger
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.factory import build_observation_encoder
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.networks.actor_vector_field import flow_onestep_distill_loss
from rl_garden.observations import ObsGroups, ObservationSchema, resolve_obs_groups
from rl_garden.policies.fql_policy import EncoderSharing, FQLPolicy


class FQLCore:
    """Shared FQL loss/network logic."""

    _SUPPORTED_POLICY_KWARGS = frozenset(
        {"actor_extractor_class", "actor_extractor_kwargs"}
    )

    # The FQL family has no "shared" mode (both actor and critic losses
    # training one encoder with no stop-gradient): only "shared_critic_grad"
    # (one encoder, actor path detached -- the default) or "separate" (three
    # independent encoder instances, matching the JAX reference) make sense.
    # Checked by ObservationEncoderMixin._resolve_encoder_sharing against the
    # resolved value, and read statically by algorithm_registry's preflight.
    encoder_sharing_choices: tuple = ("shared_critic_grad", "separate")

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
        encoder_sharing: Optional[EncoderSharing] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        reference_compatible: bool = False,
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
        if reference_compatible:
            invalid = []
            if encoder_config is not None:
                invalid.append("visual/encoded observations")
            if critic_encoder_config is not None or obs_groups is not None:
                invalid.append("asymmetric observation encoders/groups")
            if encoder_sharing not in (None, "shared_critic_grad"):
                invalid.append("encoder_sharing")
            if n_critics != 2:
                invalid.append("n_critics != 2")
            if actor_use_layer_norm or not critic_use_layer_norm:
                invalid.append("actor_use_layer_norm/critic_use_layer_norm")
            if actor_use_group_norm or critic_use_group_norm or critic_dropout_rate is not None:
                invalid.append("group norm or critic dropout")
            if backbone_type != "mlp" or activation_fn != "gelu" or kernel_init != "xavier_uniform":
                invalid.append("backbone_type, activation_fn, or kernel_init")
            if invalid:
                raise ValueError(
                    "reference_compatible FQL only supports the published state MLP recipe; unsupported: "
                    + ", ".join(invalid)
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
        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self.critic_encoder_config = critic_encoder_config
        self.reference_compatible = reference_compatible

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
            "reference_compatible": self.reference_compatible,
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
        observation_space = self.env.single_observation_space
        if self.reference_compatible:
            if ObservationSchema.from_space(observation_space).has_images:
                raise ValueError(
                    "reference_compatible FQL only supports the published state MLP recipe; "
                    "unsupported: visual/encoded observations"
                )
            action_space = self.env.single_action_space
            if not (
                torch.allclose(torch.as_tensor(action_space.low), torch.full(action_space.shape, -1.0))
                and torch.allclose(torch.as_tensor(action_space.high), torch.ones(action_space.shape))
            ):
                raise ValueError(
                    "reference_compatible FQL requires actions normalized to [-1, 1], "
                    "matching the JAX reference's literal clipping."
                )
        # Mixin resolves the actor/critic pair (actor_onestep_flow's own
        # encoder + critic's own encoder, or one shared encoder). FQL needs a
        # third encoder for actor_bc_flow -- the plan's documented single
        # exception to the actor_extractor/critic_extractor contract -- built
        # directly over the same key subset as the actor role.
        extractor_kwargs = self._policy_extractor_kwargs(observation_space)
        if self.encoder_sharing == "separate":
            actor_keys = resolve_obs_groups(
                self.observation_encoders.schema, self.obs_groups
            )["actor"].keys
            actor_bc_flow_encoder = build_observation_encoder(
                observation_space, self.encoder_config, keys=actor_keys
            )
        else:
            actor_bc_flow_encoder = None
        self.policy = FQLPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
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
            actor_bc_flow_encoder=actor_bc_flow_encoder,
            reference_compatible=self.reference_compatible,
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

    def _sample_train_batch(self, batch_size: int):
        if self.offline_sampling == "with_replace":
            return self.replay_buffer.sample(batch_size)
        if self.offline_sampling == "without_replace":
            sample = getattr(self.replay_buffer, "sample_without_repeat", None)
            if sample is None:
                raise ValueError(
                    "offline_sampling='without_replace' requires a replay buffer "
                    "with sample_without_repeat()."
                )
            return sample(batch_size)
        raise ValueError(f"Unknown offline_sampling: {self.offline_sampling!r}")

    @staticmethod
    def _critic_loss(q_all: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        # Mean over the critic ensemble, matching the reference's single
        # `mean((q - target_q)**2)` over the full (n_critics, batch) array --
        # summing here instead would scale the critic gradient by n_critics.
        expanded_target = target_q.unsqueeze(0).expand_as(q_all)
        return torch.stack(
            [F.mse_loss(q_pred, q_target) for q_pred, q_target in zip(q_all, expanded_target)]
        ).mean()

    def _aggregate_target_q(self, q_all: torch.Tensor) -> torch.Tensor:
        if self.q_agg == "min":
            return q_all.min(dim=0).values
        return q_all.mean(dim=0)

    def _clip_grad_norm(self, params) -> None:
        if self.grad_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(list(params), self.grad_clip_norm)

    def _critic_update(self, data, obs_features: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            next_features_critic = self.policy.extract_critic_features(data.next_obs)
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

        return {"critic_loss": float(critic_loss.detach().item())}

    def _actor_update(self, data, obs_features: torch.Tensor) -> dict[str, float]:
        # In "shared_critic_grad" mode bc/onestep/q features all alias one detached
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

    def _update_targets(self) -> None:
        polyak_update(
            self.policy.critic.parameters(),
            self.policy.critic_target.parameters(),
            self.tau,
        )

    def reference_losses(
        self, data, *, next_noise: torch.Tensor, x0: torch.Tensor, times: torch.Tensor,
        distill_noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all published FQL losses from caller-supplied random tensors.

        This is intentionally tensor-valued for a JAX/PyTorch parity harness.
        It only applies to the state-only reference recipe.
        """
        if not self.reference_compatible:
            raise RuntimeError("reference_losses requires reference_compatible=True.")
        obs_features = self.policy.extract_critic_features(data.obs)
        next_features = self.policy.extract_critic_features(data.next_obs)
        with torch.no_grad():
            next_action = self.policy.actor_onestep_flow(next_features, next_noise).clamp(-1.0, 1.0)
            next_q = self._aggregate_target_q(self.policy.q_values_all(next_features, next_action, target=True))
            target_q = data.rewards + self.gamma * (1.0 - data.dones) * next_q
        q_all = self.policy.q_values_all(obs_features, data.actions)
        critic_loss = self._critic_loss(q_all, target_q)
        x_t = (1.0 - times) * x0 + times * data.actions
        pred = self.policy.actor_bc_flow(obs_features, x_t, times)
        bc_flow_loss = F.mse_loss(pred, data.actions - x0)
        actor_actions = self.policy.actor_onestep_flow(obs_features, distill_noise)
        with torch.no_grad():
            target_actions = self.policy.compute_flow_actions(obs_features, distill_noise, self.flow_steps)
        distill_loss = F.mse_loss(actor_actions, target_actions)
        q_pi = self.policy.q_values_all(obs_features, actor_actions.clamp(-1.0, 1.0)).mean(dim=0)
        q_loss = -q_pi.mean()
        if self.normalize_q_loss:
            q_loss = (1.0 / q_pi.abs().mean()).detach() * q_loss
        actor_loss = bc_flow_loss + self.alpha * distill_loss + q_loss
        return {"critic_loss": critic_loss, "actor_loss": actor_loss, "bc_flow_loss": bc_flow_loss,
                "distill_loss": distill_loss, "q_loss": q_loss, "q": q_pi.mean(), "target_q": target_q.mean()}

    def reference_update(
        self, data, *, next_noise: torch.Tensor, x0: torch.Tensor, times: torch.Tensor,
        distill_noise: torch.Tensor,
    ) -> dict[str, float]:
        """One FQL update where every gradient and EMA source is pre-update."""
        losses = self.reference_losses(
            data,
            next_noise=next_noise, x0=x0, times=times, distill_noise=distill_noise,
        )
        critic_params = tuple(self.policy.critic_and_encoder_parameters())
        actor_params = tuple(self.policy.actor_parameters())
        critic_grads = torch.autograd.grad(losses["critic_loss"], critic_params)
        actor_grads = torch.autograd.grad(losses["actor_loss"], actor_params)
        pre_critic = [parameter.detach().clone() for parameter in self.policy.critic.parameters()]
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        for parameter, grad in zip(critic_params, critic_grads):
            parameter.grad = grad
        for parameter, grad in zip(actor_params, actor_grads):
            parameter.grad = grad
        self._clip_grad_norm(critic_params)
        self._clip_grad_norm(actor_params)
        self.critic_optimizer.step()
        self.actor_optimizer.step()
        for target, source in zip(self.policy.critic_target.parameters(), pre_critic):
            target.data.mul_(1.0 - self.tau).add_(source, alpha=self.tau)
        for scheduler in self._lr_schedulers:
            if scheduler is not None:
                scheduler.step()
        return {key: float(value.detach().item()) for key, value in losses.items() if key.endswith("loss")}

    def _reference_update(self, data) -> dict[str, float]:
        batch, action_dim = data.actions.shape
        device, dtype = data.actions.device, data.actions.dtype
        return self.reference_update(
            data,
            next_noise=torch.randn(batch, action_dim, device=device, dtype=dtype),
            x0=torch.randn(batch, action_dim, device=device, dtype=dtype),
            times=torch.rand(batch, 1, device=device, dtype=dtype),
            distill_noise=torch.randn(batch, action_dim, device=device, dtype=dtype),
        )

    def load(
        self,
        path,
        strict: bool = True,
        load_replay_buffer: bool = True,
        load_optimizers: bool = True,
    ):
        """Reject incompatible reference-mode checkpoints before any state mutation.

        ``BaseAlgorithm.load`` deliberately exposes no metadata-validation hook.
        This mode changes the policy layout and update semantics, so it must
        remain incompatible even when callers select ``strict=False``.
        """
        checkpoint = load_checkpoint_file(path, map_location=self.device)
        stored = bool(
            checkpoint.get("metadata", {})
            .get("hyperparameters", {})
            .get("reference_compatible", False)
        )
        if stored != self.reference_compatible:
            raise ValueError(
                "Incompatible checkpoint: reference_compatible mismatch "
                f"(checkpoint={stored}, current={self.reference_compatible})."
            )
        return super().load(
            path,
            strict=strict,
            load_replay_buffer=load_replay_buffer,
            load_optimizers=load_optimizers,
        )

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        """Run ``gradient_steps`` FQL updates.

        Per step: sample a batch, encode obs once (grad-enabled), then run
        the three overridable seams in order -- ``_critic_update``,
        ``_actor_update``, ``_update_targets`` -- and accumulate their
        returned metrics.
        """
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        counts: dict[str, int] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch(self.batch_size)

            # Critic's own encoding of obs -- grad-enabled, backprops into
            # the encoder via critic_loss (the encoder IS the policy's
            # critic_extractor, or actor_extractor when shared; only the
            # actor's encoder(s) differ).
            if self.reference_compatible:
                reference_metrics = self._reference_update(data)
                critic_metrics = {"critic_loss": reference_metrics["critic_loss"]}
                actor_metrics = {key: value for key, value in reference_metrics.items() if key != "critic_loss"}
            else:
                obs_features = self.policy.extract_critic_features(data.obs)
                critic_metrics = self._critic_update(data, obs_features)
                actor_metrics = self._actor_update(data, obs_features)
                self._update_targets()

            for key, value in {**critic_metrics, **actor_metrics}.items():
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
        encoder_sharing: Optional[EncoderSharing] = None,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        reference_compatible: bool = False,
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
            encoder_config=encoder_config,
            obs_groups=obs_groups,
            critic_encoder_config=critic_encoder_config,
            reference_compatible=reference_compatible,
        )

        self._setup_model()

    def _evaluate(self) -> dict[str, float]:
        metrics = super()._evaluate()
        completed = int(metrics.get("episodes_completed", 0))
        if self.reference_compatible and completed != self.num_eval_episodes:
            raise RuntimeError(
                "Reference-compatible FQL evaluation did not complete the requested "
                f"{self.num_eval_episodes} episodes before its {self.num_eval_steps}-step cap "
                f"(completed {completed})."
            )
        return metrics
