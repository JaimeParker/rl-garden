"""QAM (Q-learning with Adjoint Matching): flow-matching actor fine-tuned by
the critic's Q-gradient via adjoint matching, without backpropagating
through the full denoising chain (`3rd_party/qgf/agents/qam.py`, Li et al.
2026).

Unlike `QGF` (`rl_garden/algorithms/qgf.py`), whose actor is never trained
with an RL objective (all "policy improvement" happens at inference time),
QAM bakes the critic's guidance into training. `QAMPolicy.adj_matching`
(`rl_garden/policies/qam_policy.py`) constructs the training TARGET this
core regresses `actor_fast` toward; see that module's docstring for the
mechanism itself.

Built directly on `OfflineRLAlgorithm` (no rollout shell), following
`FQL`'s exact offline-only shape -- QAM never touches an env during
training, even the "ddpg" critic mode's `next_action` computation is a
forward-only inference call through the policy, not real env interaction.
Evaluation executes each chunk one action per env step (`ActionChunkingMixin`).

Formulas verified against `qam.py` (do not re-derive from memory):

- **A chunk window that runs past its episode's end has zero weight** in the
  critic, value and `flow_loss` terms, whose mean still runs over the whole
  batch (`valid_w = batch["valid"][..., -1]`, `qam.py:60,81,96,210`;
  standalone `3rd_party/qam/agents/qam.py:39,121`); `adj_loss` and the
  bolt-ons are unweighted, as there.
- **Critic dispatches on `critic_loss_type`** (`qam.py:42-88`): `"ddpg"`
  (the reference's actual default) bootstraps off `next_q =
  next_qs.mean(0) - rho*next_qs.std(0,unbiased=False)` (JAX's `jnp.std`
  defaults to population std / `ddof=0`, NOT PyTorch's default
  `unbiased=True` -- must pass `unbiased=False` explicitly or `rho>0`
  configs would silently diverge from the reference), using
  `next_action = self.policy.predict(next_obs, deterministic=False)` --
  critic and actor are *not* decoupled in this mode, same established
  pattern `ACFQLCore`'s own critic bootstrap already uses. `"iql"` is
  formula-identical to `QGFCore`'s critic loss (duplicated here, not
  extracted into a shared mixin -- `QGFCore` is already-shipped code and
  the duplicate is ~15 lines; see the project plan for the full
  reasoning).
- **Value loss** (`"iql"` mode only, `qam.py:90-103`): `q =
  target_critic(obs,batch_actions).min(dim=0)` -- **hardcoded min**, not
  `QGFCore`'s configurable `q_agg` (QAM's own reference has no
  `q_aggregation` field at all).
- **Actor loss** (`qam.py:188-324`): `flow_loss` (plain BC flow-matching
  MSE on `actor_slow`, **continuous-uniform `t`** matching FQL's own
  convention, NOT QGF's discrete grid) + `adj_loss` (regression onto
  `QAMPolicy.adj_matching`'s target, see that module) + at most one bolt-on
  (`assert fql_alpha * edit_scale == 0`, `qam.py:534-536`):
  - `fql_alpha>0`: distills `one_step_actor` from a `torch.no_grad()`-
    wrapped `compute_flow_actions` rollout (same implicit-stop-gradient
    convention `FQLCore`'s own module docstring already documents) + Q-max,
    literally FQL's own distill+q_loss shape.
  - `edit_scale>0` (**reconstructed** -- the reference's network-
    construction code for `edit_actor`/`edit_alpha` is missing, only the
    loss formula is specified): entropy-regularized Q-maximization of a
    small residual policy, using `AlphaTuner`'s existing
    `"lagrange_softplus"` mode as-is (see `QAMPolicy`'s docstring).

Action-space tiling, chunked replay construction, training-batch flattening,
evaluation queues and horizon metadata live in ``ActionChunkingMixin``.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch
import torch.nn.functional as F

from rl_garden.algorithms.chunked_rollout import ActionChunkingMixin
from rl_garden.algorithms.observation_encoders import EncoderSharing
from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.common.logger import Logger
from rl_garden.common.optim import make_lr_scheduler, make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.config import EncoderConfig
from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.qam import QAMParams
from rl_garden.observations import ObsGroups
from rl_garden.policies.qam_policy import QAMPolicy


class QAMCore:
    """Shared QAM loss/network logic. See module docstring."""

    # QAM's actor optimizer never includes the encoder (see
    # QAMPolicy.actor_parameters/critic_and_value_parameters) -- "shared"
    # (which would mean the actor loss also trains it) has no meaningful
    # implementation here, matching the FQL family's identical restriction
    # (rl_garden/algorithms/fql.py). Checked by ObservationEncoderMixin.
    # _resolve_encoder_sharing against the resolved value, and read
    # statically by algorithm_registry's preflight.
    encoder_sharing_choices: tuple = ("shared_critic_grad", "separate")

    def _init_qam_params(
        self,
        *,
        params: QAMParams,
        horizon_length: int = 1,
    ) -> None:
        self._init_action_chunking(horizon_length)
        if not (0.0 < params.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {params.tau}.")
        if not (0.0 < params.expectile < 1.0):
            raise ValueError(f"expectile must be in (0, 1), got {params.expectile}.")
        if params.critic_loss_type not in ("ddpg", "iql"):
            raise ValueError(
                f"critic_loss_type must be 'ddpg' or 'iql', got {params.critic_loss_type!r}."
            )
        if params.flow_steps < 1:
            raise ValueError(f"flow_steps must be >= 1, got {params.flow_steps}.")
        if params.fql_alpha > 0.0 and params.edit_scale > 0.0:
            raise ValueError(
                "Only one of fql_alpha and edit_scale can be non-zero "
                f"(got fql_alpha={params.fql_alpha}, edit_scale={params.edit_scale})."
            )
        if params.grad_clip_norm is not None and params.grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {params.grad_clip_norm}."
            )

        self._bind_params(params)
        self.net_arch: list[int] = (
            list(params.net_arch) if params.net_arch is not None else [512, 512, 512, 512]
        )

    def _optimizer_names(self) -> tuple[str, ...]:
        names = ("critic_optimizer", "actor_optimizer")
        if self.edit_scale > 0.0:
            names = names + ("edit_alpha_optimizer",)
        return names

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "net_arch": self.net_arch,
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

    def _setup_model(self) -> None:
        self.policy = QAMPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self._policy_action_space(),
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            actor_use_layer_norm=self.actor_use_layer_norm,
            critic_use_layer_norm=self.critic_use_layer_norm,
            value_use_layer_norm=self.value_use_layer_norm,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            activation_fn=self.activation_fn,
            critic_loss_type=self.critic_loss_type,
            rho=self.rho,
            expectile=self.expectile,
            flow_steps=self.flow_steps,
            best_of_n=self.best_of_n,
            inv_temp=self.inv_temp,
            residual=self.residual,
            target_actor=self.target_actor,
            clip_adj=self.clip_adj,
            use_target_grad=self.use_target_grad,
            fql_alpha=self.fql_alpha,
            edit_scale=self.edit_scale,
            edit_target_entropy=self.edit_target_entropy,
            edit_target_entropy_multiplier=self.edit_target_entropy_multiplier,
            **self._policy_extractor_kwargs(
                self.env.single_observation_space,
                augmentation_seed=self._image_augmentation_seed,
            ),
        ).to(self.device)

        self.critic_optimizer = make_optimizer(
            list(self.policy.critic_and_value_parameters()),
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
        if self.policy.edit_alpha is not None:
            self.edit_alpha_optimizer = make_optimizer(
                list(self.policy.edit_alpha.parameters()),
                lr=self.edit_alpha_lr,
                weight_decay=0.0,
                use_adamw=self.use_adamw,
            )
        self.replay_buffer = self._build_replay_buffer()

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
    def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return weight * diff.pow(2)

    def _critic_loss(
        self, data, critic_features: torch.Tensor, flat_actions: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            next_features = self.policy.extract_critic_features(data.next_obs)
            if self.critic_loss_type == "ddpg":
                next_action = self.policy.predict(data.next_obs, deterministic=False)
                next_qs = self.policy.q_values_all(next_features, next_action, target=True)
                next_q = next_qs.mean(dim=0) - self.rho * torch.std(
                    next_qs, dim=0, unbiased=False
                )
            else:  # "iql"
                next_q = self.policy.value(next_features)
            target_q = data.rewards.unsqueeze(-1) + data.discounts.unsqueeze(-1) * next_q

        complete_window = data.valid[:, -1].to(target_q.dtype).unsqueeze(-1)
        q_all = self.policy.q_values_all(critic_features, flat_actions, target=False)
        critic_loss = ((q_all - target_q).square() * complete_window).mean()
        info = {
            "critic_loss": float(critic_loss.detach().item()),
            "q_mean": float(q_all.detach().mean().item()),
        }

        if self.critic_loss_type == "iql":
            with torch.no_grad():
                target_qs = self.policy.q_values_all(
                    critic_features.detach(), flat_actions, target=True
                )
                q_for_value = target_qs.min(dim=0).values
            values = self.policy.value(critic_features)
            value_loss = (
                self._expectile_loss(q_for_value - values, self.expectile) * complete_window
            ).mean()
            critic_loss = critic_loss + value_loss
            info["value_loss"] = float(value_loss.detach().item())
            info["v_mean"] = float(values.detach().mean().item())

        return critic_loss, info

    def _actor_loss(
        self,
        features: torch.Tensor,
        critic_features: torch.Tensor,
        flat_actions: torch.Tensor,
        complete_window: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size = flat_actions.shape[0]
        device, dtype = flat_actions.device, flat_actions.dtype

        a0 = torch.randn_like(flat_actions)
        t = torch.rand(batch_size, 1, device=device, dtype=dtype)
        x_t = (1 - t) * a0 + t * flat_actions
        vel = flat_actions - a0
        pred = self.policy.actor_slow(features, x_t, t)
        flow_loss = ((pred - vel).square().mean(dim=-1) * complete_window.to(dtype)).mean()
        actor_loss = flow_loss
        info: dict[str, float] = {"flow_loss": float(flow_loss.detach().item())}

        xs, adjs, ts, pre_adj_info = self.policy.adj_matching(features, critic_features)
        h = 1.0 / self.flow_steps
        sigmas = torch.sqrt(2 * (1 - ts + h) / (ts + h))
        obs_expanded = features.unsqueeze(0).expand(self.flow_steps, batch_size, -1)
        vf_fine = self.policy.actor_fast(obs_expanded, xs, ts)
        actor_slow_eff = self.policy._effective_actor_slow()
        vf_base = actor_slow_eff(obs_expanded, xs, ts)

        if self.residual:
            adj_term = torch.square(vf_fine * 2 / sigmas + sigmas * adjs).sum(dim=-1)
        else:
            adj_term = torch.square((vf_fine - vf_base) * 2 / sigmas + sigmas * adjs).sum(dim=-1)
        adj_loss = adj_term.sum(dim=0).mean()
        actor_loss = actor_loss + adj_loss
        info["adj_loss"] = float(adj_loss.detach().item())
        info.update(pre_adj_info)

        if self.fql_alpha > 0.0:
            model = "slow,fast" if self.residual else "fast"
            fql_noises = torch.randn(
                batch_size, self.policy.full_action_dim, device=device, dtype=dtype
            )
            with torch.no_grad():
                flow_actions = self.policy.compute_flow_actions(
                    features, fql_noises, self.flow_steps, model=model
                )
            os_actions = self.policy.one_step_actor(features, fql_noises)
            fql_distill_loss = F.mse_loss(os_actions, flow_actions)
            os_clipped = os_actions.clamp(self.policy.action_low, self.policy.action_high)
            fql_qs = self.policy.q_values_all(critic_features, os_clipped, target=False)
            fql_q_loss = -fql_qs.mean(dim=0).mean()
            actor_loss = actor_loss + fql_q_loss + self.fql_alpha * fql_distill_loss
            info["fql_distill_loss"] = float(fql_distill_loss.detach().item())
            info["fql_q_loss"] = float(fql_q_loss.detach().item())

        elif self.edit_scale > 0.0:
            model = "slow,fast" if self.residual else "fast"
            edit_noises = torch.randn(
                batch_size, self.policy.full_action_dim, device=device, dtype=dtype
            )
            with torch.no_grad():
                flow_actions = self.policy.compute_flow_actions(
                    features, edit_noises, self.flow_steps, model=model
                )
            edit_features = torch.cat([features, flow_actions], dim=-1)
            edit, edit_log_prob = self.policy.edit_actor.action_log_prob(edit_features)
            edited = (flow_actions + edit * self.edit_scale).clamp(
                self.policy.action_low, self.policy.action_high
            )
            qs = self.policy.q_values_all(critic_features, edited, target=False)
            edit_q_loss = -qs.mean(dim=0).mean()

            alpha_detached = self.policy.edit_alpha.current_alpha().detach()
            edit_entropy_loss = (edit_log_prob * alpha_detached).mean()
            edit_alpha_loss = self.policy.edit_alpha.loss(
                edit_log_prob.detach(), self.policy.edit_target_entropy
            )

            actor_loss = actor_loss + edit_q_loss + edit_entropy_loss + edit_alpha_loss
            info["edit_q_loss"] = float(edit_q_loss.detach().item())
            info["edit_entropy_loss"] = float(edit_entropy_loss.detach().item())
            info["edit_alpha_loss"] = float(edit_alpha_loss.detach().item())

        info["actor_loss"] = float(actor_loss.detach().item())
        return actor_loss, info

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
            flat_actions = data.actions

            # --- critic (+ value in iql mode) ---
            critic_features = self.policy.extract_critic_features(data.obs)
            critic_loss, critic_info = self._critic_loss(data, critic_features, flat_actions)

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self._clip_grad_norm(self.policy.critic_and_value_parameters())
            self.critic_optimizer.step()
            if self._lr_schedulers[0] is not None:
                self._lr_schedulers[0].step()

            # --- actor: BasePolicy.extract_actor_features applies the
            # encoder_sharing stop-gradient rule (matches this loop's old
            # manual features.detach() exactly when encoder_sharing==
            # "shared_critic_grad", the default); the Q(s, pi(s)) terms below
            # re-extract critic-role features (stop_gradient=True) via
            # critic_features_for, mirroring SACCore's own convention. ---
            actor_features = self.policy.extract_actor_features(data.obs)
            actor_critic_features = self.policy.critic_features_for(
                data.obs, actor_features, stop_gradient=True
            )
            actor_loss, actor_info = self._actor_loss(
                actor_features, actor_critic_features, flat_actions, data.valid[:, -1]
            )

            self.actor_optimizer.zero_grad(set_to_none=True)
            if self.policy.edit_alpha is not None:
                self.edit_alpha_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip_grad_norm(self.policy.actor_parameters())
            self.actor_optimizer.step()
            if self.policy.edit_alpha is not None:
                self.edit_alpha_optimizer.step()
            if self._lr_schedulers[1] is not None:
                self._lr_schedulers[1].step()

            polyak_update(
                self.policy.critic.parameters(), self.policy.target_critic.parameters(), self.tau
            )
            polyak_update(
                self.policy.actor_slow.parameters(),
                self.policy.target_actor_slow.parameters(),
                self.tau,
            )

            for key, value in {**critic_info, **actor_info}.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

        if not compute_info:
            return {}
        return {key: metrics_sum[key] / counts[key] for key in metrics_sum}


class QAM(ActionChunkingMixin, QAMCore, OfflineRLAlgorithm):
    """Offline QAM: flow-matching actor fine-tuned via adjoint matching.
    See module docstring."""

    params_class = QAMParams

    _compatible_checkpoint_algorithms = ("QAM",)

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
        params: Optional[QAMParams] = None,
        **overrides,
    ) -> None:
        params = resolve_params(self.params_class, params, overrides)
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
        self._init_qam_params(
            params=params,
            horizon_length=horizon_length,
        )
        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        # QAM's actor optimizer never includes the encoder (see
        # QAMPolicy.actor_parameters/critic_and_value_parameters) -- "shared"
        # (which would mean the actor loss also trains it) has no meaningful
        # implementation here, matching QAMPolicy's own restriction and the
        # FQL family's identical reasoning (rl_garden/algorithms/fql.py).
        self.encoder_sharing = encoder_sharing
        self.critic_encoder_config = critic_encoder_config
        self._image_augmentation_seed = image_augmentation_seed

        self._setup_model()
