"""FlowPPO: online PPO fine-tuning for flow-matching policies (item 2 of the
RL-100 gap-closing plan, ``~/.claude/plans/git-diff-diff-haiku-subagent-
cuddly-glade.md`` Milestone D).

Ported from RL-100's stochastic flow rollout (``unidpg/diffusion_policy/
diffusers_patch/flow_match_scheduler.py::FlowMatchSchedulerExtended``) plus
its per-step PPO-clip update (``unidpg/uni_ppo.py``), verified against
source directly. **Not** a ``DPPO`` subclass -- unlike ``DiffusionCMDistillOnline``
(Milestone C), where subclassing ``DPPO`` was strictly better reuse because
``DPPOPolicy``'s frozen/trainable diffusion-pair internals were literally
reusable, none of that transfers here: a flow-matching vector field
(``ActorVectorField``) has no frozen/trainable split and no DDPM chain math.
``FlowPPOCore`` is a fresh mixin parallel to (not derived from) ``DPPOCore``,
mirroring only its proven *engineering* pattern (joint ``(env_step,
step)``-minibatch PPO update via ``torch.unravel_index``, the same
checkpoint-hook shape) -- not its diffusion-specific content.

Design notes carried over from this session's design research (see the plan
file for full derivations and verification):

- RL-100's SDE formulas are written in ``sigma``-space (``sigma=1`` noise,
  velocity data->noise); this repo's ``ActorVectorField``/FQL convention is
  the opposite (``t=0`` noise, velocity noise->data). ``flow_sde_step``/
  ``flow_logprob`` (``rl_garden/networks/actor_vector_field.py``) are
  re-derived directly in this repo's ``t``-space, numerically verified to
  collapse onto ``ActorVectorField.integrate()`` at ``noise_level=0``.
- ``"cps"``'s final SDE step has zero-variance density regardless of
  ``noise_level``; ``clip_std_min`` defaults to RL-100's actual training-
  config value (``0.0067``), not its scheduler class's unguarded ``0.0``
  default, to avoid a near-zero-std division blowing up the PPO ratio.
- Default ``logprob_mode="gaussian"`` (a true density), not RL-100's own
  ``"pseudo"`` default -- ``ratio=exp(delta_logprob)`` under ``"pseudo"``
  is not a real probability ratio and loses PPO's statistical calibration.
  ``"pseudo"`` stays available as an opt-in RL-100-parity flag.
- No DPPO-style per-step ``gamma_denoising`` advantage discount: checked
  RL-100's actual offline-chunk PPO loop (``uni_ppo.py``) and it applies the
  *same* env-level advantage to every flow step of one action's trajectory
  (no gamma-in-between-flow-steps weighting) -- so this port does the same,
  simpler than DPPO's diffusion-specific discount-by-remaining-steps trick.
  A consequence: without that discount, a training minibatch's
  ``(env_step, flow_step)`` pairs are an exactly ``flow_steps``-fold
  duplicated sample of the per-env-step advantage population (DPPO's
  discount incidentally decorrelates its own duplicates). Advantage
  normalization and quantile clamping are therefore computed once in
  ``train()`` over the true per-env-step population (before the minibatch
  loop), not per-minibatch-over-pairs -- duplication leaves mean/std
  unchanged, but a per-minibatch quantile over duplicated values would not
  be the population quantile.
- No new buffer file: ``DiffusionChainBuffer``'s actual code has zero
  diffusion-specific logic (a plain tensor allocator/filler); ``FlowPPO``
  reuses it directly with ``horizon_steps=1`` fixed (this port bakes any
  action chunking into ``FlowPPOPolicy.action_dim`` as one flat vector,
  FQL/ACFQL-style, rather than DPPO's separate per-timestep horizon axis).
- Clip coefficient is a single constant (``clip_coef``, standard PPO
  default 0.2), not DPPO's per-denoising-step annealing schedule
  (``clip_ploss_coef_base``/``rate``) -- RL-100's own flow PPO uses a plain
  scalar ``clip_ratio``/``epsilon`` too, so this is not a simplification
  relative to the reference, just relative to DPPO's own (diffusion-
  specific) tuning.

Action chunking (``ActionChunkWrapper``) is optional, same infrastructure as
DPPO's, but handled FQL/ACFQL-style: ``horizon_length`` future actions are
denoised jointly as one flat ``action_dim = horizon_length *
base_action_dim`` vector inside ``ActorVectorField``, not as a separate
per-timestep axis inside the network.

No BC-checkpoint warm-start is implemented in this port (unlike DPPO/
``DiffusionCMDistillOnline``, which load a matching offline-pretrained
checkpoint format) -- FQL/ACFQL's own checkpoint format was not asked to be
bridged into this algorithm; ``FlowPPO`` trains ``actor``/``critic`` from
scratch, same as plain ``PPO``.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.algorithms.on_policy import OnPolicyAlgorithm
from rl_garden.algorithms.ppo import ppo_clip_policy_loss
from rl_garden.buffers.diffusion_chain_buffer import DiffusionChainBuffer
from rl_garden.buffers.rollout_buffer import RolloutBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.obs_utils import flatten_leading_dims, index_obs
from rl_garden.common.optim import ScheduleType, make_lr_scheduler, make_optimizer
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.combined import (
    CombinedExtractor,
    ImageEncoderFactory,
    default_image_encoder_factory,
)
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.networks import Activation, KernelInit
from rl_garden.policies.flow_ppo_policy import FlowPPOPolicy


class FlowPPOCore:
    """Shared FlowPPO loss/hyperparameter logic."""

    def _init_flow_ppo_params(
        self,
        *,
        horizon_length: int = 1,
        flow_steps: int = 10,
        actor_mlp_dims: Optional[Sequence[int]] = None,
        actor_activation_fn: Optional[Activation] = None,
        critic_mlp_dims: Optional[Sequence[int]] = None,
        critic_activation_fn: Optional[Activation] = None,
        kernel_init: Optional[KernelInit] = None,
        sde_type: Literal["sde", "cps"] = "cps",
        noise_level: float = 0.7,
        clip_std_min: float = 0.0067,
        sigma_safe_max: float = 0.9,
        logprob_mode: Literal["gaussian", "pseudo"] = "gaussian",
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        critic_warmup_updates: int = 0,
        update_epochs: int = 5,
        update_batch_size: int = 50_000,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0.0,
        clip_advantage_upper_quantile: float = 1.0,
        vf_coef: float = 0.5,
        target_kl: Optional[float] = 1.0,
    ) -> None:
        if horizon_length < 1:
            raise ValueError(f"horizon_length must be >= 1, got {horizon_length}.")
        if flow_steps <= 0:
            raise ValueError(f"flow_steps must be positive, got {flow_steps}.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )

        self.horizon_length = horizon_length
        self.flow_steps = flow_steps
        self.actor_mlp_dims = list(actor_mlp_dims) if actor_mlp_dims is not None else [256, 256, 256]
        self.actor_activation_fn = actor_activation_fn
        self.critic_mlp_dims = list(critic_mlp_dims) if critic_mlp_dims is not None else [256, 256, 256]
        self.critic_activation_fn = critic_activation_fn
        self.kernel_init = kernel_init
        self.sde_type = sde_type
        self.noise_level = noise_level
        self.clip_std_min = clip_std_min
        self.sigma_safe_max = sigma_safe_max
        self.logprob_mode = logprob_mode
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.lr_schedule: ScheduleType = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_min_ratio = lr_min_ratio
        self.grad_clip_norm = grad_clip_norm
        self.critic_warmup_updates = critic_warmup_updates
        self.update_epochs = update_epochs
        self.update_batch_size = update_batch_size
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss_coef = clip_vloss_coef
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile
        self.vf_coef = vf_coef
        self.target_kl = target_kl

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("actor_optimizer", "critic_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "horizon_length": self.horizon_length,
            "flow_steps": self.flow_steps,
            "actor_mlp_dims": self.actor_mlp_dims,
            "critic_mlp_dims": self.critic_mlp_dims,
            "sde_type": self.sde_type,
            "noise_level": self.noise_level,
            "clip_std_min": self.clip_std_min,
            "sigma_safe_max": self.sigma_safe_max,
            "logprob_mode": self.logprob_mode,
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


class FlowPPO(FlowPPOCore, OnPolicyAlgorithm):
    _compatible_checkpoint_algorithms = ("FlowPPO",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        num_steps: int = 50,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        horizon_length: int = 1,
        flow_steps: int = 10,
        actor_mlp_dims: Optional[Sequence[int]] = None,
        actor_activation_fn: Optional[Activation] = None,
        critic_mlp_dims: Optional[Sequence[int]] = None,
        critic_activation_fn: Optional[Activation] = None,
        kernel_init: Optional[KernelInit] = None,
        sde_type: Literal["sde", "cps"] = "cps",
        noise_level: float = 0.7,
        clip_std_min: float = 0.0067,
        sigma_safe_max: float = 0.9,
        logprob_mode: Literal["gaussian", "pseudo"] = "gaussian",
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant",
        lr_warmup_steps: int = 0,
        lr_decay_steps: int = 0,
        lr_min_ratio: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        critic_warmup_updates: int = 0,
        update_epochs: int = 5,
        update_batch_size: int = 50_000,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0.0,
        clip_advantage_upper_quantile: float = 1.0,
        vf_coef: float = 0.5,
        target_kl: Optional[float] = 1.0,
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        finite_horizon_gae: bool = False,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
    ) -> None:
        obs_space = env.single_observation_space
        if not isinstance(obs_space, (spaces.Box, spaces.Dict)):
            raise TypeError(
                f"FlowPPO supports Box or Dict observation spaces, got {type(obs_space)}."
            )
        self.image_encoder_factory = image_encoder_factory
        super().__init__(
            env=env,
            eval_env=eval_env,
            num_steps=num_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            finite_horizon_gae=finite_horizon_gae,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_final_checkpoint=save_final_checkpoint,
        )
        self._init_flow_ppo_params(
            horizon_length=horizon_length,
            flow_steps=flow_steps,
            actor_mlp_dims=actor_mlp_dims,
            actor_activation_fn=actor_activation_fn,
            critic_mlp_dims=critic_mlp_dims,
            critic_activation_fn=critic_activation_fn,
            kernel_init=kernel_init,
            sde_type=sde_type,
            noise_level=noise_level,
            clip_std_min=clip_std_min,
            sigma_safe_max=sigma_safe_max,
            logprob_mode=logprob_mode,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_min_ratio=lr_min_ratio,
            grad_clip_norm=grad_clip_norm,
            critic_warmup_updates=critic_warmup_updates,
            update_epochs=update_epochs,
            update_batch_size=update_batch_size,
            norm_adv=norm_adv,
            clip_coef=clip_coef,
            clip_vloss_coef=clip_vloss_coef,
            clip_advantage_lower_quantile=clip_advantage_lower_quantile,
            clip_advantage_upper_quantile=clip_advantage_upper_quantile,
            vf_coef=vf_coef,
            target_kl=target_kl,
        )

        expected_action_shape = (self.horizon_length,) + env.single_action_space.shape[1:]
        if env.single_action_space.shape != expected_action_shape:
            raise ValueError(
                "FlowPPO expects env.single_action_space to already be chunked by "
                "ActionChunkWrapper(env, act_steps=horizon_length); got shape "
                f"{env.single_action_space.shape}, expected {expected_action_shape}."
            )

        self._setup_model()

    def _default_features_extractor_class(self) -> type[BaseFeaturesExtractor]:
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Box):
            return FlattenExtractor
        return CombinedExtractor

    def _default_features_extractor_kwargs(self) -> dict[str, Any]:
        if isinstance(self.env.single_observation_space, spaces.Dict):
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

    def _setup_model(self) -> None:
        obs_space = self.env.single_observation_space
        raw_action_space = spaces.Box(
            low=self.env.single_action_space.low[0],
            high=self.env.single_action_space.high[0],
            shape=self.env.single_action_space.shape[1:],
            dtype=self.env.single_action_space.dtype,
        )
        features_extractor = self._build_features_extractor()
        self.policy = FlowPPOPolicy(
            observation_space=obs_space,
            features_extractor=features_extractor,
            action_space=raw_action_space,
            flow_steps=self.flow_steps,
            horizon_length=self.horizon_length,
            actor_mlp_dims=self.actor_mlp_dims,
            actor_activation_fn=self.actor_activation_fn,
            critic_mlp_dims=self.critic_mlp_dims,
            critic_activation_fn=self.critic_activation_fn,
            kernel_init=self.kernel_init,
            sde_type=self.sde_type,
            noise_level=self.noise_level,
            clip_std_min=self.clip_std_min,
            sigma_safe_max=self.sigma_safe_max,
            logprob_mode=self.logprob_mode,
        ).to(self.device)

        self.actor_optimizer = make_optimizer(
            list(self.policy.actor.parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=True,
        )
        self.critic_optimizer = make_optimizer(
            list(self.policy.critic.parameters())
            + list(self.policy.features_extractor.parameters()),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
            use_adamw=True,
        )
        self._lr_schedulers = [
            make_lr_scheduler(
                opt,
                schedule_type=self.lr_schedule,
                warmup_steps=self.lr_warmup_steps,
                decay_steps=self.lr_decay_steps,
                min_lr_ratio=self.lr_min_ratio,
            )
            for opt in (self.actor_optimizer, self.critic_optimizer)
        ]

        self.rollout_buffer = RolloutBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        self._chain_buffer = DiffusionChainBuffer(
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            ft_denoising_steps=self.flow_steps,
            horizon_steps=1,
            action_dim=self.policy.action_dim,
            device=self.device,
        )

    # --- rollout ---

    def _snapshot_window_initial_hidden(self, hidden, next_done: torch.Tensor):
        self._chain_buffer.reset()
        return hidden

    def _rollout_step(self, obs, hidden, episode_starts: torch.Tensor):
        del episode_starts
        with torch.no_grad():
            obs_t = self._obs_to_policy_device(obs)
            chain, log_probs, action = self.policy.sample_rollout_chain(obs_t)
            values = self.policy.predict_values(obs_t)
        self._chain_buffer.add(chain.unsqueeze(2), log_probs.unsqueeze(2))
        actions = self.policy.to_chunk_shape(action)
        mean_log_probs = log_probs.mean(dim=(1, 2))
        entropy = torch.zeros(self.num_envs, device=self.device)
        return actions, values, mean_log_probs, entropy, hidden

    def _predict_last_values(self, obs, hidden) -> torch.Tensor:
        return self.policy.predict_values(obs)

    # --- update ---

    def train(self) -> dict[str, float]:
        self.policy.train()
        current_itr = self._global_update
        self._global_update += 1

        num_steps, num_envs = self.num_steps, self.num_envs
        k = self.flow_steps
        total = num_steps * num_envs

        obs_flat = flatten_leading_dims(self.rollout_buffer.obs)
        returns_flat = self.rollout_buffer.returns.reshape(total)
        values_flat = self.rollout_buffer.values.reshape(total)
        advantages_flat = self.rollout_buffer.advantages.reshape(total)
        chains_flat = self._chain_buffer.chains.reshape(total, k + 1, self.policy.action_dim)
        old_logprobs_flat = self._chain_buffer.old_log_probs.reshape(total, k, self.policy.action_dim)

        # Normalize/clamp once over the true (env_step) population, not
        # inside the minibatch loss: unlike DPPO, FlowPPO applies no
        # per-step discount to decorrelate a given env-step's advantage
        # across its `k` flow-step pairs (see module docstring -- RL-100's
        # own flow PPO does the same), so a (env_step, flow_step)-pair
        # minibatch is an exactly k-fold-duplicated sample of this
        # population. Duplication leaves mean/std unchanged, but a
        # per-minibatch quantile over duplicated values is not the
        # population quantile -- compute both here, once, over the
        # genuinely distinct advantages.
        if self.norm_adv:
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (
                advantages_flat.std() + 1e-8
            )
        adv_min = torch.quantile(advantages_flat, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages_flat, self.clip_advantage_upper_quantile)
        advantages_flat = advantages_flat.clamp(min=adv_min, max=adv_max)

        total_pairs = total * k
        metrics_sum: dict[str, float] = {}
        num_updates = 0
        train_actor = current_itr >= self.critic_warmup_updates

        for _ in range(self.update_epochs):
            perm = torch.randperm(total_pairs, device=self.device)
            num_batches = max(1, total_pairs // self.update_batch_size)
            stop = False
            for b in range(num_batches):
                idx = perm[b * self.update_batch_size : (b + 1) * self.update_batch_size]
                batch_inds, flow_step_inds = torch.unravel_index(idx, (total, k))

                obs_b = index_obs(obs_flat, batch_inds)
                features_b = self.policy._features(obs_b)
                chains_prev_b = chains_flat[batch_inds, flow_step_inds]
                chains_next_b = chains_flat[batch_inds, flow_step_inds + 1]
                returns_b = returns_flat[batch_inds]
                values_b = values_flat[batch_inds]
                advantages_b = advantages_flat[batch_inds]
                logprobs_b = old_logprobs_flat[batch_inds, flow_step_inds]

                loss, info = self._flow_ppo_loss(
                    features_b,
                    chains_prev_b,
                    chains_next_b,
                    flow_step_inds,
                    returns_b,
                    values_b,
                    advantages_b,
                    logprobs_b,
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if train_actor:
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.actor.parameters(), self.grad_clip_norm
                        )
                    self.actor_optimizer.step()
                self.critic_optimizer.step()

                for key, value in info.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                num_updates += 1

                if self.target_kl is not None and info["approx_kl"] > self.target_kl:
                    stop = True
                    break
            if stop:
                break

        for sched in self._lr_schedulers:
            if sched is not None:
                sched.step()

        metrics = {key: value / num_updates for key, value in metrics_sum.items()}
        y_pred, y_true = values_flat, returns_flat
        var_y = torch.var(y_true)
        explained_var = (
            float("nan") if var_y == 0 else float((1 - torch.var(y_true - y_pred) / var_y).item())
        )
        metrics["explained_variance"] = explained_var
        return metrics

    def _flow_ppo_loss(
        self,
        features_b: torch.Tensor,
        chains_prev_b: torch.Tensor,
        chains_next_b: torch.Tensor,
        flow_step_inds_b: torch.Tensor,
        returns_b: torch.Tensor,
        values_b: torch.Tensor,
        advantages_b: torch.Tensor,
        logprobs_b: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Gradient isolation: features_b is grad-enabled (the critic loss
        # below, `self.policy.critic(features_b)`, trains the shared
        # features_extractor -- SACPolicy's own convention, mirrors
        # DPPO._dppo_loss's identical comment). The actor's log-prob path
        # must not also backprop into it -- detach the copy fed to the actor
        # rather than re-running the (possibly image) encoder a second time.
        features_actor = features_b.detach()
        newlogprobs = self.policy.get_logprobs_subsample(
            features_actor, chains_prev_b, chains_next_b, flow_step_inds_b
        ).mean(dim=-1)
        oldlogprobs = logprobs_b.mean(dim=-1)

        # advantages_b is already normalized/clamped once over the full
        # rollout population in train() -- see the comment there.
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()

        pg_loss = ppo_clip_policy_loss(advantages_b, ratio, self.clip_coef)

        newvalues = self.policy.critic(features_b).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns_b) ** 2
            v_clipped = values_b + torch.clamp(
                newvalues - values_b, -self.clip_vloss_coef, self.clip_vloss_coef
            )
            v_loss_clipped = (v_clipped - returns_b) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalues - returns_b) ** 2).mean()

        loss = pg_loss + self.vf_coef * v_loss
        info = {
            "loss": float(loss.detach().item()),
            "pg_loss": float(pg_loss.detach().item()),
            "v_loss": float(v_loss.detach().item()),
            "clipfrac": clipfrac,
            "approx_kl": float(approx_kl.item()),
            "ratio": float(ratio.mean().item()),
        }
        return loss, info
