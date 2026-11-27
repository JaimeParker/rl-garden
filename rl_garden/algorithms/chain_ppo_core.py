"""``ChainPPOCore``: PPO over the steps of a sampled action chain.

``DPPO`` (a DDPM denoising chain) and ``FlowPPO`` (a stochastic flow-SDE
chain) sample each env action through ``K`` stochastic chain steps and treat
every ``(env step, chain step)`` pair as one PPO sample: the policy ratio of
a pair is the ratio of that chain step's transition density
``p(x_{k+1} | x_k, obs)`` under the current and the rollout policy, and the
pair takes the advantage of its env step. This class owns everything the two
share; a subclass supplies its policy and the chain-specific terms.

What it owns:

- the rollout step (``_rollout_step``): the subclass's
  ``_sample_rollout_chain`` under ``no_grad``, the critic value, the chain
  buffer fill; the chain buffer reset at each window start
  (``_snapshot_window_initial_hidden``) and the bootstrap value
  (``_predict_last_values``);
- the update (``train``): advantages normalized (``norm_adv``) and clamped
  to their ``[clip_advantage_lower_quantile, clip_advantage_upper_quantile]``
  quantiles, per minibatch of pairs or once over the rollout's env steps
  (the class attribute ``advantage_normalization``, set by each subclass to
  what its reference does), then ``update_epochs`` passes
  over random minibatches of ``update_batch_size`` pairs (``randperm`` over
  all ``num_steps * num_envs * K`` pairs, the trailing remainder skipped);
  per minibatch the clipped surrogate, the value loss (clipped by
  ``clip_vloss_coef`` when set), one backward, the critic step every update
  and the actor step (gradient norm clipped to ``grad_clip_norm`` over the
  actor optimizer's parameters) only after ``critic_warmup_updates``
  updates; ``target_kl`` stops the update AFTER the minibatch whose
  approx-KL exceeds it took its step (the DPPO reference's timing); one LR
  scheduler step per optimizer; metrics averaged over the minibatches plus
  ``explained_variance``;
- the two optimizers ``actor_optimizer``/``critic_optimizer`` (AdamW), their
  LR schedulers (checkpoint state ``lr_scheduler_states``) and the shared
  checkpoint metadata.

A subclass implements ``_setup_model`` (build ``self.policy``, then call
``_build_chain_training_state``), ``_sample_rollout_chain`` and
``_chain_policy_terms``. ``self.policy`` must offer ``predict_values(obs)``
(differentiable, ``(B,)``), ``predict``, ``clamp_action`` and ``train()``.
"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar, Literal, Optional

import torch

from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.chain_ppo import ChainPPOCoreParams
from rl_garden.algorithms.observation_encoders import EncoderSharing
from rl_garden.algorithms.on_policy import OnPolicyAlgorithm
from rl_garden.algorithms.ppo_terms import (
    ClipCoef,
    approx_kl_clipfrac,
    clipped_value_loss,
    encoder_checkpoint_metadata,
    explained_variance,
    ppo_clip_policy_loss,
)
from rl_garden.buffers.diffusion_chain_buffer import DiffusionChainBuffer
from rl_garden.buffers.rollout_buffer import RolloutBuffer
from rl_garden.common.obs_utils import flatten_leading_dims, index_obs
from rl_garden.common.optim import make_lr_scheduler, make_optimizer
from rl_garden.common.types import Obs
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObsGroups


@dataclasses.dataclass(slots=True)
class ChainMinibatch:
    """One minibatch of ``(env step, chain step)`` pairs, ``B`` rows.

    The chain tensors keep the chain buffer's per-step shape ``(H, A)``
    (``H`` is the action horizon; 1 for ``FlowPPO``)."""

    obs: Obs
    #: ``x_k`` and ``x_{k+1}`` of each pair's chain step, ``(B, H, A)``.
    chains_prev: torch.Tensor
    chains_next: torch.Tensor
    #: Each pair's chain-step index in ``[0, K)``, ``(B,)`` long.
    chain_step_inds: torch.Tensor
    #: The rollout policy's elementwise log-probs of the step, ``(B, H, A)``.
    old_log_probs: torch.Tensor
    #: The env step's advantage, normalized and clamped per
    #: ``ChainPPOCore.advantage_normalization``.
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor


@dataclasses.dataclass(slots=True)
class ChainPolicyTerms:
    """The chain-specific side of one minibatch (``_chain_policy_terms``),
    1-D over its rows."""

    #: ``new_log_prob - old_log_prob`` of each pair's chain step.
    logratio: torch.Tensor
    #: ``logratio.exp()``.
    ratio: torch.Tensor
    #: The advantages the clipped surrogate weighs the ratio with.
    advantages: torch.Tensor
    #: A scalar, or one clip coefficient per row.
    clip_coef: ClipCoef


AdvantageNormalization = Literal["minibatch", "rollout"]


class ChainPPOCore(OnPolicyAlgorithm):
    """See the module docstring."""

    params_class = ChainPPOCoreParams

    #: Over which population advantages are normalized (``norm_adv``) and
    #: quantile-clamped: ``"minibatch"`` -- the minibatch's pairs, before
    #: ``_chain_policy_terms`` (the DPPO reference); ``"rollout"`` -- the
    #: rollout's env steps, once per update (the FlowPPO reference).
    advantage_normalization: ClassVar[AdvantageNormalization] = "rollout"

    def __init__(
        self,
        env: Any,
        *,
        encoder_config: Optional[EncoderConfig],
        obs_groups: Optional[ObsGroups],
        critic_encoder_config: Optional[EncoderConfig],
        encoder_sharing: Optional[EncoderSharing],
        params: ChainPPOCoreParams | None = None,
        **on_policy_kwargs: Any,
    ) -> None:
        """``on_policy_kwargs`` go to ``OnPolicyAlgorithm``. Does not call
        ``_setup_model``: a subclass sets its own fields first, then calls
        it."""
        parameter_names = self.params_class.__dataclass_fields__
        overrides = {name: on_policy_kwargs.pop(name) for name in tuple(on_policy_kwargs)
                     if name in parameter_names}
        params = resolve_params(self.params_class, params, overrides)
        actor_mlp_dims = params.actor_mlp_dims
        critic_mlp_dims = params.critic_mlp_dims
        grad_clip_norm = params.grad_clip_norm
        self._bind_params(params)
        # Read while the policy's extractors are resolved, so set before
        # BaseAlgorithm.__init__ runs.
        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self.critic_encoder_config = critic_encoder_config
        self.encoder_sharing = encoder_sharing
        super().__init__(env=env, **on_policy_kwargs)
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive or None, got {grad_clip_norm}.")
        if self.advantage_normalization not in ("minibatch", "rollout"):
            raise ValueError(
                f"{type(self).__name__}.advantage_normalization must be 'minibatch' or "
                f"'rollout', got {self.advantage_normalization!r}."
            )

        self.actor_mlp_dims = list(actor_mlp_dims) if actor_mlp_dims is not None else [256, 256, 256]
        self.critic_mlp_dims = list(critic_mlp_dims) if critic_mlp_dims is not None else [256, 256, 256]

    # --- subclass contract ---

    @abstractmethod
    def _sample_rollout_chain(
        self, obs: Obs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One rollout step's sampled chains, called under ``no_grad`` with
        ``obs`` on the policy device: ``(chain (N, K+1, H, A),
        old_log_probs (N, K, H, A), env_actions)``, ``old_log_probs`` the
        elementwise log-prob of each chain transition and ``env_actions``
        what ``env.step`` receives (before ``clamp_action``)."""

    @abstractmethod
    def _chain_policy_terms(self, batch: ChainMinibatch) -> ChainPolicyTerms:
        """The current policy's log-ratio on ``batch`` (differentiable in the
        actor), the advantages the surrogate uses and the clip coefficient:
        everything specific to the chain (how per-element log-probs reduce
        to one per pair, a per-chain-step advantage discount or clip
        schedule)."""

    # --- model state ---

    def _build_chain_training_state(
        self,
        actor_parameters: Iterable[torch.nn.Parameter],
        critic_parameters: Iterable[torch.nn.Parameter],
        *,
        chain_steps: int,
        horizon_steps: int,
        action_dim: int,
    ) -> None:
        """Build, after ``self.policy``: ``actor_optimizer`` over
        ``actor_parameters`` (also the parameters the actor gradient clip
        covers), ``critic_optimizer`` over ``critic_parameters``, their LR
        schedulers, the rollout buffer and the chain buffer of ``K =
        chain_steps`` trainable chain steps of ``(horizon_steps,
        action_dim)``."""
        self.actor_optimizer = make_optimizer(
            list(actor_parameters),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=True,
        )
        self.critic_optimizer = make_optimizer(
            list(critic_parameters),
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
            observation_space=self.env.single_observation_space,
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
            ft_denoising_steps=chain_steps,
            horizon_steps=horizon_steps,
            action_dim=action_dim,
            device=self.device,
        )

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("actor_optimizer", "critic_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "actor_mlp_dims": self.actor_mlp_dims,
            "critic_mlp_dims": self.critic_mlp_dims,
            **encoder_checkpoint_metadata(self),
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "lr_scheduler_states": [
                sched.state_dict() if sched is not None else None
                for sched in self._lr_schedulers
            ]
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        for sched, sched_state in zip(self._lr_schedulers, state.get("lr_scheduler_states", [])):
            if sched is not None and sched_state is not None:
                sched.load_state_dict(sched_state)

    # --- rollout ---

    def _snapshot_window_initial_hidden(self, hidden, next_done: torch.Tensor):
        # The one call site that runs right after rollout_buffer.reset().
        self._chain_buffer.reset()
        return hidden

    def _rollout_step(self, obs, hidden, episode_starts: torch.Tensor):
        """``(env_actions, values, log_probs, None, hidden)``: the rollout
        buffer's ``log_probs`` is the mean chain log-prob (the update reads
        the chain buffer's, not it); a chain policy has no entropy."""
        del episode_starts
        with torch.no_grad():
            obs_t = self._obs_to_policy_device(obs)
            chain, old_log_probs, actions = self._sample_rollout_chain(obs_t)
            values = self.policy.predict_values(obs_t)
        self._chain_buffer.add(chain, old_log_probs)
        log_probs = old_log_probs.mean(dim=(1, 2, 3))
        return actions, values, log_probs, None, hidden

    def _predict_last_values(self, obs, hidden) -> torch.Tensor:
        return self.policy.predict_values(obs)

    # --- update ---

    def train(self) -> dict[str, float]:
        self.policy.train()
        train_actor = self._global_update >= self.critic_warmup_updates
        self._global_update += 1

        chain_buffer = self._chain_buffer
        k = chain_buffer.ft_denoising_steps
        step_shape = (chain_buffer.horizon_steps, chain_buffer.action_dim)
        total = self.num_steps * self.num_envs

        obs_flat = flatten_leading_dims(self.rollout_buffer.obs)
        returns_flat = self.rollout_buffer.returns.reshape(total)
        values_flat = self.rollout_buffer.values.reshape(total)
        advantages_flat = self.rollout_buffer.advantages.reshape(total)
        if self.advantage_normalization == "rollout":
            advantages_flat = self._normalized_advantages(advantages_flat)
        chains_flat = chain_buffer.chains.reshape(total, k + 1, *step_shape)
        old_log_probs_flat = chain_buffer.old_log_probs.reshape(total, k, *step_shape)

        total_pairs = total * k
        num_batches = max(1, total_pairs // self.update_batch_size)
        metrics_sum: dict[str, float] = {}
        num_updates = 0
        stop = False
        for _ in range(self.update_epochs):
            perm = torch.randperm(total_pairs, device=self.device)
            for b in range(num_batches):
                idx = perm[b * self.update_batch_size : (b + 1) * self.update_batch_size]
                batch_inds, step_inds = torch.unravel_index(idx, (total, k))
                advantages = advantages_flat[batch_inds]
                if self.advantage_normalization == "minibatch":
                    advantages = self._normalized_advantages(advantages)
                batch = ChainMinibatch(
                    obs=index_obs(obs_flat, batch_inds),
                    chains_prev=chains_flat[batch_inds, step_inds],
                    chains_next=chains_flat[batch_inds, step_inds + 1],
                    chain_step_inds=step_inds,
                    old_log_probs=old_log_probs_flat[batch_inds, step_inds],
                    advantages=advantages,
                    returns=returns_flat[batch_inds],
                    old_values=values_flat[batch_inds],
                )
                loss, info = self._chain_minibatch_loss(batch)
                self._chain_optimizer_step(loss, train_actor)

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
        metrics["explained_variance"] = explained_variance(values_flat, returns_flat)
        return metrics

    def _normalized_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """``advantages`` (1-D: the rollout's env steps, or one minibatch's
        pairs, per ``advantage_normalization``) normalized (``norm_adv``) and
        clamped to their own quantiles."""
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        return advantages.clamp(min=adv_min, max=adv_max)

    def _chain_minibatch_loss(
        self, batch: ChainMinibatch
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``pg_loss + vf_coef * v_loss`` of one minibatch and its metrics
        (``loss``, ``pg_loss``, ``v_loss``, ``clipfrac``, ``approx_kl``,
        ``ratio``)."""
        terms = self._chain_policy_terms(batch)
        approx_kl, clipfrac = approx_kl_clipfrac(terms.logratio, terms.ratio, terms.clip_coef)
        pg_loss = ppo_clip_policy_loss(terms.advantages, terms.ratio, terms.clip_coef)
        new_values = self.policy.predict_values(batch.obs)
        v_loss = clipped_value_loss(
            new_values, batch.old_values, batch.returns, self.clip_vloss_coef
        )
        loss = pg_loss + self.vf_coef * v_loss
        info = {
            "loss": float(loss.detach().item()),
            "pg_loss": float(pg_loss.detach().item()),
            "v_loss": float(v_loss.detach().item()),
            "clipfrac": clipfrac,
            "approx_kl": float(approx_kl.item()),
            "ratio": float(terms.ratio.mean().item()),
        }
        return loss, info

    def _chain_optimizer_step(self, loss: torch.Tensor, train_actor: bool) -> None:
        """One backward of ``loss``; the critic optimizer always steps, the
        actor optimizer only when ``train_actor`` (after the critic warmup),
        with its parameters' gradient norm clipped to ``grad_clip_norm``."""
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_actor:
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in self.actor_optimizer.param_groups for p in group["params"]],
                    self.grad_clip_norm,
                )
            self.actor_optimizer.step()
        self.critic_optimizer.step()


__all__ = ["ChainMinibatch", "ChainPPOCore", "ChainPolicyTerms"]
