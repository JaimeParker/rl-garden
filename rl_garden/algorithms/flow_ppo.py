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
``FlowPPO`` and ``DPPO`` share only the chain-PPO engineering,
``ChainPPOCore`` (joint ``(env_step, chain_step)``-minibatch PPO update via
``torch.unravel_index``, the two optimizers, the checkpoint hooks) -- not
DPPO's diffusion-specific content.

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
  A training minibatch's ``(env_step, flow_step)`` pairs are a duplicated
  sample of the per-env-step advantage population, so advantage
  normalization and quantile clamping are computed once over that
  population (``advantage_normalization = "rollout"``; DPPO keeps its
  reference's per-minibatch normalization).
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

from typing import Any, Optional

import torch
from gymnasium import spaces

from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.chain_ppo import FlowPPOParams
from rl_garden.algorithms.chain_ppo_core import (
    ChainMinibatch,
    ChainPolicyTerms,
    ChainPPOCore,
)
from rl_garden.algorithms.observation_encoders import EncoderSharing
from rl_garden.common.logger import Logger
from rl_garden.common.types import Obs
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObsGroups
from rl_garden.policies.flow_ppo_policy import FlowPPOPolicy


class FlowPPO(ChainPPOCore):
    """PPO over the ``flow_steps`` stochastic SDE steps of a flow-matching
    actor, on ``ChainPPOCore``. FlowPPO-specific terms
    (``_chain_policy_terms``): each step's log-prob is the mean of its
    per-dimension terms (``logprob_mode``), the advantage is the env step's
    for every flow step, the clip coefficient a constant ``clip_coef``."""

    params_class = FlowPPOParams

    _compatible_checkpoint_algorithms = ("FlowPPO",)
    advantage_normalization = "rollout"
    # No class-level override: FlowPPOPolicy now has the full actor/critic
    # extractor contract (see FlowPPOPolicy/_setup_model below), so
    # ObservationEncoderMixin's "shared_critic_grad" default applies like
    # every other critic-bearing algorithm -- despite FlowPPO being
    # on-policy, "shared_critic_grad" (not "shared") matches its original,
    # still-correct semantics: the critic loss trains the shared encoder,
    # the actor's log-prob path does not (BasePolicy.extract_actor_features's
    # stop-gradient rule, not an algorithm-level ad hoc detach).

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        num_steps: int = 50,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
        critic_encoder_config: Optional[EncoderConfig] = None,
        encoder_sharing: Optional[EncoderSharing] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 25,
        num_eval_steps: int = 50,
        finite_horizon_gae: bool = False,
        bootstrap_at_done: str = "truncated",
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
        params: FlowPPOParams | None = None,
        **overrides: Any,
    ) -> None:
        params = resolve_params(self.params_class, params, overrides)
        horizon_length = params.horizon_length
        flow_steps = params.flow_steps
        if horizon_length < 1:
            raise ValueError(f"horizon_length must be >= 1, got {horizon_length}.")
        if flow_steps <= 0:
            raise ValueError(f"flow_steps must be positive, got {flow_steps}.")
        super().__init__(
            env,
            params=params,
            encoder_config=encoder_config,
            obs_groups=obs_groups,
            critic_encoder_config=critic_encoder_config,
            encoder_sharing=encoder_sharing,
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
            bootstrap_at_done=bootstrap_at_done,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_final_checkpoint=save_final_checkpoint,
        )

        expected_action_shape = (self.horizon_length,) + env.single_action_space.shape[1:]
        if env.single_action_space.shape != expected_action_shape:
            raise ValueError(
                "FlowPPO expects env.single_action_space to already be chunked by "
                "ActionChunkWrapper(env, act_steps=horizon_length); got shape "
                f"{env.single_action_space.shape}, expected {expected_action_shape}."
            )

        self._setup_model()

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
        }

    def _setup_model(self) -> None:
        obs_space = self.env.single_observation_space
        raw_action_space = spaces.Box(
            low=self.env.single_action_space.low[0],
            high=self.env.single_action_space.high[0],
            shape=self.env.single_action_space.shape[1:],
            dtype=self.env.single_action_space.dtype,
        )
        self.policy = FlowPPOPolicy(
            observation_space=obs_space,
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
            **self._policy_extractor_kwargs(obs_space),
        ).to(self.device)

        # Actor-role extractor trains from the actor optimizer only when it
        # is genuinely its own (encoder_sharing="separate"): under
        # "shared"/"shared_critic_grad" it is the same object the critic
        # optimizer below already covers, and extract_actor_features's
        # stop-gradient rule (not optimizer membership) is what isolates the
        # actor loss from it under "shared_critic_grad".
        actor_params = list(self.policy.actor.parameters())
        if self.policy.critic_extractor is not None:
            actor_params += list(self.policy.actor_extractor.parameters())
        critic_role_extractor = (
            self.policy.critic_extractor
            if self.policy.critic_extractor is not None
            else self.policy.actor_extractor
        )
        self._build_chain_training_state(
            actor_params,
            list(self.policy.critic.parameters()) + list(critic_role_extractor.parameters()),
            chain_steps=self.flow_steps,
            # The chunk is one flat action vector (FlowPPOPolicy.action_dim).
            horizon_steps=1,
            action_dim=self.policy.action_dim,
        )

    # --- ChainPPOCore hooks ---

    def _sample_rollout_chain(
        self, obs: Obs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The stochastic SDE trajectory and its per-step log-probs (with a
        horizon axis of 1) and the final action in chunk shape."""
        chain, log_probs, action = self.policy.sample_rollout_chain(obs)
        return chain.unsqueeze(2), log_probs.unsqueeze(2), self.policy.to_chunk_shape(action)

    def _chain_policy_terms(self, batch: ChainMinibatch) -> ChainPolicyTerms:
        # Each role extracts its own features via BasePolicy's contract:
        # extract_actor_features detaches under "shared_critic_grad" (the
        # single stop-gradient rule; SACPolicy's own convention), while the
        # critic loss (ChainPPOCore, through policy.predict_values) uses
        # extract_critic_features, which never detaches. Under "separate"
        # these are two different extractors; under "shared"/
        # "shared_critic_grad" it is the same one, called once per role.
        features_actor = self.policy.extract_actor_features(batch.obs)
        new_log_probs = self.policy.get_logprobs_subsample(
            features_actor,
            batch.chains_prev.flatten(1),
            batch.chains_next.flatten(1),
            batch.chain_step_inds,
        ).mean(dim=-1)
        old_log_probs = batch.old_log_probs.flatten(1).mean(dim=-1)
        logratio = new_log_probs - old_log_probs
        return ChainPolicyTerms(
            logratio=logratio,
            ratio=logratio.exp(),
            advantages=batch.advantages,
            clip_coef=self.clip_coef,
        )
