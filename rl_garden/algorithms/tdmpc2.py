"""``TDMPC2``: implicit world model + CEM/MPPI planner, ported from
``3rd_party/tdmpc2/tdmpc2/{tdmpc2.py,trainer/online_trainer.py}``.

Scope for this port (see ``docs/superpowers/specs`` design discussion this
was brainstormed from):

- **Single-task only.** No task embedding / action masking / per-task
  discount -- nothing else in rl-garden needs multitask training yet.
- **``num_envs == 1`` / ``num_eval_envs == 1`` required.** The CEM planner
  already rolls out ``num_samples`` (hundreds) of trajectories per env step;
  multiplying that by vectorized envs is an untested configuration upstream
  never validated. Vectorized rollout is a deliberate future extension, not
  attempted here.
- **``episodic=False`` only.** Kept as a TDMPC2-level restriction (see
  ``episodic`` docs below) even though ``SequenceReplayBuffer``'s
  strict-mode tail-step fix (model-based-base plan 1.6) now lets a sampled
  window's *last* position carry a true terminal transition -- re-enabling
  ``episodic=True`` end to end is a separate, not-yet-done change.
- **Joint world-model optimizer kept, as upstream does** (encoder + latent
  projection + dynamics + reward + termination + Q share one Adam; ``pi`` has
  its own): every other rl-garden algorithm keeps separate optimizers per
  network, but TD-MPC2's consistency loss backprops through all of those
  heads via one shared latent rollout in a single ``backward()`` call --
  splitting that into per-network optimizers would need multiple backward
  passes or manual gradient accumulation, diverging from upstream and adding
  bug surface for no behavioral benefit. ``_update_model`` folds the critic's
  value loss into this same optimizer/backward call too -- a TD-MPC2-specific
  choice (``ModelBasedAlgorithm``'s three-hook split doesn't require it, see
  that class's docstring); "model update" here means "model + critic".

**Model-based-base plan 1.5/1.7**: this class is ``ModelBasedAlgorithm``
(``OffPolicyAlgorithm``)'s first concrete consumer -- its rollout, replay
buffer wiring, checkpointing, and eval loop are the same vectorized
``training_freq``/``utd``-driven loop every other online algorithm uses,
configured with ``training_freq=1, utd=1`` (exactly one gradient step per
env step, matching upstream) and a ``num_envs == 1`` assertion below (the
CEM planner's own per-step trajectory count makes vectorized rollout a
separate, deliberately deferred extension, decision kept from the prior
single-purpose loop this replaces). ``_rollout_action`` still can't just
delegate to the inherited default (``self.policy.predict()``): TD-MPC2's own
planner-state bookkeeping (``prev_mean``/``t0``) must stay separate from
``TDMPC2Policy.predict()``'s (which ``_evaluate()`` also calls, interleaved
with rollout within one ``learn()`` call) -- sharing one mutable state
between the two would let an eval episode's planning corrupt an in-progress
training episode's warm-started mean, and vice versa. See
``_rollout_action``'s own docstring.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np
import torch

from rl_garden.algorithms.model_based import ModelBasedAlgorithm
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.obs_utils import flatten_leading_dims, index_obs
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks.twohot import soft_ce
from rl_garden.observations import ObsGroups
from rl_garden.planners.mppi import PlannerConfig
from rl_garden.policies.tdmpc2_policy import TDMPC2Policy
from rl_garden.world_models.base import State
from rl_garden.world_models.latent_consistency import LatentConsistencyModel


def _compute_discount(
    episode_length: int, discount_denom: float, discount_min: float, discount_max: float
) -> float:
    frac = episode_length / discount_denom
    return min(max((frac - 1) / frac, discount_min), discount_max)


class TDMPC2(ModelBasedAlgorithm):
    _compatible_checkpoint_algorithms = ("TDMPC2",)
    # One shared encoder feeds both the policy head and the Q-heads through a
    # single joint world-model optimizer (see module docstring) -- there is
    # no separate actor/critic split to stop-gradient between, so "shared"
    # (both paths train it, no detach) is the only sharing mode that makes
    # sense here.
    encoder_sharing = "shared"

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        episode_length: int = 100,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        seed_steps: Optional[int] = None,
        # planning
        use_planner: bool = True,
        horizon: int = 3,
        num_samples: int = 512,
        num_elites: int = 64,
        num_pi_trajs: int = 24,
        iterations: int = 6,
        min_std: float = 0.05,
        max_std: float = 2.0,
        temperature: float = 0.5,
        # architecture
        latent_dim: int = 512,
        mlp_dim: int = 512,
        simnorm_dim: int = 8,
        num_q: int = 5,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        dropout: float = 0.01,
        episodic: bool = False,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        entropy_coef: float = 1e-4,
        # optimization
        lr: float = 3e-4,
        enc_lr_scale: float = 0.3,
        grad_clip_norm: float = 20.0,
        tau: float = 0.01,
        rho: float = 0.5,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        termination_coef: float = 1.0,
        discount_denom: float = 5.0,
        discount_min: float = 0.95,
        discount_max: float = 0.995,
        # Dict-obs (pixel) encoder config -- ignored for Box observations.
        encoder_config: Optional[EncoderConfig] = None,
        obs_groups: Optional[ObsGroups] = None,
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
        # Computed before super().__init__() -- both feed OffPolicyAlgorithm
        # constructor kwargs (gamma, learning_starts) below.
        discount = _compute_discount(episode_length, discount_denom, discount_min, discount_max)
        learning_starts = (
            seed_steps if seed_steps is not None else max(1_000, 5 * episode_length)
        )

        super().__init__(
            env=env,
            eval_env=eval_env,
            buffer_size=buffer_size,
            buffer_device=buffer_device,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=discount,
            tau=tau,
            # Exactly one gradient step per env step past learning_starts
            # (upstream semantics) -- num_envs == 1 asserted below, so
            # steps_per_env == grad_steps_per_iteration == 1.
            training_freq=1,
            utd=1,
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

        if self.env.num_envs != 1:
            raise ValueError(
                f"TDMPC2 requires env.num_envs == 1 (vectorized rollout is not "
                f"supported in this port), got {self.env.num_envs}."
            )
        if self.eval_env is not None and self.eval_env.num_envs != 1:
            raise ValueError(
                f"TDMPC2 requires eval_env.num_envs == 1, got {self.eval_env.num_envs}."
            )
        if episodic:
            raise NotImplementedError(
                "episodic=True is not supported by this port yet: "
                "SequenceReplayBuffer's strict mode now lets a sampled window's "
                "last position carry a true terminal transition (model-based-"
                "base plan 1.6's tail-step fix), but every non-tail position "
                "inside a window is still rejected on any boundary, so the "
                "termination classifier would still be starved of positive "
                "examples relative to upstream's per-episode-block storage. "
                "Use the default episodic=False."
            )

        self.episode_length = episode_length
        self.seed_steps = seed_steps
        self.discount = discount

        self.use_planner = use_planner
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_pi_trajs = num_pi_trajs
        self.iterations = iterations
        self.min_std = min_std
        self.max_std = max_std
        self.temperature = temperature

        self.latent_dim = latent_dim
        self.mlp_dim = mlp_dim
        self.simnorm_dim = simnorm_dim
        self.num_q = num_q
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.dropout = dropout
        self.episodic = episodic
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.entropy_coef = entropy_coef

        self.lr = lr
        self.enc_lr_scale = enc_lr_scale
        self.grad_clip_norm = grad_clip_norm
        self.rho = rho
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.termination_coef = termination_coef
        self.discount_denom = discount_denom
        self.discount_min = discount_min
        self.discount_max = discount_max

        self.encoder_config = encoder_config
        self.obs_groups = obs_groups
        self._image_augmentation_seed = image_augmentation_seed

        self._setup_model()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_world_model_encoder(self) -> BaseFeaturesExtractor:
        return self.observation_encoders.actor

    def _setup_model(self) -> None:
        self._resolve_observation_encoders(
            self.env.single_observation_space, augmentation_seed=self._image_augmentation_seed
        )
        action_dim = int(np.prod(self.env.single_action_space.shape))
        world_model_encoder = self._build_world_model_encoder()

        # NOTE: no `.to(self.device)` here -- deliberately deferred to the
        # single `self.policy = TDMPC2Policy(...).to(self.device)` call
        # below, after every module (this model's + the policy's
        # actor/critic/critic_target) has been constructed AND initialized.
        # See TDMPC2Policy.__init__'s docstring: initialization must happen
        # on CPU-constructed (default-device) tensors, in the exact order
        # upstream's single combined init pass used, before anything moves
        # to `self.device` -- moving this model alone first would run its
        # `apply_init()` (triggered from inside TDMPC2Policy.__init__) on a
        # different device's RNG stream than upstream's CPU-then-move order.
        world_model = LatentConsistencyModel(
            encoder=world_model_encoder,
            action_dim=action_dim,
            latent_dim=self.latent_dim,
            mlp_dim=self.mlp_dim,
            simnorm_dim=self.simnorm_dim,
            num_bins=self.num_bins,
            vmin=self.vmin,
            vmax=self.vmax,
            episodic=self.episodic,
            rho=self.rho,
        )

        self.planner_cfg = PlannerConfig(
            action_dim=action_dim,
            discount=self.discount,
            horizon=self.horizon,
            num_samples=self.num_samples,
            num_elites=self.num_elites,
            num_pi_trajs=self.num_pi_trajs,
            iterations=self.iterations,
            min_std=self.min_std,
            max_std=self.max_std,
            temperature=self.temperature,
        )

        self.policy = TDMPC2Policy(
            self.env.single_observation_space,
            self.env.single_action_space,
            world_model,
            self.planner_cfg,
            mlp_dim=self.mlp_dim,
            num_q=self.num_q,
            dropout=self.dropout,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            tau=self.tau,
            use_planner=self.use_planner,
        ).to(self.device)
        world_model = self.policy.world_model  # now on self.device

        param_groups = world_model.parameter_groups()
        enc_params = list(param_groups["encoder"])
        other_params = list(param_groups["model"]) + list(self.policy.critic.parameters())
        self.world_optimizer = torch.optim.Adam(
            [
                {"params": enc_params, "lr": self.lr * self.enc_lr_scale},
                {"params": other_params, "lr": self.lr},
            ]
        )
        self.pi_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr, eps=1e-5)

        self.replay_buffer = SequenceReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=1,
            buffer_size=self.buffer_size,
            cross_episode=False,
            horizon=self.horizon,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    # ------------------------------------------------------------------
    # Rollout: own planner prev_mean/t0 bookkeeping (see class docstring on
    # why this can't just delegate to the inherited _rollout_action, which
    # would go through self.policy.predict() -- shared, wrong, state with
    # _evaluate()).
    # ------------------------------------------------------------------

    def _on_env_reset(self, obs) -> None:
        super()._on_env_reset(obs)
        self._rollout_prev_mean: Optional[torch.Tensor] = None
        self._rollout_t0 = True

    def _post_rollout_step(self, action_context, terminations, truncations, infos) -> None:
        super()._post_rollout_step(action_context, terminations, truncations, infos)
        if bool((terminations | truncations).any().item()):
            self._rollout_prev_mean = None
            self._rollout_t0 = True

    def _rollout_action(self, obs, learning_has_started: bool):
        """Planner-driven (or, with ``use_planner=False``, ``pi``-sampled)
        training-rollout action, threading this algorithm's OWN
        ``self._rollout_prev_mean``/``self._rollout_t0`` through
        ``MPPIPlanner.plan()`` directly instead of going through
        ``self.policy.predict()`` -- see class docstring."""
        if not learning_has_started:
            action = self._explore_action(obs)
        elif self.use_planner:
            obs_device = self._obs_to_policy_device(obs)
            embed = self.policy.world_model.encode(obs_device)
            is_first = torch.full(
                (embed.shape[0],), self._rollout_t0, dtype=torch.bool, device=embed.device
            )
            state = self.policy.world_model.observe(None, None, embed, is_first)
            action, self._rollout_prev_mean = self.policy.planner.plan(
                self.policy.world_model,
                state,
                policy_prior=self.policy.policy_prior,
                value_fn=self.policy.value_fn,
                prev_mean=self._rollout_prev_mean,
                t0=self._rollout_t0,
                eval_mode=False,
            )
            action = action.unsqueeze(0)
        else:
            with torch.no_grad():
                obs_device = self._obs_to_policy_device(obs)
                z = self.policy.world_model.encode(obs_device)
                action, _ = self.policy.pi(z)
        self._rollout_t0 = False
        return action, action, None

    # ------------------------------------------------------------------
    # Gradient step: ModelBasedAlgorithm's three hooks.
    # ------------------------------------------------------------------

    def _encode_window(self, obs_window: Obs, window_len: int, batch_size: int) -> torch.Tensor:
        flat = flatten_leading_dims(obs_window)
        z_flat = self.policy.world_model.encode(flat)
        return z_flat.reshape(window_len, batch_size, -1)

    def _td_target(
        self, next_z: torch.Tensor, reward: torch.Tensor, terminated: torch.Tensor
    ) -> torch.Tensor:
        action, _ = self.policy.pi(next_z)
        return reward + self.discount * (1 - terminated) * self.policy.Q(
            next_z, action, return_type="min", target=True
        )

    def _update_model(self, batch) -> tuple[dict[str, float], State]:
        """Consistency + reward + termination + **value** losses in one
        joint ``world_optimizer`` backward pass (TD-MPC2-specific: the
        critic trains in the same backward as the model, see class
        docstring). Returns the LIVE posterior ``{"z": zs}`` (plan item 0);
        ``_update_actor_critic`` detaches it before the actor update."""
        world_model = self.policy.world_model
        obs, action, reward, terminated = batch.obs, batch.action, batch.reward, batch.terminated
        horizon, batch_size = action.shape[0], action.shape[1]
        reward = reward.unsqueeze(-1)
        terminated_f = terminated.float().unsqueeze(-1)

        with torch.no_grad():
            next_obs = index_obs(obs, slice(1, None))
            next_z = self._encode_window(next_obs, horizon, batch_size)
            td_targets = self._td_target(next_z, reward, terminated_f)

        model_batch = {"obs": obs, "action": action, "reward": reward, "next_z": next_z}
        if self.episodic:
            model_batch["terminated"] = terminated_f
        losses, posterior = world_model.model_loss(model_batch)

        zs_live = posterior["z"]
        _zs = zs_live[:-1]
        qs = self.policy.Q(_zs, action, return_type="all")

        value_loss = torch.zeros((), device=self.device)
        for t in range(horizon):
            for qi in range(self.num_q):
                value_loss = value_loss + soft_ce(
                    qs[qi, t], td_targets[t], self.num_bins, self.vmin, self.vmax, world_model.bin_size
                ).mean() * self.rho**t
        value_loss = value_loss / (horizon * self.num_q)

        total_loss = (
            self.consistency_coef * losses["consistency_loss"]
            + self.reward_coef * losses["reward_loss"]
            + self.termination_coef * losses["termination_loss"]
            + self.value_coef * value_loss
        )

        self.world_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(world_model.parameters()) + list(self.policy.critic.parameters()), self.grad_clip_norm
        )
        self.world_optimizer.step()

        metrics = {
            "consistency_loss": float(losses["consistency_loss"].detach()),
            "reward_loss": float(losses["reward_loss"].detach()),
            "value_loss": float(value_loss.detach()),
            "termination_loss": float(losses["termination_loss"].detach()) if self.episodic else 0.0,
            "total_loss": float(total_loss.detach()),
            "grad_norm": float(grad_norm),
        }
        return metrics, posterior

    def _update_pi(self, zs: torch.Tensor) -> dict[str, float]:
        action, info = self.policy.pi(zs)
        qs = self.policy.Q(zs, action, return_type="avg", detach=True)
        self.policy.scale.update(qs[0])
        qs = self.policy.scale(qs)

        rho_pows = self.rho ** torch.arange(zs.shape[0], device=zs.device)
        pi_loss = (
            -(self.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho_pows
        ).mean()

        self.pi_optimizer.zero_grad(set_to_none=True)
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_clip_norm)
        self.pi_optimizer.step()

        return {
            "pi_loss": float(pi_loss.detach()),
            "pi_grad_norm": float(pi_grad_norm),
            "pi_scale": float(self.policy.scale.value.item()),
        }

    def _update_actor_critic(self, batch, posterior: State) -> dict[str, float]:
        # .detach(): posterior["z"] is live (plan item 0) and the world model
        # has already been updated in _update_model() -- the actor update
        # must not also backprop into (now-stale) world-model gradients,
        # matching upstream's update_pi(zs.detach(), ...).
        return self._update_pi(posterior["z"].detach())

    def _update_targets(self) -> None:
        self.policy.soft_update_target_Q()

    # ------------------------------------------------------------------
    # Eval hooks (see BaseAlgorithm._evaluate() -- reused unmodified)
    # ------------------------------------------------------------------

    def _eval_start_hook(self) -> None:
        self.policy.reset_episode()

    def _eval_step_hook(
        self,
        obs_before,
        critic_action: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos: dict,
    ) -> None:
        del obs_before, critic_action, rewards, infos
        done = bool((terminations | truncations).any().item())
        self.policy.notify_step_done(done)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("world_optimizer", "pi_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "episode_length": self.episode_length,
            "horizon": self.horizon,
            "num_samples": self.num_samples,
            "num_elites": self.num_elites,
            "num_pi_trajs": self.num_pi_trajs,
            "iterations": self.iterations,
            "latent_dim": self.latent_dim,
            "mlp_dim": self.mlp_dim,
            "num_q": self.num_q,
            "num_bins": self.num_bins,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "episodic": self.episodic,
            "discount_denom": self.discount_denom,
            "discount_min": self.discount_min,
            "discount_max": self.discount_max,
            "lr": self.lr,
            "enc_lr_scale": self.enc_lr_scale,
            "grad_clip_norm": self.grad_clip_norm,
            "rho": self.rho,
            "consistency_coef": self.consistency_coef,
            "reward_coef": self.reward_coef,
            "value_coef": self.value_coef,
            "termination_coef": self.termination_coef,
            "dropout": self.dropout,
            "log_std_min": self.log_std_min,
            "log_std_max": self.log_std_max,
            "entropy_coef": self.entropy_coef,
            "simnorm_dim": self.simnorm_dim,
            "use_planner": self.use_planner,
            "seed_steps": self.seed_steps,
            "encoder_sharing": self.encoder_sharing,
            "encoder_sharing_origin": self.encoder_sharing_origin,
            "encoder_config": (
                dataclasses.asdict(self.encoder_config) if self.encoder_config is not None else None
            ),
            "obs_groups": (
                dataclasses.asdict(self.obs_groups) if self.obs_groups is not None else None
            ),
            "image_augmentation_seed": self._image_augmentation_seed,
        }

    # ------------------------------------------------------------------
    # learning_starts pretrain burst (upstream seed_steps semantics) --
    # see OffPolicyAlgorithm._on_learning_starts's docstring.
    # ------------------------------------------------------------------

    def _on_learning_starts(self) -> Optional[dict[str, float]]:
        return self.train(self.learning_starts, compute_info=True)
