"""Uni-O4: Unifying Online and Offline RL with Multi-Step On-Policy Optimization.

Ported from ``3rd_party/Uni-O4`` (Lei et al., ICLR 2024, arXiv:2311.03351).
Builds on BPPO (``rl_garden/algorithms/bppo.py``): **one shared critic**
(``BPPOCriticMixin`` -- verified against ``3rd_party/Uni-O4/main.py``, which
builds exactly one ``value``/``Q_bc`` pair and passes it into
``abppo.joint_train(replay_buffer, value, ..., Q=Q, ...)``; the ensemble is
over actors only, NOT one critic per member) plus a diverse ensemble of
``num_policies`` BC/BPPO actors, sharing one ``_phase_step`` counter across
three sequential phases:

* **Phase CRITIC** (``_phase_step < critic_warmup_steps``): identical to
  BPPO's Phase A, via ``BPPOCriticMixin._phase_a_step``.
* **Phase BC_ENSEMBLE** (``critic_warmup_steps <= _phase_step <
  critic_warmup_steps + bc_ensemble_steps``): joint BC training of all N
  actors with a cross-policy diversity term, faithful port of
  ``BC_ensemble.joint_train``/``BehaviorCloning.joint_loss``
  (``3rd_party/Uni-O4/BC_ensemble.py``) at upstream's own default
  combination (``bc_kl='data'``, ``kl_type='heuristic'``, ``alpha_bc=0.1``
  -- nonzero by default upstream, so this diversity term is genuinely
  exercised). **Deliberate simplification vs. upstream**: recomputes every
  other member's log-prob of the dataset action fresh, every member, every
  step, instead of porting upstream's incremental ``all_prob_a[pre_id]``
  cache (order-dependent, shuffled-per-step bookkeeping that's easy to get
  subtly wrong -- a wrong index silently compares a member against the
  wrong peer, no crash, just degraded diversity). At ``num_policies=4``
  this is 12 extra small MLP forward passes per step, not a meaningful
  cost -- this trades nothing for removing a real correctness-risk class.
* **Phase IMPROVE** (``_phase_step >= critic_warmup_steps +
  bc_ensemble_steps``): V/Q frozen; each member independently runs BPPO's
  PPO-clip update (``ppo_clip_policy_loss``) against the shared critic,
  using one shared sampled batch per ``train()`` call (matches
  ``abppo.joint_train``). Cross-ensemble KL is out of scope for this port
  -- upstream's own default is off (``is_kl_update=False``). The
  clip-ratio decay reuses ``BPPO``'s multiplicative-then-freeze schedule
  (``clip_ratio *= clip_decay`` for the first ``clip_decay_steps`` steps)
  rather than porting upstream's own default *linear* decay
  (``clip_ratio * (1 - step/bppo_steps)``, needing the total Phase-IMPROVE
  step budget up front) -- a deliberate simplification to reuse BPPO's
  already-validated decay mechanism instead of introducing a second one.

``self.policy`` is a ``UniO4MixturePolicy``
(``rl_garden/policies/unio4_mixture_policy.py``) wrapping all N actors via
greedy self-confidence selection -- both the eval/inference entry point
``BaseAlgorithm`` expects and, since it's an ``nn.ModuleList`` under the
hood, the natural checkpoint-serialization container for all N actors'
weights.
"""
from __future__ import annotations

import warnings
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.algorithms.bppo import BPPOCriticMixin
from rl_garden.algorithms.offline import (
    OfflineEnvSpec,
    OfflineRLAlgorithm,
    run_exact_episode_eval,
)
from rl_garden.algorithms.ppo import ppo_clip_policy_loss
from rl_garden.buffers.sarsa_buffer import SarsaMCTensorReplayBuffer
from rl_garden.common.logger import Logger
from rl_garden.common.optim import make_optimizer
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.networks import KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.policies.bc_policy import BCPolicy
from rl_garden.policies.unio4_mixture_policy import UniO4MixturePolicy


class UniO4(BPPOCriticMixin, OfflineRLAlgorithm):
    """Uni-O4: BC-ensemble + shared-critic PPO-clip improvement. Box obs only."""

    _compatible_checkpoint_algorithms = ("UniO4",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        gamma: float = 0.99,
        # -- Phase CRITIC: shared V/Q warmup --
        critic_warmup_steps: int = 2_000_000,
        value_lr: float = 1e-4,
        q_lr: float = 1e-4,
        tau: float = 0.005,
        target_update_freq: int = 2,
        value_hidden_dims: Sequence[int] = (512, 512, 512),
        q_hidden_dims: Sequence[int] = (1024, 1024),
        # -- Phase BC_ENSEMBLE: joint BC training with diversity term --
        num_policies: int = 4,
        bc_ensemble_steps: int = 400_000,
        alpha_bc: float = 0.1,
        # -- Phase IMPROVE: per-member PPO-clip against the shared critic --
        actor_lr: float = 1e-4,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        clip_ratio: float = 0.25,
        clip_decay: float = 0.96,
        clip_decay_steps: int = 200,
        entropy_weight: float = 0.0,
        omega: float = 0.7,
        weight_decay: float = 0.0,
        use_adamw: bool = False,
        grad_clip_norm: Optional[float] = None,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 32,
        dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        eval_freq: int = 0,
        num_eval_steps: int = 50,
        num_eval_episodes: int = 10,
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
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=eval_freq,
            num_eval_steps=num_eval_steps,
            num_eval_episodes=num_eval_episodes,
            eval_env=eval_env,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=save_replay_buffer,
            save_final_checkpoint=save_final_checkpoint,
        )
        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                f"UniO4 supports only Box observation spaces, got {type(obs_space)}"
            )
        if critic_warmup_steps < 0:
            raise ValueError(
                f"critic_warmup_steps must be non-negative, got {critic_warmup_steps}."
            )
        if bc_ensemble_steps < 0:
            raise ValueError(
                f"bc_ensemble_steps must be non-negative, got {bc_ensemble_steps}."
            )
        if num_policies < 1:
            raise ValueError(f"num_policies must be >= 1, got {num_policies}.")
        if not (0.0 <= omega <= 1.0):
            raise ValueError(f"omega must be in [0, 1], got {omega}.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )

        self.critic_warmup_steps = critic_warmup_steps
        self.value_lr = value_lr
        self.q_lr = q_lr
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.value_hidden_dims = tuple(value_hidden_dims)
        self.q_hidden_dims = tuple(q_hidden_dims)
        self.num_policies = num_policies
        self.bc_ensemble_steps = bc_ensemble_steps
        self.alpha_bc = alpha_bc
        self.actor_lr = actor_lr
        self.actor_hidden_dims = tuple(actor_hidden_dims)
        self.clip_ratio_init = clip_ratio
        self.clip_decay = clip_decay
        self.clip_decay_steps = clip_decay_steps
        self.entropy_weight = entropy_weight
        self.omega = omega
        self.weight_decay = weight_decay
        self.use_adamw = use_adamw
        self.grad_clip_norm = grad_clip_norm
        self.use_layer_norm = use_layer_norm
        self.use_group_norm = use_group_norm
        self.num_groups = num_groups
        self.dropout_rate = dropout_rate
        self.kernel_init = kernel_init
        self.backbone_type = backbone_type
        self.std_parameterization = std_parameterization

        # Dedicated phase-gate/decay state, same reasoning as BPPO's (see
        # bppo.py) -- never reuses self._global_step/self._global_update.
        self._phase_step = 0
        self._phase_b_step = 0
        self._clip_ratio = clip_ratio
        self._best_eval_score = [float("-inf")] * num_policies

        self._setup_model()

    @property
    def _improve_phase_start(self) -> int:
        return self.critic_warmup_steps + self.bc_ensemble_steps

    # --- model setup ---

    def _build_actor_policy(self) -> BCPolicy:
        features_extractor = FlattenExtractor(
            observation_space=self.env.single_observation_space
        )
        return BCPolicy(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            features_extractor=features_extractor,
            net_arch=list(self.actor_hidden_dims),
            use_layer_norm=self.use_layer_norm,
            use_group_norm=self.use_group_norm,
            num_groups=self.num_groups,
            dropout_rate=self.dropout_rate,
            kernel_init=self.kernel_init,
            backbone_type=self.backbone_type,
            std_parameterization=self.std_parameterization,
            tanh_squash=False,
        ).to(self.device)

    def _build_replay_buffer(self) -> SarsaMCTensorReplayBuffer:
        return SarsaMCTensorReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _setup_model(self) -> None:
        actors = [self._build_actor_policy() for _ in range(self.num_policies)]
        old_actors = [self._build_actor_policy() for _ in range(self.num_policies)]
        for actor, old_actor in zip(actors, old_actors):
            old_actor.load_state_dict(actor.state_dict())
            for param in old_actor.parameters():
                param.requires_grad_(False)
        self.actors = nn.ModuleList(actors)
        self.old_actors = nn.ModuleList(old_actors)

        self._build_critic()

        for i, actor in enumerate(self.actors):
            setattr(
                self,
                f"actor_optimizer_{i}",
                make_optimizer(
                    list(actor.actor_parameters()),
                    lr=self.actor_lr,
                    weight_decay=self.weight_decay,
                    use_adamw=self.use_adamw,
                ),
            )

        self.policy = UniO4MixturePolicy(list(self.actors))
        self.replay_buffer = self._build_replay_buffer()

    # --- Phase BC_ENSEMBLE ---

    def _bc_ensemble_step(self, data) -> dict[str, float]:
        obs, actions = data.obs, data.actions
        metrics: dict[str, float] = {}

        if self.num_policies == 1:
            actor = self.actors[0]
            features = actor.extract_features(obs, stop_gradient=False)
            log_prob = actor.actor.evaluate_action_log_prob(features, actions).squeeze(-1)
            loss = (-log_prob).mean()
            optimizer = self.actor_optimizer_0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            metrics["bc_loss_0"] = float((-log_prob).detach().mean().item())
            return metrics

        with torch.no_grad():
            all_log_probs = [
                actor.actor.evaluate_action_log_prob(
                    actor.extract_features(obs), actions
                ).squeeze(-1)
                for actor in self.actors
            ]

        for i in range(self.num_policies):
            features_i = self.actors[i].extract_features(obs, stop_gradient=False)
            log_prob_i = self.actors[i].actor.evaluate_action_log_prob(
                features_i, actions
            ).squeeze(-1)
            bc_loss_i = -log_prob_i
            # bc_kl='data', kl_type='heuristic' -- upstream's own default
            # combination (BC_ensemble.joint_train/BehaviorCloning.joint_loss).
            others = torch.stack(
                [all_log_probs[j] for j in range(self.num_policies) if j != i], dim=0
            )
            max_prob_a = others.max(dim=0).values  # already detached (no_grad above)
            loss_i = (bc_loss_i - self.alpha_bc * (log_prob_i - max_prob_a)).mean()

            optimizer = getattr(self, f"actor_optimizer_{i}")
            optimizer.zero_grad(set_to_none=True)
            loss_i.backward()
            optimizer.step()

            metrics[f"bc_loss_{i}"] = float(bc_loss_i.detach().mean().item())
            metrics[f"joint_loss_{i}"] = float(loss_i.detach().item())
        return metrics

    # --- Phase IMPROVE ---

    def _weighted_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        if self.omega == 0.5:
            return advantage
        weight = torch.where(
            advantage > 0,
            torch.full_like(advantage, self.omega),
            torch.full_like(advantage, 1.0 - self.omega),
        )
        return weight * advantage

    def _improve_step(self, data) -> dict[str, float]:
        obs = data.obs
        metrics: dict[str, float] = {}
        for i in range(self.num_policies):
            with torch.no_grad():
                old_features = self.old_actors[i].extract_features(obs)
                action, old_log_prob = self.old_actors[i].actor.action_log_prob(
                    old_features
                )
                advantage = (self.q_net(obs, action) - self.value_net(obs)).squeeze(-1)
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                advantage = self._weighted_advantage(advantage)

            new_features = self.actors[i].extract_features(obs, stop_gradient=False)
            new_log_prob = self.actors[i].actor.evaluate_action_log_prob(
                new_features, action
            )
            ratio = (new_log_prob.squeeze(-1) - old_log_prob.squeeze(-1)).exp()

            clip_loss = ppo_clip_policy_loss(advantage, ratio, self._clip_ratio)
            entropy = self.actors[i].actor.entropy(new_features).squeeze(-1).mean()
            loss = clip_loss - self.entropy_weight * entropy

            optimizer = getattr(self, f"actor_optimizer_{i}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.actors[i].actor_parameters()), self.grad_clip_norm
                )
            optimizer.step()

            metrics[f"actor_loss_{i}"] = float(clip_loss.detach().item())
            metrics[f"entropy_{i}"] = float(entropy.detach().item())
            metrics[f"ratio_{i}"] = float(ratio.detach().mean().item())
        metrics["clip_ratio"] = self._clip_ratio
        return metrics

    # --- training loop ---

    def _sample_train_batch(self):
        return self.replay_buffer.sample(self.batch_size)

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        del compute_info
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch()
            if self._phase_step < self.critic_warmup_steps:
                metrics = self._phase_a_step(data)
            elif self._phase_step < self._improve_phase_start:
                metrics = self._bc_ensemble_step(data)
            else:
                if self._phase_b_step == 0:
                    # BC_ENSEMBLE -> IMPROVE transition: every old_actors[i]
                    # must start identical to actors[i] (mirrors BPPO's own
                    # Phase A->B old_policy sync, looped).
                    for i in range(self.num_policies):
                        self.old_actors[i].load_state_dict(self.actors[i].state_dict())
                metrics = self._improve_step(data)
                # Shared, ensemble-wide clip-decay/transition state --
                # incremented exactly ONCE per train() iteration here, never
                # inside _improve_step's per-member loop (which would burn
                # through clip_decay_steps num_policies-x faster than
                # intended).
                self._phase_b_step += 1
                if self._phase_b_step <= self.clip_decay_steps:
                    self._clip_ratio *= self.clip_decay
            self._phase_step += 1
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
        return {key: value / gradient_steps for key, value in metrics_sum.items()}

    # --- eval: N+1 rollouts per cycle (mixture headline + per-member gating) ---

    def _evaluate(self) -> dict[str, float]:
        metrics = super()._evaluate()  # mixture policy rollout -> "return" etc.
        mixture_policy = self.policy
        for i, actor in enumerate(self.actors):
            self.policy = actor
            try:
                member_metrics = run_exact_episode_eval(
                    self,
                    num_eval_episodes=self.num_eval_episodes,
                    num_eval_steps=self.num_eval_steps,
                )
            finally:
                self.policy = mixture_policy
            for key, value in member_metrics.items():
                metrics[f"{key}_{i}"] = value
        return metrics

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        super()._log_eval_metrics(metrics, step)
        if self._phase_step < self._improve_phase_start:
            return  # no actor improvement to gate yet
        self._maybe_sync_old_actors_from_eval(metrics, step)

    def _maybe_sync_old_actors_from_eval(
        self, metrics: dict[str, float], step: int
    ) -> None:
        """Real-env-eval-gated ``old_actors`` sync. Factored out of
        ``_log_eval_metrics`` as a pure extraction (identical behavior, same
        call site) so ``UniO4OPE`` can override it as a no-op -- real-env
        eval keeps running/logging via ``_log_eval_metrics``'s own
        ``super()`` call regardless, only the sync decision changes."""
        for i in range(self.num_policies):
            score = self._first_metric(metrics, (f"return_{i}",))
            if score != score:  # NaN: eval produced no completed episodes
                warnings.warn(
                    f"UniO4 eval score for member {i} is NaN/missing; "
                    "skipping this member's old_actors sync for this eval round.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if score > self._best_eval_score[i]:
                self._best_eval_score[i] = score
                self.old_actors[i].load_state_dict(self.actors[i].state_dict())
                if self.logger is not None:
                    self.logger.add_scalar(f"eval/old_policy_synced_{i}", 1.0, step)

    # --- checkpointing ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return (
            *self._critic_optimizer_names(),
            *(f"actor_optimizer_{i}" for i in range(self.num_policies)),
        )

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            **self._critic_checkpoint_state(),
            "old_actors": [oa.state_dict() for oa in self.old_actors],
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        self._load_critic_checkpoint_state(state)
        old_actors_state = state.get("old_actors")
        if old_actors_state is not None:
            for old_actor, sd in zip(self.old_actors, old_actors_state):
                old_actor.load_state_dict(sd)

    def _training_state_dict(self) -> dict[str, Any]:
        return {
            "phase_step": self._phase_step,
            "phase_b_step": self._phase_b_step,
            "clip_ratio": self._clip_ratio,
            "best_eval_score": list(self._best_eval_score),
        }

    def _load_training_state_dict(self, state: dict[str, Any]) -> None:
        self._phase_step = int(state.get("phase_step", 0))
        self._phase_b_step = int(state.get("phase_b_step", 0))
        self._clip_ratio = float(state.get("clip_ratio", self.clip_ratio_init))
        best = state.get("best_eval_score")
        if best is None:
            self._best_eval_score = [float("-inf")] * self.num_policies
        else:
            self._best_eval_score = [float(v) for v in best]

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            **self._critic_checkpoint_metadata(),
            "num_policies": self.num_policies,
            "bc_ensemble_steps": self.bc_ensemble_steps,
            "alpha_bc": self.alpha_bc,
            "actor_lr": self.actor_lr,
            "actor_hidden_dims": self.actor_hidden_dims,
            "clip_ratio": self.clip_ratio_init,
            "clip_decay": self.clip_decay,
            "clip_decay_steps": self.clip_decay_steps,
            "entropy_weight": self.entropy_weight,
            "omega": self.omega,
        }
