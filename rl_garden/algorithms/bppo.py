"""BPPO: Behavior Proximal Policy Optimization for offline datasets.

Ported from ``3rd_party/BPPO`` (Zhuang et al., ICLR 2023, arXiv:2302.11312).
Two-phase training, both driven by the same offline dataset:

* **Phase A** (``self._phase_step < critic_warmup_steps``): fit ``V(s)`` by
  regressing to Monte-Carlo return-to-go and ``Q(s,a)`` by SARSA/TD
  bootstrap on the dataset's real next action -- exactly Cal-QL's SARSA
  reference-value setup (``rl_garden/algorithms/calql.py``,
  ``rl_garden/buffers/sarsa_buffer.py``), reused as-is here. No actor
  update. This phase's logic lives in ``BPPOCriticMixin`` -- shared with
  ``UniO4`` (``rl_garden/algorithms/unio4.py``), which trains one critic
  across its whole actor ensemble rather than one per member.
* **Phase B** (``self._phase_step >= critic_warmup_steps``): V/Q are frozen
  (the paper's ``is_offpolicy_update=False`` default; continued off-policy
  Q refresh is out of scope). A BC-initialized actor (``BCPolicy
  (tanh_squash=False)``, an unsquashed Gaussian -- see
  ``rl_garden/policies/bc_policy.py``) is improved with a plain PPO-clip
  loss, no explicit KL/BC constraint: the trust region alone bounds
  divergence from the behavior policy. Advantage is ``Q(s,a) - V(s)`` for
  actions sampled from ``old_policy``, normalized then asymmetrically
  weighted by ``omega`` (upweights positive advantage).

``old_policy`` is synced from ``policy`` at construction (or immediately
after a ``--bc_checkpoint`` actor warm start), at the Phase A->B transition,
and thereafter only when offline eval confirms improvement (see
``_log_eval_metrics`` below) -- the paper's "safe" iteration scheme. Box
observations only, matching the D4RL MuJoCo locomotion scope this port
targets.
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.algorithms.ppo import ppo_clip_policy_loss
from rl_garden.buffers.sarsa_buffer import SarsaMCTensorReplayBuffer
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.common.logger import Logger
from rl_garden.common.optim import make_optimizer
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.networks import KernelInit
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.networks.value import ScalarQNetwork, ValueNetwork
from rl_garden.policies.bc_policy import BCPolicy


class BPPOCriticMixin:
    """Shared V(s)/Q(s,a) critic: MC-return regression + SARSA/TD bootstrap.

    Mixin, no base class (mirrors the ``CalQLCore``/``IQLCore`` idiom used
    throughout this codebase): provides building blocks
    (``_build_critic``/``_phase_a_step``/``_critic_optimizer_names``/
    ``_critic_checkpoint_state``/``_load_critic_checkpoint_state``/
    ``_critic_checkpoint_metadata``) that a subclass composes into its own
    ``_setup_model``/``_optimizer_names``/checkpoint hooks -- plain method
    calls, not cooperative ``super()`` overrides, since this mixin doesn't
    need to layer onto a parent's *own* critic logic the way e.g. Cal-QL
    layers onto CQL's regularizer.

    Owns a **dedicated ``self._critic_step`` counter**, separate from
    whatever phase-gate counter the subclass uses to decide *when* to call
    ``_phase_a_step`` -- ``BPPO`` has a 2-phase gate where ``_critic_step``
    and its own ``_phase_step`` coincide throughout Phase A, but ``UniO4``
    has a 3-phase gate (critic / BC-ensemble / improve) where they diverge
    once the BC-ensemble phase starts. Reading the subclass's own phase
    counter here for the Polyak-update cadence would silently tie the
    critic's target-update schedule to a clock it doesn't own.
    """

    def _build_critic(self) -> None:
        obs_dim = self.env.single_observation_space.shape[0]
        action_dim = self.env.single_action_space.shape[0]
        self.value_net = ValueNetwork(obs_dim, list(self.value_hidden_dims)).to(
            self.device
        )
        self.q_net = ScalarQNetwork(obs_dim, action_dim, list(self.q_hidden_dims)).to(
            self.device
        )
        self.q_target = copy.deepcopy(self.q_net).to(self.device)
        for param in self.q_target.parameters():
            param.requires_grad_(False)
        self.value_optimizer = make_optimizer(
            list(self.value_net.parameters()),
            lr=self.value_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.q_optimizer = make_optimizer(
            list(self.q_net.parameters()),
            lr=self.q_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self._critic_step = 0

    def _phase_a_step(self, data) -> dict[str, float]:
        value_pred = self.value_net(data.obs).squeeze(-1)
        value_loss = F.mse_loss(value_pred, data.mc_returns)

        with torch.no_grad():
            target_next_q = self.q_target(data.next_obs, data.next_actions).squeeze(-1)
            td_target = data.rewards + self.gamma * target_next_q
        # Two independent masks: true termination (`dones`, TD bootstrap
        # stop) and the artificial `timeouts` boundary the SARSA next-action
        # shift must not cross (`next_action_valid`) -- see
        # rl_garden/buffers/sarsa_buffer.py's module docstring.
        valid = (~data.dones.bool()) & data.next_action_valid
        q_pred = self.q_net(data.obs, data.actions).squeeze(-1)
        if valid.any():
            q_loss = F.mse_loss(q_pred[valid], td_target[valid])
        else:
            q_loss = q_pred.sum() * 0.0

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        if self._critic_step % self.target_update_freq == 0:
            polyak_update(self.q_net.parameters(), self.q_target.parameters(), self.tau)
        self._critic_step += 1

        return {
            "loss": float(value_loss.detach().item() + q_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "q_loss": float(q_loss.detach().item()),
            "v": float(value_pred.detach().mean().item()),
            "q": float(q_pred.detach().mean().item()),
        }

    def _critic_optimizer_names(self) -> tuple[str, ...]:
        return ("value_optimizer", "q_optimizer")

    def _critic_checkpoint_state(self) -> dict[str, Any]:
        return {
            "value_net": self.value_net.state_dict(),
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "critic_step": self._critic_step,
        }

    def _load_critic_checkpoint_state(self, state: dict[str, Any]) -> None:
        if "value_net" in state:
            self.value_net.load_state_dict(state["value_net"])
        if "q_net" in state:
            self.q_net.load_state_dict(state["q_net"])
        if "q_target" in state:
            self.q_target.load_state_dict(state["q_target"])
        self._critic_step = int(state.get("critic_step", 0))

    def _critic_checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "critic_warmup_steps": self.critic_warmup_steps,
            "value_lr": self.value_lr,
            "q_lr": self.q_lr,
            "tau": self.tau,
            "target_update_freq": self.target_update_freq,
            "value_hidden_dims": self.value_hidden_dims,
            "q_hidden_dims": self.q_hidden_dims,
        }


class BPPO(BPPOCriticMixin, OfflineRLAlgorithm):
    """Behavior Proximal Policy Optimization. Box observations only."""

    _compatible_checkpoint_algorithms = ("BPPO",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *,
        buffer_size: int = 1_000_000,
        buffer_device: str = "cuda",
        batch_size: int = 256,
        gamma: float = 0.99,
        # -- Phase A: critic (V/Q) warmup --
        critic_warmup_steps: int = 2_000_000,
        value_lr: float = 1e-4,
        q_lr: float = 1e-4,
        tau: float = 0.005,
        target_update_freq: int = 2,
        value_hidden_dims: Sequence[int] = (512, 512, 512),
        q_hidden_dims: Sequence[int] = (1024, 1024),
        # -- Phase B: actor (PPO-clip) improvement --
        actor_lr: float = 1e-4,
        actor_hidden_dims: Sequence[int] = (1024, 1024),
        clip_ratio: float = 0.25,
        clip_decay: float = 0.96,
        clip_decay_steps: int = 200,
        entropy_weight: float = 0.0,
        omega: float = 0.9,
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
                f"BPPO supports only Box observation spaces, got {type(obs_space)}"
            )
        if critic_warmup_steps < 0:
            raise ValueError(
                f"critic_warmup_steps must be non-negative, got {critic_warmup_steps}."
            )
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

        # Dedicated phase-gate/decay state -- deliberately NOT reusing
        # self._global_step/self._global_update, which agent.load_state_dict()
        # restores unconditionally on *any* checkpoint resume; see
        # _training_state_dict()/_load_training_state_dict() below and
        # load_actor_from()'s docstring.
        self._phase_step = 0
        self._phase_b_step = 0
        self._clip_ratio = clip_ratio
        self._best_eval_score = float("-inf")

        self._setup_model()

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
        self.policy = self._build_actor_policy()
        self.old_policy = self._build_actor_policy()
        self.old_policy.load_state_dict(self.policy.state_dict())
        for param in self.old_policy.parameters():
            param.requires_grad_(False)

        self._build_critic()

        self.actor_optimizer = make_optimizer(
            list(self.policy.actor_parameters()),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
            use_adamw=self.use_adamw,
        )
        self.replay_buffer = self._build_replay_buffer()

    def load_actor_from(self, path: str | Path) -> None:
        """Warm-start the actor from another checkpoint's ``policy`` weights
        (e.g. a ``BC`` pretrained checkpoint) -- only ``self.policy``
        (and ``self.old_policy``, synced to match) are touched.

        Deliberately bypasses ``load()``/``load_state_dict()``: those
        restore ``_global_step``/``_global_update`` from the checkpoint
        unconditionally, which would pollute BPPO's own counters with the
        source checkpoint's unrelated training length. No
        ``_compatible_checkpoint_algorithms`` entry is needed for this path
        -- it only requires the checkpoint's ``policy`` state dict to match
        ``self.policy``'s architecture (same ``net_arch``/``tanh_squash``),
        not that it came from a same-shaped ``BaseAlgorithm``.
        """
        checkpoint = load_checkpoint_file(path, map_location=self.device)
        policy_state = checkpoint["state"]["policy"]
        self.policy.load_state_dict(policy_state)
        self.old_policy.load_state_dict(policy_state)

    # --- Phase B: actor (PPO-clip) ---

    def _weighted_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        if self.omega == 0.5:
            return advantage
        weight = torch.where(
            advantage > 0,
            torch.full_like(advantage, self.omega),
            torch.full_like(advantage, 1.0 - self.omega),
        )
        return weight * advantage

    def _phase_b_step_update(self, data) -> dict[str, float]:
        obs = data.obs
        with torch.no_grad():
            old_features = self.old_policy.extract_features(obs)
            action, old_log_prob = self.old_policy.actor.action_log_prob(old_features)
            advantage = (self.q_net(obs, action) - self.value_net(obs)).squeeze(-1)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage = self._weighted_advantage(advantage)

        new_features = self.policy.extract_features(obs, stop_gradient=False)
        new_log_prob = self.policy.actor.evaluate_action_log_prob(new_features, action)
        ratio = (new_log_prob.squeeze(-1) - old_log_prob.squeeze(-1)).exp()

        clip_loss = ppo_clip_policy_loss(advantage, ratio, self._clip_ratio)
        entropy = self.policy.actor.entropy(new_features).squeeze(-1).mean()
        # Entropy is *added* to the objective in the original (BPPO's own
        # loss is already negated, unlike this codebase's _policy_loss
        # convention) -- so it's subtracted here after ppo_clip_policy_loss's
        # already-negated clip term. Flipping this sign is a real bug, not a
        # style choice: it would reward low-entropy (collapsed) policies.
        loss = clip_loss - self.entropy_weight * entropy

        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.actor_parameters()), self.grad_clip_norm
            )
        self.actor_optimizer.step()

        self._phase_b_step += 1
        if self._phase_b_step <= self.clip_decay_steps:
            self._clip_ratio *= self.clip_decay

        return {
            "loss": float(loss.detach().item()),
            "actor_loss": float(clip_loss.detach().item()),
            "entropy": float(entropy.detach().item()),
            "ratio": float(ratio.detach().mean().item()),
            "advantage": float(advantage.detach().mean().item()),
            "clip_ratio": self._clip_ratio,
        }

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
            else:
                if self._phase_b_step == 0:
                    # Phase A -> B transition: the actor about to be
                    # improved must start identical to old_policy (mirrors
                    # the original's ProximalPolicyOptimization.load loading
                    # BC weights into both _policy and _old_policy).
                    self.old_policy.load_state_dict(self.policy.state_dict())
                metrics = self._phase_b_step_update(data)
            self._phase_step += 1
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value
        return {key: value / gradient_steps for key, value in metrics_sum.items()}

    # --- old_policy sync: eval-gated, piggybacked on the runner's own
    # eval cadence (no internal eval loop) ---

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        super()._log_eval_metrics(metrics, step)
        if self._phase_step < self.critic_warmup_steps:
            return  # no actor to gate yet
        score = self._first_metric(metrics, ("return",))
        if score != score:  # NaN: eval produced no completed episodes
            warnings.warn(
                "BPPO eval score is NaN/missing; skipping old_policy sync "
                "for this eval round.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        if score > self._best_eval_score:
            self._best_eval_score = score
            self.old_policy.load_state_dict(self.policy.state_dict())
            if self.logger is not None:
                self.logger.add_scalar("eval/old_policy_synced", 1.0, step)

    # --- checkpointing ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return (*self._critic_optimizer_names(), "actor_optimizer")

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            **self._critic_checkpoint_state(),
            "old_policy": self.old_policy.state_dict(),
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        self._load_critic_checkpoint_state(state)
        if "old_policy" in state:
            self.old_policy.load_state_dict(state["old_policy"])

    def _training_state_dict(self) -> dict[str, Any]:
        return {
            "phase_step": self._phase_step,
            "phase_b_step": self._phase_b_step,
            "clip_ratio": self._clip_ratio,
            "best_eval_score": self._best_eval_score,
        }

    def _load_training_state_dict(self, state: dict[str, Any]) -> None:
        self._phase_step = int(state.get("phase_step", 0))
        self._phase_b_step = int(state.get("phase_b_step", 0))
        self._clip_ratio = float(state.get("clip_ratio", self.clip_ratio_init))
        self._best_eval_score = float(state.get("best_eval_score", float("-inf")))

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            **self._critic_checkpoint_metadata(),
            "actor_lr": self.actor_lr,
            "actor_hidden_dims": self.actor_hidden_dims,
            "clip_ratio": self.clip_ratio_init,
            "clip_decay": self.clip_decay,
            "clip_decay_steps": self.clip_decay_steps,
            "entropy_weight": self.entropy_weight,
            "omega": self.omega,
        }
