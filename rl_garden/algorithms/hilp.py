"""HILP (Park et al. 2024, "Foundation Policies with Hilbert Representations",
arXiv:2402.15567): a goal-conditioned distance value ("Hilbert
representation") trained via double-expectile IQL, plus a downstream
unsupervised skill-discovery policy trained via AWR on latent
random-linear-reward pseudo-rewards derived from that representation.

Standalone offline algorithm (``OfflineRLAlgorithm``), self-managed dataset
via ``HindsightGoalDataset`` (``rl_garden/buffers/hindsight_goal_dataset.py``)
-- no replay buffer, mirrors ``A2ABC``'s reasoning for the same shape
(``rl_garden/algorithms/a2a_bc.py``). State-based (Box observations) only.

Ported directly from ``HILP/hilp_gcrl/src/agents/hilp.py`` (read in
full). Three details that read as inconsistencies but are **deliberate**,
matching the reference exactly -- do not "fix" them:

1. ``compute_value_loss``'s expectile weight and squared term use different
   quantities: the weight comes from a single, ensemble-min-based advantage
   ``adv = (rewards + discount*masks*min(next_v1,next_v2)) - mean(v1_t,v2_t)``
   (both terms from the **target** network), while the squared residual for
   member ``i`` is ``q_i - v_i`` using that member's *own* (not min'd)
   ``next_v_i`` and the **online** ``v_i``. This is standard vanilla-IQL
   ``expectile_loss(diff)`` (weight and square from the same quantity)
   everywhere else in this port (``ExPLORe``/``IDQL``'s single-argument
   ``_expectile_loss``) -- HILP's phi-value loss alone uses this two-argument
   form (``_expectile_loss_two_arg(adv, diff)``).
2. ``compute_skill_critic_loss`` bootstraps from ``skill_value``'s own
   **live, current-step** weights with gradient explicitly stopped (not a
   separate target copy -- there is no ``skill_target_value``, see point 3),
   and has **no mask/done term** (`hilp.py:84`'s own comment: "No 'done'" --
   treated as infinite-horizon).
3. The reference creates a ``skill_target_value`` at init but never updates
   or reads it anywhere -- confirmed dead/vestigial by reading every call
   site. **Not ported.** If this looks like a missing target network, it
   isn't; the reference doesn't use one either.

The one invariant that must not regress: ``phi`` (inside ``GoalConditionedPhiValue``)
receives gradient **only** from the value loss. The pseudo-reward's
``phi(obs)``/``phi(next_obs)`` calls are explicit ``.detach()``, mirroring
the reference reading ``phi`` via a JAX ``method=`` call with no ``params=``
override (a separate, non-traced read) -- in PyTorch that isolation must be
explicit since autograd tracks by default.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.hindsight_goal_dataset import HindsightGoalDataset, HindsightGoalSample
from rl_garden.common.logger import Logger
from rl_garden.common.optim import make_optimizer
from rl_garden.networks import EnsembleQCritic, GoalConditionedPhiValue, UnsquashedGaussianActor, ValueNetwork
from rl_garden.policies.hilp_policy import HILPPolicy


class HILP(OfflineRLAlgorithm):
    _compatible_checkpoint_algorithms = ("HILP",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        dataset_path: str,
        *,
        skill_dim: int = 32,
        value_hidden_dims: Sequence[int] = (512, 512, 512),
        actor_hidden_dims: Sequence[int] = (512, 512, 512),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.95,
        skill_expectile: float = 0.9,
        skill_temperature: float = 10.0,
        skill_discount: float = 0.99,
        p_currgoal: float = 0.0,
        p_trajgoal: float = 0.625,
        p_randomgoal: float = 0.375,
        lr: float = 3e-4,
        batch_size: int = 1024,
        num_traj: Optional[int] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
        std_log: bool = True,
        log_freq: int = 1_000,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 0,
        save_final_checkpoint: bool = True,
    ) -> None:
        super().__init__(
            env=env,
            buffer_size=1,
            buffer_device="cpu",
            batch_size=batch_size,
            gamma=discount,
            offline_sampling="with_replace",
            seed=seed,
            device=device,
            logger=logger,
            std_log=std_log,
            log_freq=log_freq,
            eval_freq=0,
            eval_env=None,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=checkpoint_freq,
            save_replay_buffer=False,
            save_final_checkpoint=save_final_checkpoint,
        )
        if not isinstance(self.env.single_observation_space, spaces.Box):
            raise TypeError("HILP supports Box observation spaces only.")

        self.dataset_path = dataset_path
        self.skill_dim = skill_dim
        self.value_hidden_dims = tuple(value_hidden_dims)
        self.actor_hidden_dims = tuple(actor_hidden_dims)
        self.discount = discount
        self.tau = tau
        self.expectile = expectile
        self.skill_expectile = skill_expectile
        self.skill_temperature = skill_temperature
        self.skill_discount = skill_discount
        self.p_currgoal = p_currgoal
        self.p_trajgoal = p_trajgoal
        self.p_randomgoal = p_randomgoal
        self.lr = lr
        self.num_traj = num_traj

        self._dataset = HindsightGoalDataset(
            dataset_path,
            p_currgoal=p_currgoal,
            p_trajgoal=p_trajgoal,
            p_randomgoal=p_randomgoal,
            discount=skill_discount,
            device=self.device,
            num_traj=num_traj,
        )
        self._setup_model()

    def _setup_model(self) -> None:
        obs_dim = int(np.prod(self.env.single_observation_space.shape))
        action_space = self.env.single_action_space

        value = GoalConditionedPhiValue(obs_dim, self.skill_dim, self.value_hidden_dims)
        value_target = GoalConditionedPhiValue(obs_dim, self.skill_dim, self.value_hidden_dims)
        value_target.load_state_dict(value.state_dict())
        for p in value_target.parameters():
            p.requires_grad_(False)

        # Confirmed by reading create_learner (hilp.py:203-262) directly:
        # skill_value/skill_critic use `value_hidden_dims` (NOT
        # actor_hidden_dims) and use_layer_norm=True (GoalConditionedValue/
        # GoalConditionedCritic's own default), matching the phi-value.
        # skill_actor uses `actor_hidden_dims` via a plain jaxrl_m `Policy`,
        # which is built on plain `MLP` (ReLU, no LayerNorm) -- `Policy`
        # itself has no `use_layer_norm` param at all -- so skill_actor gets
        # neither use_layer_norm nor a non-default activation_fn.
        skill_value = ValueNetwork(
            obs_dim + self.skill_dim, self.value_hidden_dims,
            use_layer_norm=True, activation_fn="gelu",
        )
        skill_critic = EnsembleQCritic(
            obs_dim + self.skill_dim, action_space, self.value_hidden_dims,
            n_critics=2, use_layer_norm=True, activation_fn="gelu",
        )
        skill_critic_target = EnsembleQCritic(
            obs_dim + self.skill_dim, action_space, self.value_hidden_dims,
            n_critics=2, use_layer_norm=True, activation_fn="gelu",
        )
        skill_critic_target.load_state_dict(skill_critic.state_dict())
        for p in skill_critic_target.parameters():
            p.requires_grad_(False)

        skill_actor = UnsquashedGaussianActor(
            obs_dim + self.skill_dim, action_space, self.actor_hidden_dims,
            std_parameterization="uniform", tanh_mean=False, log_std_min=-5.0,
        )

        self.policy = HILPPolicy(
            value=value, value_target=value_target, skill_value=skill_value,
            skill_critic=skill_critic, skill_critic_target=skill_critic_target,
            skill_actor=skill_actor,
        ).to(self.device)

        self.value_optimizer = make_optimizer(list(self.policy.value.parameters()), lr=self.lr)
        self.skill_value_optimizer = make_optimizer(
            list(self.policy.skill_value.parameters()), lr=self.lr
        )
        self.skill_critic_optimizer = make_optimizer(
            list(self.policy.skill_critic.parameters()), lr=self.lr
        )
        self.skill_actor_optimizer = make_optimizer(
            list(self.policy.skill_actor.parameters()), lr=self.lr
        )

    def _sample_skills(self, batch_size: int) -> torch.Tensor:
        z = torch.randn(batch_size, self.skill_dim, device=self.device)
        return z / z.norm(dim=-1, keepdim=True)

    @staticmethod
    def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return weight * diff.pow(2)

    @staticmethod
    def _expectile_loss_two_arg(
        adv: torch.Tensor, diff: torch.Tensor, expectile: float
    ) -> torch.Tensor:
        weight = torch.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * diff.pow(2)

    def _compute_value_loss(self, batch: HindsightGoalSample) -> tuple[torch.Tensor, dict[str, float]]:
        masks = 1.0 - batch.success
        rewards = batch.success - 1.0

        with torch.no_grad():
            next_v1_t, next_v2_t = self.policy.value_target(batch.next_obs, batch.goals)
            next_v_min = torch.minimum(next_v1_t, next_v2_t)
            q_for_adv = rewards + self.discount * masks * next_v_min

            v1_t, v2_t = self.policy.value_target(batch.obs, batch.goals)
            v_t = (v1_t + v2_t) / 2.0
            adv = q_for_adv - v_t

            q1 = rewards + self.discount * masks * next_v1_t
            q2 = rewards + self.discount * masks * next_v2_t

        v1, v2 = self.policy.value(batch.obs, batch.goals)
        value_loss1 = self._expectile_loss_two_arg(adv, q1 - v1, self.expectile).mean()
        value_loss2 = self._expectile_loss_two_arg(adv, q2 - v2, self.expectile).mean()
        value_loss = value_loss1 + value_loss2
        return value_loss, {"value_loss": float(value_loss.detach().item()), "adv": float(adv.mean().item())}

    def _compute_skill_losses(
        self, batch: HindsightGoalSample
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size = batch.obs.shape[0]
        with torch.no_grad():
            phis = self.policy.value.phi(batch.obs)
            next_phis = self.policy.value.phi(batch.next_obs)
        skills = self._sample_skills(batch_size)
        skill_rewards = ((next_phis - phis) * skills).sum(dim=-1)

        obs_skill = torch.cat([batch.obs, skills], dim=-1)
        next_obs_skill = torch.cat([batch.next_obs, skills], dim=-1)

        with torch.no_grad():
            tq1, tq2 = self.policy.skill_critic_target(obs_skill, batch.actions)
            tq_min = torch.minimum(tq1, tq2).squeeze(-1)

        v_online = self.policy.skill_value(obs_skill).squeeze(-1)
        skill_value_loss = self._expectile_loss(
            tq_min - v_online, self.skill_expectile
        ).mean()

        with torch.no_grad():
            next_v_for_critic = self.policy.skill_value(next_obs_skill).squeeze(-1)
        critic_target = skill_rewards + self.skill_discount * next_v_for_critic
        q1, q2 = self.policy.skill_critic(obs_skill, batch.actions)
        skill_critic_loss = (
            (q1.squeeze(-1) - critic_target).pow(2) + (q2.squeeze(-1) - critic_target).pow(2)
        ).mean()

        with torch.no_grad():
            v_for_actor = self.policy.skill_value(obs_skill).squeeze(-1)
            tq1a, tq2a = self.policy.skill_critic_target(obs_skill, batch.actions)
            q_for_actor = torch.minimum(tq1a, tq2a).squeeze(-1)
            adv_actor = q_for_actor - v_for_actor
            exp_a = torch.exp(adv_actor * self.skill_temperature).clamp(max=100.0)
        log_probs = self.policy.skill_actor.evaluate_action_log_prob(
            obs_skill, batch.actions
        ).squeeze(-1)
        skill_actor_loss = -(exp_a * log_probs).mean()

        total = skill_value_loss + skill_critic_loss + skill_actor_loss
        metrics = {
            "skill_value_loss": float(skill_value_loss.detach().item()),
            "skill_critic_loss": float(skill_critic_loss.detach().item()),
            "skill_actor_loss": float(skill_actor_loss.detach().item()),
        }
        return total, metrics

    def _polyak_update(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(
                self.policy.value.parameters(), self.policy.value_target.parameters()
            ):
                p_targ.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
            for p, p_targ in zip(
                self.policy.skill_critic.parameters(),
                self.policy.skill_critic_target.parameters(),
            ):
                p_targ.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            batch = self._dataset.sample(self.batch_size)

            value_loss, value_metrics = self._compute_value_loss(batch)
            skill_loss, skill_metrics = self._compute_skill_losses(batch)
            total_loss = value_loss + skill_loss

            self.value_optimizer.zero_grad(set_to_none=True)
            self.skill_value_optimizer.zero_grad(set_to_none=True)
            self.skill_critic_optimizer.zero_grad(set_to_none=True)
            self.skill_actor_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self.value_optimizer.step()
            self.skill_value_optimizer.step()
            self.skill_critic_optimizer.step()
            self.skill_actor_optimizer.step()
            self._polyak_update()

            metrics = {"loss": float(total_loss.detach().item()), **value_metrics, **skill_metrics}
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        del compute_info
        return {key: value / gradient_steps for key, value in metrics_sum.items()}

    def _optimizer_names(self) -> tuple[str, ...]:
        return (
            "value_optimizer",
            "skill_value_optimizer",
            "skill_critic_optimizer",
            "skill_actor_optimizer",
        )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "skill_dim": self.skill_dim,
            "value_hidden_dims": self.value_hidden_dims,
            "actor_hidden_dims": self.actor_hidden_dims,
            "discount": self.discount,
            "tau": self.tau,
            "expectile": self.expectile,
            "skill_expectile": self.skill_expectile,
            "skill_temperature": self.skill_temperature,
            "skill_discount": self.skill_discount,
            "p_currgoal": self.p_currgoal,
            "p_trajgoal": self.p_trajgoal,
            "p_randomgoal": self.p_randomgoal,
        }
