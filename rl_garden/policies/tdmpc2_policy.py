"""``TDMPC2Policy``: the ``BasePolicy`` wrapper owning TD-MPC2's actor
(Gaussian policy prior), critic (Q-ensemble) and critic target, with
``LatentConsistencyModel`` as its ``actor_extractor`` -- the world model is
the stateful feature extractor (``encode``/``observe`` produce the latent
features the actor/critic heads consume). This policy is now IN the
``BasePolicy`` contract (calls ``BasePolicy.__init__`` properly, unlike its
former namesake that used to live in the now-deleted
``rl_garden.algorithms.tdmpc2`` package, which bypassed it entirely); see
``.agents/rules/adding-algorithm.md`` part A.5.

``critic_extractor=None``/``encoder_sharing="shared"``: the world model is
the only encoder, trained jointly by the world-model loss and the critic
loss through one shared optimizer (``TDMPC2._setup_model``'s
``world_optimizer`` -- see that module's docstring for why TD-MPC2 keeps one
joint optimizer instead of this repo's usual per-network split), so there is
no separate actor/critic extractor split to stop-gradient between.
``WorldModel.extract``/``features_dim``/``update_normalizer`` (duck-typed
``BaseFeaturesExtractor`` surface, see ``rl_garden.world_models.base``) is
what makes this legal without ``LatentConsistencyModel`` subclassing
``BaseFeaturesExtractor``.
"""
from __future__ import annotations

from typing import Optional

import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.common.utils import polyak_update
from rl_garden.networks import init as net_init
from rl_garden.networks.normed_mlp import mlp
from rl_garden.networks.q_ensemble import QEnsemble
from rl_garden.networks.running_scale import RunningScale
from rl_garden.networks.twohot import two_hot_inv
from rl_garden.planners.mppi import MPPIPlanner, PlannerConfig
from rl_garden.policies._tdmpc2_math import gaussian_logprob, log_std, squash
from rl_garden.policies.base import BasePolicy
from rl_garden.world_models.latent_consistency import LatentConsistencyModel


class TDMPC2Policy(BasePolicy):
    """Construction order is RNG-stream-critical -- see
    ``LatentConsistencyModel.apply_init()``'s docstring for the full
    rationale. Summary: every module (the world model's own +
    ``actor``/``critic``/``critic_target`` here) is constructed with default
    ``nn.Linear`` init first (phase 1), and only THEN does every module's
    TD-MPC2 ``trunc_normal_`` init pass run, in the same relative order --
    ``world_model.apply_init()`` (encoder/latent_proj/dynamics/reward) then
    ``actor`` then ``critic`` (+ zero its last layer) then ``critic_target``
    (phase 2) -- reproducing upstream's single combined
    ``self.apply(weight_init)`` pass over one class that used to hold all of
    these submodules together, before the model/actor/critic split (plan
    1.2/1.3). Splitting construction (phase 1) from init (phase 2) like this
    is what lets that single RNG-consuming pass be reproduced across two
    separate ``nn.Module`` objects.
    """

    def __init__(
        self,
        observation_space: "spaces.Box | spaces.Dict",
        action_space: spaces.Space,
        world_model: LatentConsistencyModel,
        planner_cfg: PlannerConfig,
        *,
        mlp_dim: int = 512,
        num_q: int = 5,
        dropout: float = 0.01,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        tau: float = 0.01,
        use_planner: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            actor_extractor=world_model,
            critic_extractor=None,
            encoder_sharing="shared",
        )
        latent_dim = world_model.latent_dim
        action_dim = world_model.action_dim

        # Phase 1: construct with default nn.Linear init (world model's own
        # modules were already constructed by its own __init__, before this
        # policy was built -- see class docstring).
        self.actor = mlp(latent_dim, 2 * [mlp_dim], 2 * action_dim)
        self.critic = QEnsemble(
            latent_dim + action_dim, 2 * [mlp_dim], world_model.num_bins, num_q, dropout=dropout
        )
        self.critic_target = QEnsemble(
            latent_dim + action_dim, 2 * [mlp_dim], world_model.num_bins, num_q, dropout=dropout
        )
        self.num_q = num_q
        self.tau = tau
        self.scale = RunningScale(tau=tau)
        self.register_buffer("log_std_min", torch.tensor(log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(log_std_max - log_std_min))

        # Phase 2: TD-MPC2 trunc_normal_ init, same order as phase 1.
        world_model.apply_init()
        self.actor.apply(net_init.weight_init)
        self.critic.apply(net_init.weight_init)
        for q in self.critic.qs:
            net_init.zero_([q[-1].weight])
        self.critic_target.apply(net_init.weight_init)
        # Clone the target *after* both critic/critic_target init passes
        # (matching upstream cloning after its own combined init, see
        # LatentConsistencyModel.apply_init()'s docstring) so it starts as
        # an exact copy of the zero-initialized online critic.
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self._planner = MPPIPlanner(planner_cfg)
        self.use_planner = use_planner
        self._prev_mean: Optional[torch.Tensor] = None
        self._t0 = True

    @property
    def world_model(self) -> LatentConsistencyModel:
        return self.actor_extractor

    # ------------------------------------------------------------------
    # Actor / critic
    # ------------------------------------------------------------------

    def pi(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Samples a squashed action from the Gaussian policy prior."""
        mean, log_std_raw = self.actor(z).chunk(2, dim=-1)
        log_std_ = log_std(log_std_raw, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        log_prob = gaussian_logprob(eps, log_std_)
        action_dims = eps.shape[-1]

        action = mean + eps * log_std_.exp()
        mean, action, log_prob = squash(mean, action, log_prob)

        entropy_scale = action_dims * log_prob / (log_prob + 1e-8)
        info = {
            "mean": mean,
            "log_std": log_std_,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def Q(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        return_type: str = "min",
        target: bool = False,
        detach: bool = False,
    ) -> torch.Tensor:
        """``return_type`` in {"min", "avg", "all"}; ``target`` selects the
        Polyak-averaged ``critic_target`` ensemble; ``detach`` stops
        gradient into ``self.critic``'s parameters (used for the actor
        update) without touching values."""
        assert return_type in ("min", "avg", "all")
        za = torch.cat([z, a], dim=-1)
        qnet = self.critic_target if target else self.critic
        out = qnet(za)
        if detach and not target:
            out = out.detach()

        if return_type == "all":
            return out

        qidx = torch.randperm(self.num_q, device=out.device)[:2]
        q_scalar = two_hot_inv(
            out[qidx], self.world_model.num_bins, self.world_model.vmin, self.world_model.vmax
        )
        if return_type == "min":
            return q_scalar.min(0).values
        return q_scalar.sum(0) / 2

    def soft_update_target_Q(self) -> None:
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

    # ------------------------------------------------------------------
    # Planner callbacks -- shared by predict() below and by
    # TDMPC2._rollout_action (rl_garden.algorithms.tdmpc2), which threads
    # its own prev_mean/t0 through MPPIPlanner.plan() directly instead of
    # going through predict() (predict()'s prev_mean/t0 state is reserved
    # for eval episodes, see reset_episode()/notify_step_done()) --
    # _evaluate() calls predict() interleaved with rollout within one
    # learn() call, so sharing one mutable prev_mean/t0 between the two
    # would let an eval episode's planning corrupt an in-progress training
    # episode's warm-started mean, and vice versa.
    # ------------------------------------------------------------------

    def policy_prior(self, state) -> torch.Tensor:
        return self.pi(state["z"])[0]

    def value_fn(self, state, action: torch.Tensor) -> torch.Tensor:
        return self.Q(state["z"], action, return_type="avg")

    @property
    def planner(self) -> MPPIPlanner:
        return self._planner

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Call at the start of every episode (training rollout or eval)."""
        self._prev_mean = None
        self._t0 = True

    def notify_step_done(self, done: bool) -> None:
        """Call after every env step with whether that step ended the
        episode (terminated or truncated) -- the *next* ``predict()`` call
        should then plan with ``t0=True`` / a cold planning mean."""
        if done:
            self.reset_episode()

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        if self.use_planner:
            embed = self.world_model.encode(obs)
            is_first = torch.full((embed.shape[0],), self._t0, dtype=torch.bool, device=embed.device)
            state = self.world_model.observe(None, None, embed, is_first)
            action, self._prev_mean = self._planner.plan(
                self.world_model,
                state,
                policy_prior=self.policy_prior,
                value_fn=self.value_fn,
                prev_mean=self._prev_mean,
                t0=self._t0,
                eval_mode=deterministic,
            )
            action = action.unsqueeze(0)
        else:
            with torch.no_grad():
                z = self.world_model.encode(obs)
                action, info = self.pi(z)
                if deterministic:
                    action = info["mean"]
        self._t0 = False
        return action
