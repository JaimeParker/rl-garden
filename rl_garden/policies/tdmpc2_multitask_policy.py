"""``MultitaskTDMPC2Policy``: ``BasePolicy`` wrapper so
``BaseAlgorithm.save()``/``load()``/``state_dict()`` keep working unmodified
for ``TDMPC2Multitask``. Still **out of** the ``BasePolicy`` contract (unlike
the single-task ``rl_garden.policies.tdmpc2_policy.TDMPC2Policy``, which
graduated into it per plan 1.3) -- multitask training never touches a live
env (see ``rl_garden.algorithms.tdmpc2_multitask``'s module docstring), so there is no
observation/action space or rollout ``actor_extractor`` role for this class
to declare; it remains a documented exception in
``.agents/rules/adding-algorithm.md`` part A.5.

Owns the actor (Gaussian prior head with task-conditioned action masking),
critic (Q-ensemble), critic target and running scale -- moved out of
``MultitaskWorldModel`` per plan 1.3, mirroring the single-task
model/policy split (``rl_garden.policies.tdmpc2_policy.TDMPC2Policy``).

``predict()`` (action selection via the CEM planner) is out of scope for v1:
multitask training never touches a live env, so nothing calls it yet. It
raises rather than silently returning a wrong action, to be picked up
explicitly when multitask evaluation is added.
"""
from __future__ import annotations

import torch

from rl_garden.common.types import Obs
from rl_garden.common.utils import polyak_update
from rl_garden.networks import init as net_init
from rl_garden.networks.normed_mlp import mlp
from rl_garden.networks.q_ensemble import QEnsemble
from rl_garden.networks.running_scale import RunningScale
from rl_garden.networks.twohot import two_hot_inv
from rl_garden.policies._tdmpc2_math import gaussian_logprob, log_std, squash
from rl_garden.policies.base import BasePolicy
from rl_garden.world_models.multitask_latent_consistency import MultitaskWorldModel


class MultitaskTDMPC2Policy(BasePolicy):
    def __init__(
        self,
        world_model: MultitaskWorldModel,
        *,
        mlp_dim: int = 512,
        num_q: int = 5,
        dropout: float = 0.01,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        tau: float = 0.01,
    ) -> None:
        # Bypasses BasePolicy.__init__ (which requires an
        # actor_extractor/action_space/observation_space) -- see class
        # docstring.
        torch.nn.Module.__init__(self)
        self.world_model = world_model
        latent_dim = world_model.latent_dim
        action_dim = world_model.action_dim
        task_dim = world_model.task_dim

        # Construction order mirrors TDMPC2Policy.__init__ (deferred-init,
        # see LatentConsistencyModel.apply_init()'s docstring for the full
        # RNG-stream rationale): build actor/critic/critic_target with
        # default nn.Linear init first, THEN run every module's TD-MPC2
        # init pass -- world_model.apply_init(), then actor, then critic
        # (+zero its last layer), then critic_target (cloned from critic).
        self.actor = mlp(latent_dim + task_dim, 2 * [mlp_dim], 2 * action_dim)
        self.critic = QEnsemble(
            latent_dim + action_dim + task_dim, 2 * [mlp_dim], world_model.num_bins, num_q, dropout=dropout
        )
        self.critic_target = QEnsemble(
            latent_dim + action_dim + task_dim, 2 * [mlp_dim], world_model.num_bins, num_q, dropout=dropout
        )
        self.num_q = num_q
        self.tau = tau
        self.scale = RunningScale(tau=tau)
        self.register_buffer("log_std_min", torch.tensor(log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(log_std_max - log_std_min))

        world_model.apply_init()
        self.actor.apply(net_init.weight_init)
        self.critic.apply(net_init.weight_init)
        for q in self.critic.qs:
            net_init.zero_([q[-1].weight])
        self.critic_target.apply(net_init.weight_init)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    def pi(self, z: torch.Tensor, task: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.world_model.task_embed(z, task)
        mean, log_std_raw = self.actor(z).chunk(2, dim=-1)
        log_std_ = log_std(log_std_raw, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        action_mask = self.world_model.action_mask(task)
        mean = mean * action_mask
        log_std_ = log_std_ * action_mask
        eps = eps * action_mask
        action_dims = action_mask.sum(-1).unsqueeze(-1)

        log_prob = gaussian_logprob(eps, log_std_)
        scaled_log_prob = log_prob * action_dims

        action = mean + eps * log_std_.exp()
        mean, action, log_prob = squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
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
        task: torch.Tensor,
        return_type: str = "min",
        target: bool = False,
        detach: bool = False,
    ) -> torch.Tensor:
        assert return_type in ("min", "avg", "all")
        z = self.world_model.task_embed(z, task)
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

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        raise NotImplementedError(
            "TDMPC2Multitask has no online rollout/eval path in this port "
            "(training is offline-only, see rl_garden.algorithms.tdmpc2_multitask); predict() "
            "is unused until multitask evaluation is implemented."
        )
