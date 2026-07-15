"""CEM/MPPI planner, ported from ``TDMPC2._estimate_value``/``TDMPC2._plan``
in ``3rd_party/tdmpc2/tdmpc2/tdmpc2.py``.

Upstream keeps the planning mean as a hidden ``nn.Buffer`` (``self._prev_mean``)
on the agent, implicitly warm-started across calls within the same episode.
Here it is an explicit argument/return value instead: callers (the training
rollout loop, the eval-loop hooks) own and thread this state through
themselves, matching rl-garden's convention of explicit per-call state over
hidden mutable module attributes, and letting the planner be tested without a
full agent/episode loop around it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_garden.algorithms.tdmpc2 import math_utils
from rl_garden.algorithms.tdmpc2.world_model import WorldModel
from rl_garden.common.types import Obs


@dataclass
class PlannerConfig:
    action_dim: int
    horizon: int = 3
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    iterations: int = 6
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5


def _estimate_value(
    world_model: WorldModel,
    z: torch.Tensor,
    actions: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Rolls out ``actions`` (horizon, num_samples, action_dim) from latent
    ``z`` and returns the discounted, terminal-Q-bootstrapped return."""
    horizon = actions.shape[0]
    num_samples = actions.shape[1]
    g = torch.zeros(num_samples, 1, device=z.device)
    disc = 1.0
    termination = torch.zeros(num_samples, 1, dtype=torch.float32, device=z.device)
    for t in range(horizon):
        reward = math_utils.two_hot_inv(
            world_model.reward(z, actions[t]), world_model.num_bins, world_model.vmin, world_model.vmax
        )
        z = world_model.next(z, actions[t])
        g = g + disc * (1 - termination) * reward
        disc = disc * discount
        if world_model.episodic:
            termination = torch.clip(
                termination + (world_model.termination(z) > 0.5).float(), max=1.0
            )
    action, _ = world_model.pi(z)
    return g + disc * (1 - termination) * world_model.Q(z, action, target=True, return_type="avg")


def plan(
    world_model: WorldModel,
    obs: Obs,
    prev_mean: torch.Tensor | None,
    discount: float,
    cfg: PlannerConfig,
    t0: bool,
    eval_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plans one action via CEM/MPPI. ``obs`` must have a batch dim of 1.

    Returns ``(action, new_prev_mean)`` -- pass ``new_prev_mean`` back in as
    ``prev_mean`` on the next call within the same episode (``t0=False``);
    pass ``prev_mean=None`` (and ``t0=True``) at the first step of an episode.
    """
    device = next(world_model.parameters()).device
    with torch.no_grad():
        z = world_model.encode(obs)

        if cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                cfg.horizon, cfg.num_pi_trajs, cfg.action_dim, device=device
            )
            _z = z.repeat(cfg.num_pi_trajs, 1)
            for t in range(cfg.horizon - 1):
                pi_actions[t], _ = world_model.pi(_z)
                _z = world_model.next(_z, pi_actions[t])
            pi_actions[-1], _ = world_model.pi(_z)

        z = z.repeat(cfg.num_samples, 1)
        mean = torch.zeros(cfg.horizon, cfg.action_dim, device=device)
        std = torch.full(
            (cfg.horizon, cfg.action_dim), cfg.max_std, dtype=torch.float, device=device
        )
        if not t0 and prev_mean is not None:
            mean[:-1] = prev_mean[1:]

        actions = torch.empty(cfg.horizon, cfg.num_samples, cfg.action_dim, device=device)
        if cfg.num_pi_trajs > 0:
            actions[:, : cfg.num_pi_trajs] = pi_actions

        score = None
        elite_actions = None
        for _ in range(cfg.iterations):
            r = torch.randn(
                cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim, device=device
            )
            actions_sample = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
            actions[:, cfg.num_pi_trajs :] = actions_sample

            value = _estimate_value(world_model, z, actions, discount).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            max_value = elite_value.max(0).values
            score = torch.exp(cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = (
                (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)
                / (score.sum(0) + 1e-9)
            ).sqrt()
            std = std.clamp(cfg.min_std, cfg.max_std)

        rand_idx = math_utils.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, a_std = actions[0], std[0]
        if not eval_mode:
            a = a + a_std * torch.randn(cfg.action_dim, device=device)
        new_prev_mean = mean
        return a.clamp(-1, 1), new_prev_mean
