"""``MPPIPlanner``: CEM/MPPI decision-time planner, ported from
``TDMPC2._estimate_value``/``TDMPC2._plan`` in
``3rd_party/tdmpc2/tdmpc2/tdmpc2.py``. TD-MPC2's own planner, not shared with
DreamerV3 (which produces actions from an amortized actor, not a per-step
search) -- this is why it lives in a dedicated ``rl_garden.planners`` package
rather than as a ``WorldModel`` method.

Upstream keeps the planning mean as a hidden ``nn.Buffer`` (``self._prev_mean``)
on the agent, implicitly warm-started across calls within the same episode.
Here it is an explicit argument/return value instead: callers (the training
rollout loop, the eval-loop hooks) own and thread this state through
themselves, matching rl-garden's convention of explicit per-call state over
hidden mutable module attributes, and letting the planner be tested without a
full agent/episode loop around it.

The planner is written against a single-tensor ``state["z"]`` model (TD-MPC2's
``LatentConsistencyModel``) -- it repeats/concatenates that tensor directly
rather than tree-mapping over an arbitrary ``State`` dict, since MPPI has no
other consumer in this repo (Dreamer's imagination-based actor-critic uses
``rl_garden.world_models.imagine`` instead, which *is* state-shape-agnostic).

``policy_prior``/``value_fn`` are injected callbacks rather than direct
``model.pi``/``model.Q`` calls: the actor (policy prior head) and critic
(Q-ensemble) live on ``TDMPC2Policy``, not on the world model (plan 1.3), so
the planner -- which only ever touches the model's ``step``/``reward``/
``continue_`` methods plus its two-hot decode constants -- stays decoupled
from where the policy/value networks actually live.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from rl_garden.networks.twohot import two_hot_inv
from rl_garden.world_models.base import State, WorldModel


def _gumbel_softmax_sample(p: torch.Tensor, temperature: float = 1.0, dim: int = 0) -> torch.Tensor:
    """Sample an index from the Gumbel-Softmax distribution over
    probabilities ``p``, ported from ``3rd_party/tdmpc2/tdmpc2/common
    /math.py`` (moved here, not imported from ``rl_garden.policies
    ._tdmpc2_math`` where the rest of the TD-MPC2 math helpers live --
    planners must not import from policies, see module docstring; this is
    the planner's only consumer of this function)."""
    logits = p.log()
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / temperature
    y_soft = gumbels.softmax(dim)
    return y_soft.argmax(-1)


@dataclass
class PlannerConfig:
    action_dim: int
    #: MDP discount factor for the planning rollout -- shared with the
    #: critic's own TD-target discount (both come from the same
    #: ``_compute_discount(...)`` call in ``TDMPC2._setup_model``), bundled
    #: here since it's the one additional per-rollout constant
    #: ``_estimate_value`` needs beyond the search hyperparameters below.
    discount: float = 0.99
    horizon: int = 3
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    iterations: int = 6
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5


def _estimate_value(
    model: WorldModel,
    state: State,
    actions: torch.Tensor,
    discount: float,
    *,
    policy_prior: Callable[[State], torch.Tensor],
    value_fn: Callable[[State, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Rolls ``actions`` (horizon, num_samples, action_dim) forward from
    ``state`` and returns the discounted, terminal-value-bootstrapped
    return."""
    horizon = actions.shape[0]
    num_samples = actions.shape[1]
    z0 = state["z"]
    g = torch.zeros(num_samples, 1, device=z0.device)
    disc = 1.0
    termination = torch.zeros(num_samples, 1, dtype=torch.float32, device=z0.device)
    cur_state = state
    for t in range(horizon):
        reward = two_hot_inv(
            model.reward(cur_state, actions[t]), model.num_bins, model.vmin, model.vmax
        )
        cur_state = model.step(cur_state, actions[t])
        g = g + disc * (1 - termination) * reward
        disc = disc * discount
        if model.episodic:
            continue_prob = model.continue_(cur_state)
            termination = torch.clip(termination + (continue_prob < 0.5).float(), max=1.0)
    action = policy_prior(cur_state)
    # Bootstraps with the ONLINE value_fn (no target Q), matching upstream
    # TDMPC2._estimate_value (3rd_party/tdmpc2/tdmpc2/tdmpc2.py:136:
    # `self.model.Q(z, action, task, return_type='avg')`, no `target=`); only
    # the TD-target used to train the critic uses the target Q -- the caller
    # (TDMPC2Policy) is responsible for passing an online (non-target) Q as
    # ``value_fn``.
    return g + disc * (1 - termination) * value_fn(cur_state, action)


class MPPIPlanner:
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config

    def plan(
        self,
        model: WorldModel,
        state: State,
        *,
        policy_prior: Callable[[State], torch.Tensor],
        value_fn: Callable[[State, torch.Tensor], torch.Tensor],
        prev_mean: Optional[torch.Tensor],
        t0: bool,
        eval_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plans one action via CEM/MPPI from ``state`` (batch size 1).

        Returns ``(action, new_prev_mean)`` -- pass ``new_prev_mean`` back in
        as ``prev_mean`` on the next call within the same episode
        (``t0=False``); pass ``prev_mean=None`` (and ``t0=True``) at the
        first step of an episode.
        """
        cfg = self.config
        z0 = state["z"]
        device = z0.device
        with torch.no_grad():
            if cfg.num_pi_trajs > 0:
                pi_actions = torch.empty(
                    cfg.horizon, cfg.num_pi_trajs, cfg.action_dim, device=device
                )
                pi_state: State = {"z": z0.repeat(cfg.num_pi_trajs, 1)}
                for t in range(cfg.horizon - 1):
                    pi_actions[t] = policy_prior(pi_state)
                    pi_state = model.step(pi_state, pi_actions[t])
                pi_actions[-1] = policy_prior(pi_state)

            z = z0.repeat(cfg.num_samples, 1)
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

                value = _estimate_value(
                    model, {"z": z}, actions, cfg.discount, policy_prior=policy_prior, value_fn=value_fn
                ).nan_to_num(0)
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

            rand_idx = _gumbel_softmax_sample(score.squeeze(1))
            actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
            a, a_std = actions[0], std[0]
            if not eval_mode:
                a = a + a_std * torch.randn(cfg.action_dim, device=device)
            new_prev_mean = mean
            return a.clamp(-1, 1), new_prev_mean
