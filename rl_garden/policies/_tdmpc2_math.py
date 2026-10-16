"""TD-MPC2-specific math helpers, ported from
``3rd_party/tdmpc2/tdmpc2/common/math.py``.

Private to the TD-MPC2 actor (``rl_garden.policies.tdmpc2_policy``) -- unlike
``rl_garden.networks.{symlog,twohot,normed_mlp,running_scale}``, none of
these are shared with DreamerV3 or any other model-based algorithm
(Gaussian-prior reparameterized sampling + tanh squashing are TD-MPC2
actor-specific). Gumbel-softmax elite selection (``gumbel_softmax_sample``,
also ported from upstream's ``common/math.py``) lives in
``rl_garden.planners.mppi`` instead -- a planner module must not import from
``rl_garden.policies`` (planners are policy-agnostic, see that module's
docstring), so it can't live here despite the shared upstream origin.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def log_std(x: torch.Tensor, low: torch.Tensor, dif: torch.Tensor) -> torch.Tensor:
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps: torch.Tensor, log_std_: torch.Tensor) -> torch.Tensor:
    """Gaussian log-probability of a reparameterized sample."""
    residual = -0.5 * eps.pow(2) - log_std_
    log_prob = residual - 0.9189385175704956  # 0.5 * log(2*pi)
    return log_prob.sum(-1, keepdim=True)


def squash(
    mu: torch.Tensor, pi: torch.Tensor, log_pi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply tanh squashing and correct the log-probability accordingly."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


def critic_with_detached_params(critic: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``critic(x)`` evaluated with detached copies of its parameters: the
    output stays differentiable w.r.t. ``x`` but sends no gradient into the
    critic's parameters (upstream TD-MPC2's ``_detach_Qs``, used by the actor
    update)."""
    params = {name: p.detach() for name, p in critic.named_parameters()}
    return torch.func.functional_call(critic, params, (x,))
