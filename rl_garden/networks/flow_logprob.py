"""Reverse-ODE log-density estimator for a flow-matching BC policy, ported
from FAC's ``logprob_given_actions`` (``3rd_party/FAC/agents/fac.py:288-399``).

A velocity field trained by conditional flow matching (base ``N(0, I)`` at
``t=0``, data at ``t=1``) admits an exact log-density for any action via the
instantaneous change-of-variables formula: integrate the ODE backward from
the action at ``t=1`` to the Gaussian base at ``t=0`` while accumulating the
field's divergence along the trajectory, then combine the base log-density
with the accumulated divergence.
"""
from __future__ import annotations

import math
from typing import Literal

import torch

from rl_garden.networks.actor_vector_field import ActorVectorField

LogpMethod = Literal["exact", "hutch-rade", "hutch-gaus"]

_VALID_METHODS = ("exact", "hutch-rade", "hutch-gaus")


def flow_divergence(
    vector_field: ActorVectorField,
    features: torch.Tensor,
    x: torch.Tensor,
    times: torch.Tensor,
    *,
    method: LogpMethod,
    num_probes: int,
) -> torch.Tensor:
    """Per-sample divergence of ``vector_field(features, x, times)`` w.r.t.
    ``x``, matching FAC's ``make_divergence_exact``/``make_divergence_hutchinson``
    (``3rd_party/FAC/agents/fac.py:288-317``). ``exact`` computes the full
    per-sample Jacobian trace via ``vmap(jacrev(...))``. ``hutch-rade``/
    ``hutch-gaus`` estimate the trace as ``eps^T J eps`` via one batched
    ``torch.func.vjp`` per probe -- the batch Jacobian is block-diagonal, so a
    single batched vjp with ``eps`` of shape ``(B, A)`` already yields
    per-sample values -- averaged over ``num_probes`` probes."""
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method: {method!r}")
    if num_probes < 1:
        raise ValueError(f"num_probes must be >= 1, got {num_probes}")

    if method == "exact":

        def single_sample_velocity(
            feat_i: torch.Tensor, x_i: torch.Tensor, t_i: torch.Tensor
        ) -> torch.Tensor:
            return vector_field(feat_i.unsqueeze(0), x_i.unsqueeze(0), t_i.unsqueeze(0)).squeeze(0)

        jacobian_fn = torch.func.jacrev(single_sample_velocity, argnums=1)
        jacobian = torch.func.vmap(jacobian_fn)(features, x, times)  # (B, A, A)
        return jacobian.diagonal(dim1=-2, dim2=-1).sum(-1)

    def batched_velocity(x_in: torch.Tensor) -> torch.Tensor:
        return vector_field(features, x_in, times)

    _, vjp_fn = torch.func.vjp(batched_velocity, x)
    divergence = x.new_zeros(x.shape[0])
    for _ in range(num_probes):
        if method == "hutch-rade":
            eps = torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2 - 1
        else:
            eps = torch.randn_like(x)
        (vjp_result,) = vjp_fn(eps)
        divergence = divergence + (vjp_result * eps).sum(-1)
    return divergence / num_probes


def flow_log_prob(
    vector_field: ActorVectorField,
    features: torch.Tensor,
    actions: torch.Tensor,
    *,
    num_steps: int,
    method: LogpMethod = "exact",
    num_probes: int = 8,
) -> torch.Tensor:
    """Log-density of ``actions`` under the flow-matching BC policy
    ``vector_field``, via reverse-Euler integration, matching FAC's
    ``logprob_given_actions`` (``3rd_party/FAC/agents/fac.py:358-399``):
    starting from ``x = actions`` at ``t=1``, for ``k = 0 .. num_steps - 1``
    with ``t_k = (num_steps - k) / num_steps``, accumulate
    ``logdiv += divergence(v(x, t_k)) / num_steps`` and step
    ``x -= v(x, t_k) / num_steps``; the returned value is
    ``base_log_prob(x) - logdiv`` where ``base_log_prob`` is the standard
    normal log-density.

    Runs entirely under an internal ``torch.no_grad()`` (``torch.func``
    primitives work correctly under an ambient ``no_grad()``, see
    ``rl_garden/policies/qam_policy.py``) so no autograd graph is built
    through the Euler loop; the result always has ``requires_grad == False``,
    regardless of the caller's grad context.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method: {method!r}")
    if num_probes < 1:
        raise ValueError(f"num_probes must be >= 1, got {num_probes}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if not vector_field.use_time_conditioning:
        raise ValueError("flow_log_prob requires a time-conditioned vector_field.")

    with torch.no_grad():
        x = actions
        batch_size, action_dim = actions.shape
        log_divergence = actions.new_zeros(batch_size)
        for k in range(num_steps):
            t_k = (num_steps - k) / num_steps
            times = actions.new_full((batch_size, 1), t_k)
            velocity = vector_field(features, x, times)
            divergence = flow_divergence(
                vector_field, features, x, times, method=method, num_probes=num_probes
            )
            log_divergence = log_divergence + divergence / num_steps
            x = x - velocity / num_steps

        base_log_prob = -0.5 * (x**2).sum(-1) - 0.5 * action_dim * math.log(2 * math.pi)
        return (base_log_prob - log_divergence).detach()
