"""Flow-matching value network for Value Flows (Dong et al., "Value Flows:
Bringing Distributional Reinforcement Learning to Flow Matching Policies",
arXiv 2510.07650), ported from ``3rd_party/value-flows/utils/networks.py``'s
``ValueVectorField``.

Unlike FloQ's ``CriticVectorField`` (``rl_garden/networks/critic_vector_field.py``),
the reference's ``ValueVectorField`` has no internal ensembling, no time
embedding, and no HL-Gauss probability embedding of the return axis: it is a
plain MLP over ``concat([returns, times, features, actions])``. The twin
critics (``critic_flow1``/``critic_flow2`` in the reference) are built as two
independent instances of this class by ``ValueFlowsPolicy``
(``rl_garden/policies/value_flows_policy.py``), not internal ensembling here.

``integrate_returns``/``integrate_returns_with_jvp`` mirror the reference's
``compute_flow_returns`` (``agents/value_flows.py:248-300``) exactly, including
its ``init_times``-implicitly-zero / ``end_times``-implicitly-one convention
and per-step clipping of the primal (never the JVP tangent). The JVP branch
uses ``torch.func.jvp`` in place of ``jax.jvp``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from rl_garden.networks.mlp import Activation, KernelInit, create_mlp, resolve_activation


class ValueFlowVectorField(nn.Module):
    """Plain velocity-field MLP over the scalar-return axis.

    ``forward(features (B,F), actions (B,A), returns (B,1), times (B,1)) ->
    (B,1)``, matching the reference's ``ValueVectorField.__call__(returns,
    times, observations, actions)`` input concatenation order
    (``[returns, times, features, actions]``) up to argument order in this
    port's own signature.
    """

    def __init__(
        self,
        features_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        *,
        use_layer_norm: bool = False,
        kernel_init: Optional[KernelInit] = None,
        activation_fn: Optional[Activation] = None,
    ) -> None:
        super().__init__()
        input_dim = 1 + 1 + features_dim + action_dim  # returns + times + features + actions
        self.mlp = create_mlp(
            input_dim=input_dim,
            output_dim=1,
            net_arch=hidden_dims,
            activation_fn=resolve_activation(activation_fn, default=nn.ReLU),
            use_layer_norm=use_layer_norm,
            kernel_init=kernel_init,
        )

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([returns, times, features, actions], dim=-1)
        return self.mlp(inputs)


def integrate_returns(
    net: ValueFlowVectorField,
    features: torch.Tensor,
    actions: torch.Tensor,
    x0: torch.Tensor,
    *,
    steps: int,
    end_times: Optional[torch.Tensor] = None,
    clip_range: Optional[tuple[float, float]] = None,
) -> torch.Tensor:
    """Euler-integrate ``net``'s velocity field from ``x0`` (at time 0) to
    ``end_times`` (default 1), taking ``steps`` equal-sized steps -- the time
    fed to the net at step ``i`` is ``i * (end_times / steps)``.

    ``features``/``actions``: ``(B, F)``/``(B, A)``. ``x0``: ``(B, 1)``.
    ``end_times``: ``(B, 1)`` or ``None`` (treated as ones). ``clip_range``,
    when given, clamps the primal after every step (never the caller-visible
    intermediate velocity). Returns ``(B, 1)``.
    """
    batch_size = features.shape[0]
    device, dtype = features.device, features.dtype
    if end_times is None:
        end_times = torch.ones(batch_size, 1, device=device, dtype=dtype)
    step_size = end_times / steps  # init_times is always 0 in this port

    returns = x0
    for step in range(steps):
        times = step * step_size
        velocity = net(features, actions, returns, times)
        returns = returns + step_size * velocity
        if clip_range is not None:
            returns = returns.clamp(clip_range[0], clip_range[1])
    return returns


def integrate_returns_with_jvp(
    net: ValueFlowVectorField,
    features: torch.Tensor,
    actions: torch.Tensor,
    x0: torch.Tensor,
    *,
    steps: int,
    clip_range: Optional[tuple[float, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler-integrate ``net``'s velocity field from ``x0`` at time 0 to time
    1 while propagating the Jacobian-vector product of the primal w.r.t.
    ``x0`` (tangent initialized to ones), via ``torch.func.jvp`` -- ports the
    reference's ``compute_flow_returns(..., return_jac_eps_prod=True)``
    branch. The primal is clipped after every step when ``clip_range`` is
    given; the tangent is never clipped.

    Returns ``(returns (B,1), tangent (B,1))``.
    """
    batch_size = features.shape[0]
    device, dtype = features.device, features.dtype
    step_size = 1.0 / steps

    returns = x0
    tangent = torch.ones_like(x0)
    for step in range(steps):
        times = torch.full((batch_size, 1), step * step_size, device=device, dtype=dtype)

        def velocity_fn(ret: torch.Tensor) -> torch.Tensor:
            return net(features, actions, ret, times)

        velocity, jac_eps_prod = torch.func.jvp(velocity_fn, (returns,), (tangent,))
        returns = returns + step_size * velocity
        tangent = tangent + step_size * jac_eps_prod
        if clip_range is not None:
            returns = returns.clamp(clip_range[0], clip_range[1])
    return returns, tangent
