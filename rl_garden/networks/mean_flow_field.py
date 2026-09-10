"""MeanFlow vector-field network and loss, ported from ``MeanFlow``
(unofficial PyTorch implementation of "Mean Flows for One-step Generative
Modeling" and "Improved Mean Flows" https://github.com/haidog-yaqub/MeanFlow).

Unlike ``ActorVectorField``'s one-step distillation (``flow_onestep_distill_loss``,
Milestone B of the RL-100 plan) or ``DiffusionCMDistillOnline``'s online LCM
self-consistency distillation, MeanFlow trains a single network end-to-end
via a JVP-based differential identity to directly regress an *average*
velocity field ``u(x, tau, rho)`` that is one-step-sampleable by
construction -- no separate multi-step teacher or distillation phase.

Time convention note (the same class of sign/direction issue Milestone D hit
with RL-100's sigma-space flow scheduler): upstream's ``meanflow.py`` uses
``t=0`` data / ``t=1`` noise, velocity pointing data->noise
(``v = e - x``). ``ActorVectorField``'s own convention (``fql.py``,
``flow_bc_policy.py``) is the opposite: ``tau=0`` noise / ``tau=1`` data,
velocity noise->data (``vel_target = actions - x_0``). This module is
re-derived directly in rl-garden's ``tau``-convention (not transcribed),
via the mapping ``tau = 1 - t``, ``rho = 1 - r``, ``v_rlgarden = -v_MeanFlow``,
``u_rlgarden = -u_MeanFlow`` -- verified independently against upstream's
own identity by substitution, and pinned by a golden-value regression test
against the actual vendored ``MeanFlow/meanflow.py`` code
(``tests/test_mean_flow_bc.py``).

The MeanFlow identity, re-derived from the definition of average velocity
``(rho-tau)*u(x_tau,tau,rho) = integral_tau^rho v(x_s,s) ds`` via the
Leibniz rule (differentiating w.r.t. the lower limit ``tau``, ``rho``
fixed):

    u(x_tau, tau, rho) = v(x_tau, tau) + (rho - tau) * d/dtau[u(x_tau, tau, rho)]

where ``d/dtau[u]`` is the total derivative of ``u`` along the trajectory
(``x_tau`` evolving at rate ``v``, ``rho`` held fixed) -- computed via
``torch.autograd.functional.jvp``, matching upstream's own JVP call.

CFG (classifier-free guidance) and the image ``Normalizer`` are dropped:
rl-garden BC has no class labels (``cfg_scale=None`` collapses upstream's
own loss to its simple, non-CFG branch, not an approximation of it), and no
sibling BC algorithm normalizes ``data.actions`` before computing its loss.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn

from rl_garden.networks.mlp import Activation, KernelInit, create_mlp, resolve_activation

MeanFlowMode = Literal["meanflow", "i-meanflow"]


class MeanFlowActorField(nn.Module):
    """MLP producing both the average velocity ``u`` and the instantaneous
    velocity ``v`` from one shared trunk + a single combined output layer
    (split via ``chunk``), mirroring ``ActorVectorField``'s own
    ``create_mlp(..., output_dim=...)`` shape (a single trunk+output
    ``nn.Sequential``, rather than ``3rd_party/MeanFlow``'s separate
    ``final_layer_u``/``final_layer_v`` -- at MLP scale there is no
    meaningful cost to always computing both, so a second output head
    would only add complexity)."""

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
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be non-empty.")
        self.action_dim = action_dim

        input_dim = features_dim + action_dim + 2  # + tau, rho
        self.mlp = create_mlp(
            input_dim=input_dim,
            output_dim=2 * action_dim,
            net_arch=hidden_dims,
            activation_fn=resolve_activation(activation_fn, default=nn.ReLU),
            use_layer_norm=use_layer_norm,
            kernel_init=kernel_init,
        )

    def forward(
        self,
        features: torch.Tensor,
        x: torch.Tensor,
        tau: torch.Tensor,
        rho: torch.Tensor,
        *,
        return_v: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(torch.cat([features, x, tau, rho], dim=-1))
        u, v = out.split(self.action_dim, dim=-1)
        if not return_v:
            return u
        return u, v


def _adaptive_l2_loss(error: torch.Tensor, *, gamma: float, c: float) -> torch.Tensor:
    """Adaptive L2 loss (``sg(w) * ||error||^2``), ported verbatim from
    ``3rd_party/MeanFlow/meanflow.py::adaptive_l2_loss`` with the reduction
    generalized from image dims ``(1,2,3)`` to the flat action dim
    ``(-1,)`` -- these are equivalent reductions (both average the squared
    error over every non-batch dim)."""
    delta_sq = torch.mean(error**2, dim=-1)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (w.detach() * delta_sq).mean()


def mean_flow_loss_from_samples(
    net: MeanFlowActorField,
    features: torch.Tensor,
    actions: torch.Tensor,
    x_0: torch.Tensor,
    tau: torch.Tensor,
    rho: torch.Tensor,
    *,
    mode: MeanFlowMode = "i-meanflow",
    gamma: float = 0.0,
    c: float = 1e-2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MeanFlow loss given explicit samples (noise, times) -- mirrors
    ``flow_onestep_distill_loss``'s explicit-``noise`` shape (Milestone B)
    so this is directly testable against upstream without depending on this
    module's own sampling.

    ``adaptive_l2_c`` defaults to upstream's own value, calibrated for
    image-pixel-scale squared error -- unvalidated at action scale. If a
    training run's loss curve looks degenerate (flat, or scaled oddly),
    this is the first knob to check, not the time-convention math.
    """
    x_tau = (1 - tau) * x_0 + tau * actions
    v_target = actions - x_0

    u_pred, v_pred = net(features, x_tau, tau, rho, return_v=True)

    with torch.no_grad():
        _, v_c = net(features, x_tau, tau, rho, return_v=True)
        model_partial = partial(net, features, rho=rho, return_v=False)
        _, dudtau = torch.autograd.functional.jvp(
            model_partial,
            (x_tau, tau),
            (v_c, torch.ones_like(tau)),
            create_graph=False,
        )

    fm_loss = _adaptive_l2_loss(v_pred - v_target.detach(), gamma=gamma, c=c)

    # v_est: u reconstructed back into an instantaneous velocity via the
    # MeanFlow identity (dudtau already stopgrad'd -- computed under
    # torch.no_grad() above); used as the i-meanflow loss target directly
    # and, detached, for the mf_v_mse monitoring metric in both modes --
    # matches upstream's single unconditional `v_est` definition.
    v_est = u_pred - (rho - tau) * dudtau

    if mode == "meanflow":
        u_target = v_target + (rho - tau) * dudtau
        mf_loss = _adaptive_l2_loss(u_pred - u_target.detach(), gamma=gamma, c=c)
    elif mode == "i-meanflow":
        mf_loss = _adaptive_l2_loss(v_est - v_target.detach(), gamma=gamma, c=c)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    aux = {
        "fm_loss": fm_loss.detach(),
        "mf_loss": mf_loss.detach(),
        "mf_v_mse": ((v_est.detach() - v_target) ** 2).mean().detach(),
    }
    return mf_loss + fm_loss, aux
