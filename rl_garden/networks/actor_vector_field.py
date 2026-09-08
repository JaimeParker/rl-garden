"""Flow-matching vector-field network for FQL (Flow Q-Learning).

Ports FQL's ``ActorVectorField`` (see ``utils/networks.py``):
one MLP class instantiated twice with disjoint parameters -- a time-
conditioned "teacher" (``obs, x_t, t -> velocity``) and a time-free
"one-step" "student" (``obs, noise -> velocity``, whose single output is
used directly as the action, not integrated further). Because the two
instances take different input widths (the teacher's input includes a
scalar time slot, the student's doesn't), which of the two roles an
instance plays is fixed at construction time via ``use_time_conditioning``,
not a per-call flag.

Unlike FQL's reference (D4RL/OGBench actions pre-normalized to [-1,1],
hard-clipped with a literal ``jnp.clip(actions, -1, 1)`` at every call
site), this port clips to the actual ``action_space`` bounds -- rl-garden
does not assume actions are pre-normalized to [-1,1].
"""
from __future__ import annotations

import math
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_garden.networks.mlp import Activation, KernelInit, create_mlp, resolve_activation


class ActorVectorField(nn.Module):
    def __init__(
        self,
        features_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        *,
        use_time_conditioning: bool,
        use_layer_norm: bool = False,
        kernel_init: Optional[KernelInit] = None,
        activation_fn: Optional[Activation] = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.use_time_conditioning = use_time_conditioning

        input_dim = features_dim + action_dim + (1 if use_time_conditioning else 0)
        # activate_final=False matches FQL: a plain linear output layer, no
        # activation on the last layer.
        self.mlp = create_mlp(
            input_dim=input_dim,
            output_dim=action_dim,
            net_arch=hidden_dims,
            activation_fn=resolve_activation(activation_fn, default=nn.ReLU),
            use_layer_norm=use_layer_norm,
            kernel_init=kernel_init,
        )

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_time_conditioning:
            if times is None:
                raise ValueError("This ActorVectorField instance requires `times`.")
            inputs = torch.cat([features, actions, times], dim=-1)
        else:
            if times is not None:
                raise ValueError("This ActorVectorField instance does not take `times`.")
            inputs = torch.cat([features, actions], dim=-1)
        return self.mlp(inputs)

    def integrate(
        self,
        features: torch.Tensor,
        x_0: torch.Tensor,
        num_steps: int,
        *,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euler-integrate this vector field from `x_0` over `num_steps`,
        returning the final action. Requires a time-conditioned instance
        (the teacher)."""
        if not self.use_time_conditioning:
            raise ValueError("integrate() requires a time-conditioned ActorVectorField.")
        x = x_0
        for step in range(num_steps):
            t = torch.full(
                (features.shape[0], 1),
                step / num_steps,
                device=features.device,
                dtype=features.dtype,
            )
            velocity = self(features, x, t)
            x = x + velocity / num_steps
        if low is not None and high is not None:
            x = x.clamp(low, high)
        return x


def flow_onestep_distill_loss(
    teacher: ActorVectorField,
    student_action: torch.Tensor,
    features: torch.Tensor,
    noise: torch.Tensor,
    num_steps: int,
    *,
    low: Optional[torch.Tensor] = None,
    high: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-step flow-matching distillation loss (FQL's ``distill_loss``):
    MSE between a one-step student action and a no-grad multi-step Euler
    unroll of the time-conditioned `teacher`."""
    with torch.no_grad():
        target = teacher.integrate(features, noise, num_steps, low=low, high=high)
    return F.mse_loss(student_action, target)


def flow_sde_step(
    velocity: torch.Tensor,
    x: torch.Tensor,
    tau: torch.Tensor,
    dtau: float,
    *,
    sde_type: Literal["sde", "cps"] = "cps",
    noise_level: float = 0.7,
    clip_std_min: float = 0.0067,
    sigma_safe_max: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One stochastic flow-matching transition step, ported from RL-100's
    ``FlowMatchSchedulerExtended._compute_step`` (``3rd_party/RL-100/RL-100/
    rl_100/unidpg/diffusion_policy/diffusers_patch/flow_match_scheduler.py``).

    RL-100's scheduler is written in ``sigma``-space (``sigma=1`` noise,
    ``sigma=0`` data; velocity points data->noise). This repo's
    ``ActorVectorField``/FQL convention is the opposite (``tau=0`` noise,
    ``tau=1`` data; velocity points noise->data, see ``fql.py``'s
    ``vel_target = action - x_0``). The formulas below are re-derived
    directly in this repo's ``tau``-space (substituting ``sigma = 1 - tau``,
    ``v_RL100 = -velocity``) rather than transcribed as-is -- a literal port
    would silently integrate every step backwards. Verified: at
    ``noise_level=0`` (``std -> 0``) both variants collapse exactly to
    ``ActorVectorField.integrate``'s existing Euler step ``x + velocity*dtau``.

    Returns ``(mean, std)`` of the transition distribution over the next
    chain state; both already include ``clip_std_min`` (applied to keep
    sampling and log-prob evaluation consistent -- see the final-step note
    below).

    ``clip_std_min`` defaults to RL-100's actual training-config value
    (``config/rl100_3d_flow.yaml``: ``0.0067``), not its scheduler class's
    unguarded default of ``0.0``: at the final step (``tau_next=1``), the
    ``"cps"`` variant's ``std_dev_t = sigma_next * sin(...) = 0`` regardless
    of ``noise_level`` -- an unclamped std there makes ``flow_logprob``
    divide by a near-zero denominator, producing an astronomically
    large-magnitude ratio term that would dominate every PPO minibatch.
    """
    sigma = 1.0 - tau
    tau_next = tau + dtau
    sigma_next = 1.0 - tau_next

    if sde_type == "sde":
        sigma_safe = sigma.clamp(max=sigma_safe_max)
        std_dev_t = torch.sqrt(sigma_safe / (1.0 - sigma_safe)) * noise_level
        mean = x * (1.0 - std_dev_t**2 / (2.0 * sigma_safe) * dtau) + velocity * (
            1.0 + std_dev_t**2 * (1.0 - sigma_safe) / (2.0 * sigma_safe)
        ) * dtau
        std = std_dev_t * math.sqrt(dtau)
    elif sde_type == "cps":
        std_dev_t = sigma_next * math.sin(noise_level * math.pi / 2)
        pred_data = x + sigma * velocity
        pred_noise = x - tau * velocity
        std = std_dev_t
        mean = pred_data * tau_next + pred_noise * torch.sqrt(
            (sigma_next**2 - std_dev_t**2).clamp(min=0.0)
        )
    else:
        raise ValueError(f"Unknown sde_type: {sde_type!r}")

    if clip_std_min > 0:
        std = std.clamp(min=clip_std_min)
    return mean, std


def flow_logprob(
    next_x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    mode: Literal["gaussian", "pseudo"] = "gaussian",
) -> torch.Tensor:
    """Elementwise (no reduction) log-density of ``next_x`` under the
    ``flow_sde_step`` transition, ported from RL-100's
    ``FlowMatchSchedulerExtended._compute_logprob``. ``"pseudo"`` is not a
    true density (``ratio = exp(delta_logprob)`` under it is not a real
    probability ratio) -- kept only as an opt-in RL-100-parity flag; this
    port's own default is ``"gaussian"``."""
    if mode == "gaussian":
        std_safe = std.clamp(min=1e-12)
        return (
            -((next_x - mean) ** 2) / (2.0 * std_safe**2)
            - torch.log(std_safe)
            - 0.5 * math.log(2 * math.pi)
        )
    elif mode == "pseudo":
        action_dim = next_x.shape[-1]
        return -((next_x - mean) ** 2) / action_dim
    else:
        raise ValueError(f"Unknown logprob mode: {mode!r}")
