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

from typing import Optional, Sequence

import torch
import torch.nn as nn

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
