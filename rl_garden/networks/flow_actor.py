"""Flow-matching actor for SAC (SACFlow).

Ports RLinf's ``JaxFlowTActor`` fixed-noise branch (``noise_std_head=False``,
RLinf's own default -- see
``RLinf/rlinf/models/embodiment/modules/flow_actor.py``), swapping
its transformer-decoder backbone for a plain MLP vector field conditioned on
``(features, x_t, t)``, the same conditioning FQL's ``ActorVectorField`` uses
(``3rd_party/fql/utils/networks.py``). Everything else -- the Euler sampling
loop, the per-step Gaussian log-prob accumulation, and the tanh-Jacobian
correction at the end -- follows RLinf's math directly.

One departure from RLinf: RLinf uses a smaller fixed noise std at rollout
(0.02) than at training (0.3) via a ``train`` flag threaded into its
``sac_forward``. rl-garden's ``action_log_prob(features)`` contract (see
``SquashedGaussianActor``) has no such flag and is used identically for
rollout and training-time sampling, so this actor uses a single ``noise_std``
throughout. ``deterministic_action`` (needed for ``SACPolicy.predict(deterministic=True)``,
which has no RLinf/flow-matching precedent) is defined as the zero-noise
Euler trajectory: integrate using the predicted velocity mean only, then
tanh-squash.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions.normal import Normal

from rl_garden.networks.mlp import KernelInit, create_mlp


class FlowMatchingActor(nn.Module):
    """Flow-matching actor trained with plain SAC's ``alpha*log_pi - Q`` loss.

    Drop-in replacement for ``SquashedGaussianActor`` in ``SACPolicy``-family
    classes: exposes ``action_log_prob``, ``deterministic_action``, and
    ``action_scale``/``action_bias`` buffers with the same names/shapes.
    """

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int],
        *,
        denoising_steps: int = 4,
        noise_std: float = 0.3,
        use_layer_norm: bool = False,
        kernel_init: Optional[KernelInit] = "xavier_uniform",
    ) -> None:
        super().__init__()
        if denoising_steps < 1:
            raise ValueError(f"denoising_steps must be >= 1, got {denoising_steps}")
        if noise_std <= 0:
            raise ValueError(f"noise_std must be > 0, got {noise_std}")

        self.action_dim = int(np.prod(action_space.shape))
        self.denoising_steps = denoising_steps
        self.noise_std = noise_std

        trunk_input_dim = features_dim + self.action_dim + 1  # + 1 for scalar time
        self.trunk = create_mlp(
            input_dim=trunk_input_dim,
            output_dim=-1,
            net_arch=hidden_dims,
            use_layer_norm=use_layer_norm,
            kernel_init=kernel_init,
        )
        trunk_out_dim = hidden_dims[-1] if len(hidden_dims) > 0 else trunk_input_dim
        self.fc_velocity = nn.Linear(trunk_out_dim, self.action_dim)
        # create_mlp() only initializes self.trunk; RLinf's own weight init
        # (Xavier-uniform weight, zero bias on every nn.Linear, unconditional
        # -- flow_actor.py's _init_weights) also covers its output head, so
        # match that here for the same default.
        if kernel_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.fc_velocity.weight)
            nn.init.zeros_(self.fc_velocity.bias)

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def _velocity_mean(self, features: torch.Tensor, x: torch.Tensor, step: int) -> torch.Tensor:
        time = torch.full(
            (features.shape[0], 1),
            step / self.denoising_steps,
            device=features.device,
            dtype=features.dtype,
        )
        trunk_input = torch.cat([features, x, time], dim=-1)
        return self.fc_velocity(self.trunk(trunk_input))

    def action_log_prob(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_t = 1.0 / self.denoising_steps
        x = torch.randn(
            features.shape[0], self.action_dim, device=features.device, dtype=features.dtype
        )
        init_dist = Normal(torch.zeros_like(x), torch.ones_like(x))
        total_log_prob = init_dist.log_prob(x).sum(-1, keepdim=True)

        for step in range(self.denoising_steps):
            velocity_mean = self._velocity_mean(features, x, step)
            x_next_mean = x + velocity_mean * delta_t
            noise = torch.randn_like(x_next_mean)
            x = x_next_mean + self.noise_std * noise
            # x - x_next_mean == noise_std * noise regardless of theta (noise_std is a
            # fixed hyperparameter, not network output), so this term is analytically
            # constant in theta -- it contributes zero gradient. The actor's entropy
            # term (alpha * log_prob) therefore backprops into the network only through
            # the tanh-Jacobian correction below; -Q(s, action) is unaffected and still
            # backprops through every step's velocity_mean.
            step_log_prob = Normal(x_next_mean, self.noise_std).log_prob(x).sum(-1, keepdim=True)
            total_log_prob = total_log_prob + step_log_prob

        y_t = torch.tanh(x)
        action = y_t * self.action_scale + self.action_bias
        tanh_correction = torch.log(
            self.action_scale * (1 - y_t.pow(2)) + 1e-6
        ).sum(-1, keepdim=True)
        log_prob = total_log_prob - tanh_correction
        return action, log_prob

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        delta_t = 1.0 / self.denoising_steps
        x = torch.randn(
            features.shape[0], self.action_dim, device=features.device, dtype=features.dtype
        )
        for step in range(self.denoising_steps):
            velocity_mean = self._velocity_mean(features, x, step)
            x = x + velocity_mean * delta_t
        return torch.tanh(x) * self.action_scale + self.action_bias
