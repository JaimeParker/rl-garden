"""Flow-matching Behavioral Cloning policy with a shared feature extractor.

Actor-only, pure imitation learning: no critic, no distillation head, no
Q-guidance. Reuses ``ActorVectorField`` (``rl_garden/networks/actor_vector_field.py``)
directly -- the same network FQL's ``actor_bc_flow`` (``fql_policy.py``) already
is, minus everything else FQL adds on top. The BC-flow regression loss below
is the exact triplet from ``fql.py``'s actor update (``x_0``, continuous
``t``, ``x_t = (1-t)*x_0 + t*actions``, ``vel_target = actions - x_0``),
confirmed by direct comparison to be standard conditional (CondOT)
flow-matching regression, not an FQL-specific approximation.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import Activation, ActorVectorField, KernelInit
from rl_garden.policies.base import BasePolicy


class FlowBCPolicy(BasePolicy):
    """Actor-only policy for flow-matching Behavioral Cloning."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] = (512, 512, 512, 512),
        *,
        use_layer_norm: bool = False,
        kernel_init: Optional[KernelInit] = None,
        activation_fn: Optional[Activation] = None,
        flow_steps: int = 10,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "FlowBCPolicy requires a Box action space."
        if flow_steps < 1:
            raise ValueError(f"flow_steps must be >= 1, got {flow_steps}.")
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.flow_steps = flow_steps

        fd = features_extractor.features_dim
        action_dim = int(np.prod(action_space.shape))
        self.actor_bc_flow = ActorVectorField(
            fd,
            action_dim,
            hidden_dims=list(net_arch),
            use_time_conditioning=True,
            use_layer_norm=use_layer_norm,
            kernel_init=kernel_init,
            activation_fn=activation_fn,
        )

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        # Flow matching has no separate deterministic eval path: sampling
        # always integrates from a fresh N(0,1) latent (like a GAN's z), not
        # exploration noise to zero out. `deterministic` is accepted for
        # BasePolicy contract compatibility but has no effect here --
        # matches FQLPolicy.predict()'s identical stance.
        del deterministic
        features = self.extract_features(obs)
        noise = torch.randn(
            features.shape[0],
            self.actor_bc_flow.action_dim,
            device=features.device,
            dtype=features.dtype,
        )
        return self.actor_bc_flow.integrate(
            features, noise, self.flow_steps, low=self.action_low, high=self.action_high
        )

    def bc_flow_loss(self, obs: Obs, actions: torch.Tensor) -> torch.Tensor:
        """Conditional flow-matching regression loss (CondOT path): straight
        line from noise to the expert action, MSE against the constant
        target velocity ``actions - x_0``."""
        features = self.extract_features(obs, stop_gradient=False)
        batch_size = actions.shape[0]
        device, dtype = actions.device, actions.dtype
        x_0 = torch.randn_like(actions)
        t = torch.rand(batch_size, 1, device=device, dtype=dtype)
        x_t = (1 - t) * x_0 + t * actions
        vel_target = actions - x_0
        pred_vel = self.actor_bc_flow(features, x_t, t)
        return F.mse_loss(pred_vel, vel_target)

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        """All trainable parameters: encoder + flow vector field."""
        return self.parameters()
