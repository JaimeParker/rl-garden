"""MeanFlow Behavioral Cloning policy with a shared feature extractor.

Actor-only, pure imitation learning -- mirrors ``FlowBCPolicy``'s shape
(no critic, no distillation head, no Q-guidance), swapping ``ActorVectorField``
for ``MeanFlowActorField`` (``rl_garden/networks/mean_flow_field.py``) and
``bc_flow_loss`` for ``mean_flow_loss`` (the MeanFlow identity loss, see that
module's docstring for the time-convention re-derivation).

Unlike ``FlowBCPolicy``, whose ``flow_steps`` (default 10) integrates a
*teacher* velocity field, this policy's default ``num_sample_steps=1``
performs true one-step generation via the trained *average*-velocity head
``u`` -- the entire point of MeanFlow is one-step sampling without a
separate distillation phase.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.mean_flow_field import (
    MeanFlowActorField,
    MeanFlowMode,
    mean_flow_loss_from_samples,
)
from rl_garden.policies.base import BasePolicy


class MeanFlowBCPolicy(BasePolicy):
    """Actor-only policy for MeanFlow Behavioral Cloning."""

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
        num_sample_steps: int = 1,
        mode: MeanFlowMode = "i-meanflow",
        time_dist_mu: float = 0.4,
        time_dist_sigma: float = 1.0,
        adaptive_l2_gamma: float = 0.0,
        adaptive_l2_c: float = 1e-2,
    ) -> None:
        super().__init__()
        assert isinstance(
            action_space, spaces.Box
        ), "MeanFlowBCPolicy requires a Box action space."
        if num_sample_steps < 1:
            raise ValueError(f"num_sample_steps must be >= 1, got {num_sample_steps}.")
        if mode not in ("meanflow", "i-meanflow"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.num_sample_steps = num_sample_steps
        self.mode = mode
        self.time_dist_mu = time_dist_mu
        self.time_dist_sigma = time_dist_sigma
        self.adaptive_l2_gamma = adaptive_l2_gamma
        self.adaptive_l2_c = adaptive_l2_c

        fd = features_extractor.features_dim
        action_dim = int(np.prod(action_space.shape))
        self.actor_mean_flow = MeanFlowActorField(
            fd,
            action_dim,
            hidden_dims=list(net_arch),
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
        # MeanFlow has no separate deterministic eval path: sampling always
        # integrates from a fresh N(0,1) latent, same stance as FlowBCPolicy
        # / FQLPolicy.
        del deterministic
        features = self.extract_features(obs)
        batch_size = features.shape[0]
        device, dtype = features.device, features.dtype
        x = torch.randn(
            batch_size, self.actor_mean_flow.action_dim, device=device, dtype=dtype
        )
        taus = torch.linspace(0.0, 1.0, self.num_sample_steps + 1, device=device, dtype=dtype)
        for i in range(self.num_sample_steps):
            tau = taus[i].expand(batch_size, 1)
            rho = taus[i + 1].expand(batch_size, 1)
            u = self.actor_mean_flow(features, x, tau, rho, return_v=False)
            x = x + (rho - tau) * u
        return x.clamp(self.action_low, self.action_high)

    def sample_train_times(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(tau, rho)`` pairs, ``tau <= rho``, both in ``[0, 1]``.

        Ported from ``3rd_party/MeanFlow``'s ``sample_t_r`` (two i.i.d.
        ``sigmoid(Normal(mu, sigma))`` draws, sorted), with ``mu`` sign-flipped
        (``time_dist_mu`` positive vs. upstream's negative default) to account
        for this port's flipped time axis: ``1 - sigmoid(z) = sigmoid(-z)``,
        so sampling ``sigmoid(Normal(+mu, sigma))`` here reproduces upstream's
        ``sigmoid(Normal(-mu, sigma))`` marginal under the ``tau = 1 - t``
        mapping.
        """
        samples = torch.randn(batch_size, 2, device=device, dtype=dtype)
        samples = samples * self.time_dist_sigma + self.time_dist_mu
        samples = torch.sigmoid(samples)
        tau = samples.amin(dim=1, keepdim=True)
        rho = samples.amax(dim=1, keepdim=True)
        return tau, rho

    def mean_flow_loss(self, obs: Obs, actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MeanFlow identity loss (``fm_loss + mf_loss``); see
        ``mean_flow_field.mean_flow_loss_from_samples`` for the formula and
        ``mean_flow_field``'s module docstring for the time-convention
        derivation. Returns ``(total_loss, aux)`` where ``aux`` carries
        upstream's own logged breakdown (``fm_loss``, ``mf_loss``,
        ``mf_v_mse``)."""
        features = self.extract_features(obs, stop_gradient=False)
        batch_size = actions.shape[0]
        device, dtype = actions.device, actions.dtype
        x_0 = torch.randn_like(actions)
        tau, rho = self.sample_train_times(batch_size, device, dtype)
        return mean_flow_loss_from_samples(
            self.actor_mean_flow,
            features,
            actions,
            x_0,
            tau,
            rho,
            mode=self.mode,
            gamma=self.adaptive_l2_gamma,
            c=self.adaptive_l2_c,
        )

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        """All trainable parameters: encoder + mean-flow vector field."""
        return self.parameters()
