"""WSRL actor/critic policy with Q-ensemble (REDQ) and CQL support.

Extends SACPolicy to support:
  - Variable n_critics (2-10+) for REDQ ensemble
  - Critic subsampling for efficient target computation
  - Optional CQL alpha Lagrange multiplier for auto-tuning
  - Batch Q-value evaluation for CQL loss computation

Key differences from SACPolicy:
  - ContinuousCritic supports n_critics > 2
  - Added q_values_subsampled() for REDQ target computation
  - Added cql_alpha_lagrange network (optional)
  - Actor and critic use same interface as SACPolicy for compatibility
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.policies.base import BasePolicy

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0


def _mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    *,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    c = in_dim
    for h in hidden:
        layers.append(nn.Linear(c, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        c = h
    layers.append(nn.Linear(c, out_dim))
    return nn.Sequential(*layers)


def _softplus_inverse(x: float) -> float:
    return float(np.log(np.expm1(x)))


class Actor(nn.Module):
    """Policy network with tanh-squashed Gaussian output."""

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int] = (256, 256),
        use_layer_norm: bool = False,
        std_parameterization: Literal["exp", "uniform"] = "exp",
    ) -> None:
        super().__init__()
        if std_parameterization not in ("exp", "uniform"):
            raise ValueError(
                "std_parameterization must be 'exp' or 'uniform', "
                f"got {std_parameterization!r}"
            )
        self.std_parameterization = std_parameterization
        act_dim = int(np.prod(action_space.shape))
        layers: list[nn.Module] = []
        c = features_dim
        for h in hidden_dims:
            layers.append(nn.Linear(c, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            c = h
        self.trunk = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(c, act_dim)
        if std_parameterization == "exp":
            self.fc_logstd = nn.Linear(c, act_dim)
            self.log_stds = None
        else:
            self.fc_logstd = None
            self.log_stds = nn.Parameter(torch.zeros(act_dim))

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(features)
        mean = self.fc_mean(x)
        if self.std_parameterization == "exp":
            assert self.fc_logstd is not None
            log_std = self.fc_logstd(x)
        else:
            assert self.log_stds is not None
            log_std = self.log_stds.expand_as(mean)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def action_log_prob(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(features)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        mean, _ = self(features)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class ContinuousCritic(nn.Module):
    """Ensemble of Q(s, a) MLPs with support for variable ensemble size (REDQ)."""

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int] = (256, 256, 256),
        n_critics: int = 10,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.n_critics = n_critics
        act_dim = int(np.prod(action_space.shape))
        self.q_nets = nn.ModuleList(
            [
                _mlp(
                    features_dim + act_dim,
                    hidden_dims,
                    out_dim=1,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(n_critics)
            ]
        )

    def forward(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Returns tuple of Q-values from each critic."""
        x = torch.cat([features, actions], dim=-1)
        return tuple(q(x) for q in self.q_nets)

    def forward_all(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Returns stacked Q-values as tensor of shape (n_critics, batch_size, 1)."""
        q_values = self.forward(features, actions)
        return torch.stack(q_values, dim=0)


class CQLAlphaLagrange(nn.Module):
    """Lagrange multiplier for auto-tuning CQL alpha."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        if init_value <= 0:
            raise ValueError("CQL alpha Lagrange init value must be positive.")
        self.log_alpha = nn.Parameter(
            torch.tensor(_softplus_inverse(init_value), dtype=torch.float32)
        )

    def forward(self) -> torch.Tensor:
        return F.softplus(self.log_alpha)


class WSRLPolicy(BasePolicy):
    """WSRL policy with Q-ensemble (REDQ) and optional CQL alpha Lagrange multiplier.

    Key features:
    - Variable n_critics (default 10 for REDQ)
    - Critic subsampling for efficient target computation
    - Optional CQL alpha auto-tuning via Lagrange multiplier
    - Compatible with SACPolicy interface
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        actor_hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        n_critics: int = 10,
        critic_subsample_size: Optional[int] = 2,
        use_cql_alpha_lagrange: bool = False,
        cql_alpha_lagrange_init: float = 1.0,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        std_parameterization: Literal["exp", "uniform"] = "exp",
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "WSRL requires a Box action space."
        assert n_critics >= 2, f"n_critics must be >= 2, got {n_critics}"
        if critic_subsample_size is not None:
            assert critic_subsample_size <= n_critics, (
                f"critic_subsample_size ({critic_subsample_size}) must be <= n_critics ({n_critics})"
            )

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size

        fd = features_extractor.features_dim
        self.actor = Actor(
            fd,
            action_space,
            hidden_dims=actor_hidden_dims,
            use_layer_norm=actor_use_layer_norm,
            std_parameterization=std_parameterization,
        )
        self.critic = ContinuousCritic(
            fd,
            action_space,
            hidden_dims=critic_hidden_dims,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
        )
        # Separate target critic
        self.critic_target = ContinuousCritic(
            fd,
            action_space,
            hidden_dims=critic_hidden_dims,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Optional CQL alpha Lagrange multiplier
        self.use_cql_alpha_lagrange = use_cql_alpha_lagrange
        if use_cql_alpha_lagrange:
            self.cql_alpha_lagrange = CQLAlphaLagrange(init_value=cql_alpha_lagrange_init)
        else:
            self.cql_alpha_lagrange = None

    # --- feature extraction helpers ---

    def extract_features(self, obs: Obs, detach: bool = False) -> torch.Tensor:
        features = self.features_extractor(obs)
        if detach:
            features = features.detach()
        return features

    # --- public inference API ---

    def forward(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        return self.predict(obs, deterministic=deterministic)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)
        if deterministic:
            return self.actor.deterministic_action(features)
        action, _ = self.actor.action_log_prob(features)
        return action

    # --- helpers for WSRL.train() ---

    def actor_action_log_prob(
        self, obs: Obs, detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy and return action, log_prob, features."""
        features = self.extract_features(obs, detach=detach_encoder)
        action, log_prob = self.actor.action_log_prob(features)
        return action, log_prob, features

    def q_values(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """Returns tuple of Q-values from all critics."""
        net = self.critic_target if target else self.critic
        return net(features, actions)

    def q_values_all(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        """Returns stacked Q-values as tensor of shape (n_critics, batch_size, 1)."""
        net = self.critic_target if target else self.critic
        return net.forward_all(features, actions)

    def q_values_subsampled(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        subsample_size: Optional[int] = None,
        target: bool = True,
    ) -> torch.Tensor:
        """Returns subsampled Q-values for REDQ target computation.

        Args:
            features: State features
            actions: Actions to evaluate
            subsample_size: Number of critics to subsample (default: self.critic_subsample_size)
            target: Use target network (default: True)

        Returns:
            Tensor of shape (subsample_size, batch_size, 1)
        """
        if subsample_size is None:
            subsample_size = self.critic_subsample_size

        # Get all Q-values
        q_all = self.q_values_all(features, actions, target=target)  # (n_critics, batch, 1)

        # Subsample critics
        if subsample_size is not None and subsample_size < self.n_critics:
            indices = torch.randint(
                0, self.n_critics, (subsample_size,), device=q_all.device
            )
            q_subsampled = q_all[indices]  # (subsample_size, batch, 1)
        else:
            q_subsampled = q_all

        return q_subsampled

    def min_q_value(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        subsample_size: Optional[int] = None,
        target: bool = True,
    ) -> torch.Tensor:
        """Returns minimum Q-value across (subsampled) ensemble.

        Args:
            features: State features
            actions: Actions to evaluate
            subsample_size: Number of critics to subsample (default: self.critic_subsample_size)
            target: Use target network (default: True)

        Returns:
            Tensor of shape (batch_size, 1)
        """
        q_subsampled = self.q_values_subsampled(
            features, actions, subsample_size=subsample_size, target=target
        )
        return q_subsampled.min(dim=0)[0]  # (batch, 1)

    # --- CQL alpha Lagrange multiplier ---

    def get_cql_alpha(self) -> torch.Tensor:
        """Returns CQL alpha value (either from Lagrange multiplier or fixed)."""
        if self.use_cql_alpha_lagrange:
            return self.cql_alpha_lagrange()
        else:
            raise ValueError("CQL alpha Lagrange multiplier not enabled")

    # --- parameter groups for optimizers ---

    def critic_and_encoder_parameters(self):
        """Encoder trained via Q-loss (matches sac_rgbd.py pattern)."""
        yield from self.critic.parameters()
        yield from self.features_extractor.parameters()

    def actor_parameters(self):
        """Actor-only; encoder is detached on the actor path."""
        yield from self.actor.parameters()

    def cql_alpha_lagrange_parameters(self):
        """CQL alpha Lagrange multiplier parameters."""
        if self.use_cql_alpha_lagrange:
            yield from self.cql_alpha_lagrange.parameters()
        else:
            return iter([])
