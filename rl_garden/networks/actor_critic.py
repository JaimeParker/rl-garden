from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.networks.mlp import create_mlp


def get_actor_critic_arch(
    net_arch: Sequence[int] | dict[str, Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Resolve actor/critic hidden dims from a shared net_arch spec.

    Mirrors SB3 semantics:
      - list[int] -> same architecture for actor and critic
      - dict(pi=[...], qf=[...]) -> separate architectures
    """
    if isinstance(net_arch, dict):
        if "pi" not in net_arch or "qf" not in net_arch:
            raise ValueError("net_arch dict must contain both 'pi' and 'qf' keys.")
        return list(net_arch["pi"]), list(net_arch["qf"])

    return list(net_arch), list(net_arch)


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor for SAC/WSRL families."""

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int],
        *,
        use_layer_norm: bool = False,
        std_parameterization: Literal["exp", "uniform"] = "exp",
        log_std_mode: Literal["clamp", "tanh"] = "clamp",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if std_parameterization not in ("exp", "uniform"):
            raise ValueError(
                "std_parameterization must be 'exp' or 'uniform', "
                f"got {std_parameterization!r}"
            )
        if log_std_mode not in ("clamp", "tanh"):
            raise ValueError(
                "log_std_mode must be 'clamp' or 'tanh', "
                f"got {log_std_mode!r}"
            )

        self.std_parameterization = std_parameterization
        self.log_std_mode = log_std_mode
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        act_dim = int(np.prod(action_space.shape))
        self.trunk = create_mlp(
            input_dim=features_dim,
            output_dim=-1,
            net_arch=hidden_dims,
            use_layer_norm=use_layer_norm,
        )
        trunk_dim = hidden_dims[-1] if len(hidden_dims) > 0 else features_dim

        self.fc_mean = nn.Linear(trunk_dim, act_dim)
        if std_parameterization == "exp":
            self.fc_logstd = nn.Linear(trunk_dim, act_dim)
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
            raw_log_std = self.fc_logstd(x)
        else:
            assert self.log_stds is not None
            raw_log_std = self.log_stds.expand_as(mean)

        if self.log_std_mode == "tanh":
            log_std = torch.tanh(raw_log_std)
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
                log_std + 1
            )
        else:
            log_std = torch.clamp(raw_log_std, self.log_std_min, self.log_std_max)

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


class EnsembleQCritic(nn.Module):
    """Ensemble of Q(s, a) MLPs."""

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int],
        *,
        n_critics: int = 2,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if n_critics < 1:
            raise ValueError(f"n_critics must be >= 1, got {n_critics}")

        self.n_critics = n_critics
        act_dim = int(np.prod(action_space.shape))
        self.q_nets = nn.ModuleList(
            [
                create_mlp(
                    input_dim=features_dim + act_dim,
                    output_dim=1,
                    net_arch=hidden_dims,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(n_critics)
            ]
        )

    def forward(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x = torch.cat([features, actions], dim=-1)
        return tuple(q(x) for q in self.q_nets)

    def forward_all(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q_values = self.forward(features, actions)
        return torch.stack(q_values, dim=0)
