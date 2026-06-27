"""FlashSAC actor/critic policy.

Owns three nn.Modules:
  - actor   : FlashSACActor (inverted-residual + BN + RMSNorm + NormalTanhPolicy)
  - critic  : FlashSACDoubleCritic (ensemble categorical distributional Q)
  - critic_target : copy of critic, updated via polyak
No shared encoder — obs goes directly to actor/critic.
Asymmetric observations are supported: the first `actor_obs_dim` elements are
used as actor input; the full vector is used as critic input.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.networks.flash_sac_layers import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
    FlashSACBlock,
    FlashSACEmbedder,
    NormalTanhPolicy,
    UnitRMSNorm,
)


# ---------------------------------------------------------------------------
# Network modules (mirrors flash_rl/agents/flashSAC/network.py)
# ---------------------------------------------------------------------------


class FlashSACActor(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(hidden_dim=hidden_dim, action_dim=action_dim)

    def get_mean_and_std(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedder(observations, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return self.predictor.get_mean_and_std(x, training)

    def forward(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.embedder(observations, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return self.predictor(x, training)


class FlashSACDoubleCritic(nn.Module):
    """Ensemble categorical distributional double Q-critic."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
    ):
        super().__init__()
        self.num_qs = num_qs
        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((observations, actions), dim=-1)  # (B, obs+act)
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)  # (num_qs, B, obs+act)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return self.predictor(x, training)


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class FlashSACPolicy(nn.Module):
    """Policy container for FlashSAC — holds actor, critic, critic_target."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        actor_hidden_dim: int = 128,
        actor_num_blocks: int = 2,
        critic_hidden_dim: int = 256,
        critic_num_blocks: int = 2,
        num_bins: int = 101,
        min_v: float = -5.0,
        max_v: float = 5.0,
        asymmetric_obs_dim: int = 0,
    ) -> None:
        super().__init__()
        obs_dim = int(observation_space.shape[0])
        action_dim = int(action_space.shape[0])

        self.asymmetric_obs_dim = asymmetric_obs_dim
        # actor sees first `actor_obs_dim` dimensions; critic sees full obs
        self.actor_obs_dim = obs_dim - asymmetric_obs_dim if asymmetric_obs_dim > 0 else obs_dim
        self.critic_obs_dim = obs_dim

        self.actor = FlashSACActor(
            num_blocks=actor_num_blocks,
            input_dim=self.actor_obs_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim,
        )
        critic_input_dim = self.critic_obs_dim + action_dim
        self.critic = FlashSACDoubleCritic(
            num_blocks=critic_num_blocks,
            input_dim=critic_input_dim,
            hidden_dim=critic_hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )
        self.critic_target = FlashSACDoubleCritic(
            num_blocks=critic_num_blocks,
            input_dim=critic_input_dim,
            hidden_dim=critic_hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    # --- observation slicing ---

    def actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Return actor's view of obs (sliced when asymmetric)."""
        if self.asymmetric_obs_dim > 0:
            return obs[:, : self.actor_obs_dim]
        return obs

    # --- public inference interface ---

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action for env interaction (no grad)."""
        actor_obs = self.actor_obs(obs)
        if deterministic:
            mean, _ = self.actor.get_mean_and_std(actor_obs, training=False)
            return torch.tanh(mean)
        actions, _ = self.actor(actor_obs, training=False)
        return actions

    def extract_features(self, obs: torch.Tensor, stop_gradient: bool = False) -> torch.Tensor:
        """Return obs for critic input (full obs; detach if requested)."""
        if stop_gradient:
            return obs.detach()
        return obs

    def min_q_value(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        subsample_size: None = None,
        target: bool = False,
    ) -> torch.Tensor:
        """Return elementwise min Q over critics, shape (B, 1)."""
        net = self.critic_target if target else self.critic
        with torch.no_grad():
            qs, _ = net(observations=features, actions=actions, training=False)
        return torch.minimum(qs[0], qs[1]).unsqueeze(-1)

    def actor_action_log_prob(
        self, obs: torch.Tensor, stop_gradient: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (action, log_prob, features) for SACCore compatibility."""
        actor_obs = self.actor_obs(obs)
        actions, info = self.actor(actor_obs, training=True)
        return actions, info["log_prob"], obs

    def actor_diagnostics(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Entropy diagnostic used by SACCore._actor_diagnostics."""
        with torch.no_grad():
            actor_obs = self.actor_obs(obs)
            _, info = self.actor(actor_obs, training=False)
            log_prob = info["log_prob"]
        return {"entropy": -log_prob.mean()}

    # --- weight normalization ---

    @torch.no_grad()
    def normalize_weights(self) -> None:
        """Call normalize_parameters() on every eligible sub-module."""
        for module in self.modules():
            if module is not self and hasattr(module, "normalize_parameters"):
                module.normalize_parameters()
