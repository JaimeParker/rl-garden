"""DDPG actor/critic policy for DrQ-v2."""
from __future__ import annotations

import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks.ddpg_actor import DDPGActor
from rl_garden.networks.ddpg_critic import DrQv2Critic
from rl_garden.policies.base import BasePolicy


class DDPGPolicy(BasePolicy):
    """DrQ-v2 DDPG policy: shared encoder + actor + double-Q critic.

    Key differences from ``SACPolicy``
    ----------------------------------
    * **No entropy / alpha**.  DDPG has no entropy regularisation.
    * **No learnable std**.  Actor std comes from an external schedule.
    * **No log-prob**.  ``predict()`` returns only the action.
    * Critic target does **not** subtract ``alpha * log_prob``.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        feature_dim: int = 50,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "DDPG requires a Box action space."

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor

        fd = features_extractor.features_dim

        self.critic = DrQv2Critic(
            features_dim=fd,
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )
        self.critic_target = DrQv2Critic(
            features_dim=fd,
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # --- Actor (DDPG-style, external std) ---
        self.actor = DDPGActor(
            features_dim=fd,
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(
        self,
        obs: Obs,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        return self._extract_features(obs, stop_gradient=stop_gradient)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        obs: Obs,
        deterministic: bool = False,
        std: float = 0.0,
    ) -> torch.Tensor:
        features = self.extract_features(obs)
        if deterministic:
            return self.actor.deterministic_action(features)
        dist = self.actor(features, std)
        return dist.sample(clip=None)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def actor_action(
        self,
        obs: Obs,
        std: float,
        noise_clip: float | None = None,
        stop_gradient: bool = True,
    ) -> torch.Tensor:
        """Sample an action from the actor (no log-prob)."""
        features = self.extract_features(obs, stop_gradient=stop_gradient)
        return self.actor_action_from_features(features, std, noise_clip=noise_clip)

    def actor_action_from_features(
        self,
        features: torch.Tensor,
        std: float,
        noise_clip: float | None = None,
    ) -> torch.Tensor:
        dist = self.actor(features, std)
        return dist.sample(clip=noise_clip)

    def q_values_all(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        target: bool = False,
    ) -> torch.Tensor:
        net = self.critic_target if target else self.critic
        return net.forward_all(features, actions)

    def min_q_value(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        target: bool = True,
    ) -> torch.Tensor:
        return self.q_values_all(features, actions, target=target).min(dim=0).values

    # ------------------------------------------------------------------
    # Parameter groups for optimizers
    # ------------------------------------------------------------------

    def critic_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from self.features_extractor.parameters()

    def actor_parameters(self):
        yield from self.actor.parameters()
