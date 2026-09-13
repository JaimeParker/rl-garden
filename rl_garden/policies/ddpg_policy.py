"""DDPG actor/critic policy for DrQ-v2."""
from __future__ import annotations

from typing import Optional

import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks.ddpg_actor import DDPGActor
from rl_garden.networks.ddpg_critic import DrQv2Critic
from rl_garden.policies.base import BasePolicy, EncoderSharing


class DDPGPolicy(BasePolicy):
    """DrQ-v2 DDPG policy: actor/critic extractor(s) + actor + double-Q critic.

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
        *,
        actor_extractor: BaseFeaturesExtractor,
        critic_extractor: Optional[BaseFeaturesExtractor] = None,
        encoder_sharing: EncoderSharing = "shared_critic_grad",
        feature_dim: int = 50,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            actor_extractor=actor_extractor,
            critic_extractor=critic_extractor,
            encoder_sharing=encoder_sharing,
        )
        assert isinstance(action_space, spaces.Box), "DDPG requires a Box action space."

        actor_fd = self.actor_features_dim
        critic_fd = self.critic_features_dim

        self.critic = DrQv2Critic(
            features_dim=critic_fd,
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )
        self.critic_target = DrQv2Critic(
            features_dim=critic_fd,
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # --- Actor (DDPG-style, external std) ---
        self.actor = DDPGActor(
            features_dim=actor_fd,
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
        """Raw actor-extractor access with an explicit ``stop_gradient`` --
        an escape hatch for diagnostics/back-compat callers (e.g. rollout
        action selection) that need to pick the flag themselves. ``DDPG``'s
        own training loop uses ``extract_actor_features``/
        ``extract_critic_features``/``actor_features_from_critic``
        (``BasePolicy``) instead, which apply the ``encoder_sharing``
        stop-gradient rule automatically and reuse one obs forward pass per
        role per training step (DrQ-v2's single augmented view)."""
        return self.actor_extractor.extract(obs, stop_gradient=stop_gradient)

    def critic_features_for(
        self,
        obs: Obs,
        actor_features: torch.Tensor,
        stop_gradient: bool = True,
    ) -> torch.Tensor:
        """Critic-role features for ``obs``, for callers that already hold
        actor-role features for the same ``obs`` (the actor loss's
        Q(s, pi(s)) term). Reuses ``actor_features`` -- zero extra compute --
        when the encoder is shared; re-extracts through the critic's own
        encoder only when it's genuinely separate. Available for direct use
        (tests, other callers); ``DDPG``'s own ``train()`` instead reuses
        its own already-computed critic-role features tensor directly
        (valid regardless of sharing mode, since it's the same obs), rather
        than routing through ``actor_features`` here."""
        if self.critic_extractor is None:
            return actor_features
        return self.extract_critic_features(obs, stop_gradient=stop_gradient)

    def prepare_batch_all(self, obs: Obs, next_obs: Optional[Obs] = None) -> None:
        """Call ``prepare_batch`` on every distinct extractor instance this
        policy owns (deduped by identity, so the shared default case still
        calls it exactly once)."""
        extractors = {id(self.actor_extractor): self.actor_extractor}
        if self.critic_extractor is not None:
            extractors[id(self.critic_extractor)] = self.critic_extractor
        for extractor in extractors.values():
            extractor.prepare_batch(obs, next_obs)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        obs: Obs,
        deterministic: bool = False,
        std: float = 0.0,
    ) -> torch.Tensor:
        features = self.extract_actor_features(obs)
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
    ) -> torch.Tensor:
        """Sample an action from the actor (no log-prob)."""
        features = self.extract_actor_features(obs)
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
        # Encoder trained via Q-loss. critic_extractor is None (falls back
        # to actor_extractor) in the (default) shared case, so this is
        # unchanged there.
        yield from self.critic.parameters()
        yield from (self.critic_extractor or self.actor_extractor).parameters()

    def actor_parameters(self):
        # Actor-only; the shared-encoder case trains the encoder via
        # critic_and_encoder_parameters' Q-loss instead (actor path is
        # stop-gradiented under "shared_critic_grad"). When critic_extractor
        # is genuinely separate, actor_extractor is actor-exclusive --
        # nothing else would ever train it -- so it belongs on this
        # optimizer instead.
        if self.critic_extractor is not None and self.critic_extractor is not self.actor_extractor:
            yield from self.actor_extractor.parameters()
        yield from self.actor.parameters()
