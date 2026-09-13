"""Policy abstraction: actor/critic feature-extractor contract + predict.

Every policy with a critic/value head owns one or two feature extractors --
``actor_extractor`` (always) and ``critic_extractor`` (only when
``encoder_sharing="separate"``; ``None`` means the critic reads through the
actor's own extractor). This is the single place stop-gradient semantics for
actor-role features are decided; every consumer picks a role
(``extract_actor_features`` for policy nets / BC / flow / diffusion nets /
distill heads / teacher-student actors, ``extract_critic_features`` for Q, V,
TD-target nets, discriminators, reward heads) instead of touching an
extractor directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor

# Mirrors rl_garden.algorithms._observation.EncoderSharing (the algorithm
# layer's own copy of this Literal) without importing the algorithms layer
# from policies -- policies sit below algorithms; see
# .agents/rules/repository-map.md.
EncoderSharing = Literal["shared_critic_grad", "shared", "separate"]


class BasePolicy(nn.Module, ABC):
    """Owns the actor/critic feature extractor(s) and ``predict``.

    ``critic_extractor=None`` means the critic reads through
    ``actor_extractor`` (both the ``"shared_critic_grad"`` and ``"shared"``
    encoder-sharing modes); a concrete ``critic_extractor`` means
    ``"separate"`` sharing -- two independently trained extractor instances.
    """

    def __init__(
        self,
        observation_space: "spaces.Box | spaces.Dict",
        action_space: spaces.Space,
        *,
        actor_extractor: BaseFeaturesExtractor,
        critic_extractor: Optional[BaseFeaturesExtractor] = None,
        encoder_sharing: EncoderSharing = "shared_critic_grad",
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.actor_extractor = actor_extractor
        self.critic_extractor = critic_extractor
        self.encoder_sharing = encoder_sharing

    @property
    def actor_features_dim(self) -> int:
        return self.actor_extractor.features_dim

    @property
    def critic_features_dim(self) -> int:
        return (self.critic_extractor or self.actor_extractor).features_dim

    @abstractmethod
    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor: ...

    @property
    def actor_features_detached(self) -> bool:
        """Single source of truth for the actor-role stop-gradient rule: True
        iff ``encoder_sharing == "shared_critic_grad"`` and the actor/critic
        extractors are the same object (including the default
        ``critic_extractor is None`` case -- there is then only one
        extractor, trained only by the critic loss under this sharing mode).

        Subclasses whose actor/critic heads sit on the far side of a
        post-encoder stage (an RNN, GTrXL, or diffusion-chain conditioning
        tensor -- see ``SequenceSAC``/``SequencePPO``/``DPPO``) read this
        property to decide whether to detach that later tensor themselves,
        since gating must happen after the stage, not at the raw encoder
        output that ``extract_actor_features`` detaches below.
        """
        return self.encoder_sharing == "shared_critic_grad" and (
            self.critic_extractor is None or self.critic_extractor is self.actor_extractor
        )

    def extract_actor_features(self, obs: Obs) -> torch.Tensor:
        """Actor-role features for ``obs``; stop-gradient per
        ``actor_features_detached``."""
        return self.actor_extractor.extract(obs, stop_gradient=self.actor_features_detached)

    def actor_features_from_critic(
        self, critic_features: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Reuse an already-computed critic-role features tensor for the
        actor role instead of a second, redundant extractor forward pass --
        restores DrQ-v2's single augmented view for the common case where
        one obs batch feeds both the critic loss and, right after, the
        actor loss.

        Returns ``critic_features.detach()`` when ``actor_features_detached``
        is True: detaching cuts the actor path out of the extractor's graph
        entirely, so reuse is safe regardless of whether the critic's own
        ``optimizer.step()`` (an in-place weight update) has already run by
        the time the actor loss is computed. Returns ``None`` otherwise
        (``"separate"`` -- a genuinely different extractor for each role, no
        tensor to share; or ``"shared"`` -- both roles need a live gradient
        through the encoder, and this codebase's actor/critic updates are
        two sequential ``optimizer.step()`` calls, not one combined
        backward, so reusing a live, non-detached tensor across both would
        risk the in-place-update-corrupts-a-pending-backward hazard a fresh
        call avoids). Callers fall back to ``extract_actor_features(obs)``
        when this returns ``None``.
        """
        if self.actor_features_detached:
            return critic_features.detach()
        return None

    def extract_critic_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        """Critic-role features for ``obs`` -- the critic's own extractor
        under ``"separate"`` sharing, else the shared actor extractor. Never
        detaches by default (the encoder-sharing stop-gradient rule only
        ever applies to the actor role); ``stop_gradient`` is an explicit,
        sharing-independent override for callers with their own reason to
        detach (e.g. eval-only Q reads, a frozen-encoder training phase)."""
        extractor = self.critic_extractor if self.critic_extractor is not None else self.actor_extractor
        return extractor.extract(obs, stop_gradient=stop_gradient)

    def update_normalizer(self, obs: Obs) -> None:
        """Update running obs-normalization statistics on every distinct
        extractor this policy owns (deduped by identity, so the common
        shared-extractor case updates it exactly once)."""
        extractors = {id(self.actor_extractor): self.actor_extractor}
        if self.critic_extractor is not None:
            extractors[id(self.critic_extractor)] = self.critic_extractor
        for extractor in extractors.values():
            extractor.update_normalizer(obs)
