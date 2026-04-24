"""RGBD SAC: subclass of ``SAC`` that adds visual observations.

Only two overrides differ from the base class:
  1. ``_build_features_extractor`` — use ``CombinedExtractor`` (image + state)
  2. ``_build_replay_buffer``       — use ``DictReplayBuffer``

The ``SAC.train`` loop is reused unchanged. One RGBD-specific optimization
from sac_rgbd.py L696-L723 matters: for the actor update, detach the encoder
so gradients don't flow from policy loss into the (expensive) CNN/ResNet.
We bake that into ``_actor_loss`` via ``detach_encoder=True``.
"""
from __future__ import annotations

from typing import Callable, Optional

from gymnasium import spaces

from rl_garden.algorithms.sac import SAC
from rl_garden.buffers.dict_buffer import DictReplayBuffer
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.combined import (
    CombinedExtractor,
    ImageEncoderFactory,
    default_image_encoder_factory,
)


class RGBDSAC(SAC):
    def __init__(
        self,
        *args,
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: tuple[str, ...] = ("rgb", "depth"),
        state_key: str = "state",
        use_proprio: bool = True,
        proprio_latent_dim: int = 64,
        detach_encoder_on_actor: bool = True,
        batch_size: int = 512,
        utd: float = 0.25,
        **kwargs,
    ) -> None:
        self._image_encoder_factory = image_encoder_factory or default_image_encoder_factory()
        self._image_keys = image_keys
        self._state_key = state_key
        self._use_proprio = use_proprio
        self._proprio_latent_dim = proprio_latent_dim
        self.detach_encoder_on_actor = detach_encoder_on_actor
        # RGBD defaults to smaller batch / utd (matches sac_rgbd.py).
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("utd", utd)
        super().__init__(*args, **kwargs)

    def _build_features_extractor(self) -> BaseFeaturesExtractor:
        obs_space = self.env.single_observation_space
        assert isinstance(obs_space, spaces.Dict), (
            "RGBDSAC expects a Dict observation space; got " + str(type(obs_space))
        )
        return CombinedExtractor(
            observation_space=obs_space,
            image_keys=self._image_keys,
            state_key=self._state_key,
            image_encoder_factory=self._image_encoder_factory,
            proprio_latent_dim=self._proprio_latent_dim,
            use_proprio=self._use_proprio,
        )

    def _build_replay_buffer(self):
        return DictReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _actor_loss(self, obs):
        # Override to detach encoder on the actor path.
        import torch

        alpha = self._current_alpha().detach()
        action, log_prob, features = self.policy.actor_action_log_prob(
            obs, detach_encoder=self.detach_encoder_on_actor
        )
        q_values = self.policy.q_values(features, action, target=False)
        min_q = torch.min(torch.stack(q_values, dim=0), dim=0).values
        return (alpha * log_prob - min_q).mean(), log_prob.detach()
