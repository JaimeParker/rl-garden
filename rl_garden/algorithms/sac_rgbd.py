"""RGBD SAC: subclass of ``SAC`` that adds visual observations.

This subclass keeps the base ``SAC`` training loop and only contributes:
  1. RGBD-specific default extractor settings (``CombinedExtractor``)
  2. Dict replay buffer construction
  3. Encoder detachment on the actor path

One RGBD-specific optimization from sac_rgbd.py L696-L723 matters: for the
actor update, detach the encoder so gradients don't flow from policy loss
into the (expensive) CNN/ResNet.
"""
from __future__ import annotations

from typing import Any, Optional

from gymnasium import spaces

from rl_garden.algorithms.sac import SAC
from rl_garden.buffers.dict_buffer import DictReplayBuffer
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
        policy_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if not detach_encoder_on_actor:
            raise ValueError(
                "RGBDSAC requires detach_encoder_on_actor=True so the image encoder "
                "is trained only by critic loss."
            )
        self._image_encoder_factory = image_encoder_factory or default_image_encoder_factory()
        self._image_keys = image_keys
        self._state_key = state_key
        self._use_proprio = use_proprio
        self._proprio_latent_dim = proprio_latent_dim
        # RGBD defaults to smaller batch / utd (matches sac_rgbd.py).
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("utd", utd)
        super().__init__(*args, policy_kwargs=policy_kwargs, **kwargs)

    def _default_features_extractor_class(self):
        obs_space = self.env.single_observation_space
        assert isinstance(obs_space, spaces.Dict), (
            "RGBDSAC expects a Dict observation space; got " + str(type(obs_space))
        )
        return CombinedExtractor

    def _default_features_extractor_kwargs(self) -> dict[str, Any]:
        return {
            "image_keys": self._image_keys,
            "state_key": self._state_key,
            "image_encoder_factory": self._image_encoder_factory,
            "proprio_latent_dim": self._proprio_latent_dim,
            "use_proprio": self._use_proprio,
        }

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
        # The RGB image encoder is trained only through critic loss.
        import torch

        alpha = self._current_alpha().detach()
        action, log_prob, features = self.policy.actor_action_log_prob(obs, detach_encoder=True)
        q_values = self.policy.q_values(features, action, target=False)
        min_q = torch.min(torch.stack(q_values, dim=0), dim=0).values
        return (alpha * log_prob - min_q).mean(), log_prob.detach()
