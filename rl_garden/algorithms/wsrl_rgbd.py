"""RGBD WSRL: subclass of ``WSRL`` that adds visual observations.

This subclass keeps the base ``WSRL`` training loop (CQL + Cal-QL + REDQ) and only contributes:
  1. RGBD-specific default extractor settings (``CombinedExtractor``)
  2. Dict replay buffer construction (MCDictReplayBuffer for Cal-QL)
  3. Image stop-gradient on the actor path

The image stop-gradient optimization is critical for vision-based RL: for the
actor update, image encodings use ``stop_gradient=True`` so gradients don't
flow from policy loss into the (expensive) CNN/ResNet. The encoder is trained
only through critic loss.
"""
from __future__ import annotations

from typing import Any, Optional

from gymnasium import spaces

from rl_garden.algorithms.wsrl import WSRL
from rl_garden.buffers.mc_buffer import MCDictReplayBuffer
from rl_garden.encoders.combined import (
    CombinedExtractor,
    ImageEncoderFactory,
    default_image_encoder_factory,
)


class WSRLRGBD(WSRL):
    """WSRL with vision support (RGB/RGBD observations).

    Extends WSRL to handle Dict observation spaces with image and state keys.
    Uses CombinedExtractor to process visual observations and proprioceptive state.
    Detaches encoder gradients on actor path for efficient training.

    Args:
        image_encoder_factory: Factory for creating image encoders (ResNet, PlainConv, etc.)
        image_keys: Tuple of image observation keys (e.g., ("rgb", "depth"))
        state_key: Key for proprioceptive state observations
        use_proprio: Whether to use proprioceptive state
        proprio_latent_dim: Latent dimension for proprioceptive MLP
        detach_encoder_on_actor: Backward-compatible guard; RGBD actor path always uses stop_gradient
        batch_size: Batch size for training (default 512 for vision)
        utd: Update-to-data ratio (default 0.25 for vision)
        **kwargs: Additional arguments passed to WSRL
    """

    def __init__(
        self,
        *args,
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: tuple[str, ...] = ("rgb", "depth"),
        state_key: str = "state",
        use_proprio: bool = True,
        proprio_latent_dim: int = 64,
        image_fusion_mode: str = "stack_channels",
        enable_stacking: bool = False,
        detach_encoder_on_actor: bool = True,
        batch_size: int = 512,
        utd: float = 0.25,
        policy_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if not detach_encoder_on_actor:
            raise ValueError(
                "WSRLRGBD always uses stop_gradient=True on the actor image path "
                "so the image encoder is trained only by critic loss."
            )
        self._image_encoder_factory = image_encoder_factory or default_image_encoder_factory()
        self._image_keys = image_keys
        self._state_key = state_key
        self._use_proprio = use_proprio
        self._proprio_latent_dim = proprio_latent_dim
        self._image_fusion_mode = image_fusion_mode
        self._enable_stacking = enable_stacking
        # RGBD defaults to smaller batch / utd for vision-based training.
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("utd", utd)
        super().__init__(*args, policy_kwargs=policy_kwargs, **kwargs)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "image_keys": self._image_keys,
            "state_key": self._state_key,
            "use_proprio": self._use_proprio,
            "proprio_latent_dim": self._proprio_latent_dim,
            "image_fusion_mode": self._image_fusion_mode,
            "enable_stacking": self._enable_stacking,
        }

    def _default_features_extractor_class(self):
        obs_space = self.env.single_observation_space
        assert isinstance(obs_space, spaces.Dict), (
            "WSRLRGBD expects a Dict observation space; got " + str(type(obs_space))
        )
        return CombinedExtractor

    def _default_features_extractor_kwargs(self) -> dict[str, Any]:
        return {
            "image_keys": self._image_keys,
            "state_key": self._state_key,
            "image_encoder_factory": self._image_encoder_factory,
            "proprio_latent_dim": self._proprio_latent_dim,
            "use_proprio": self._use_proprio,
            "fusion_mode": self._image_fusion_mode,
            "enable_stacking": self._enable_stacking,
        }

    def _build_replay_buffer(self):
        """Build MCDictReplayBuffer for Cal-QL with dict observations."""
        return MCDictReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
            gamma=self.gamma,
            sparse_reward_mc=self.sparse_reward_mc,
            sparse_negative_reward=self.sparse_negative_reward,
            success_threshold=self.success_threshold,
        )

    def _actor_loss(self, obs):
        """Compute actor loss with stop-gradient on image encodings.

        The image encoder is trained only through critic loss, not policy loss.
        This prevents expensive backprop through the CNN/ResNet on every actor update.
        """
        import torch

        alpha = self._current_alpha().detach()
        action, log_prob, features = self.policy.actor_action_log_prob(
            obs, stop_gradient=True
        )

        # Actor loss uses all critics; REDQ subsampling is for target/CQL paths.
        min_q = self.policy.min_q_value(
            features, action, subsample_size=None, target=False
        )

        return (alpha * log_prob - min_q).mean(), log_prob.detach()
