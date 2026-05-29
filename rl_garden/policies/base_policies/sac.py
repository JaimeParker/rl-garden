"""SAC checkpoint adapter for residual base-policy inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.checkpoint import load_checkpoint_file, validate_checkpoint_metadata
from rl_garden.common.types import Obs
from rl_garden.encoders import (
    CombinedExtractor,
    FlattenExtractor,
    ImageEncoderFactory,
    default_image_encoder_factory,
    resnet_encoder_factory,
)
from rl_garden.policies.base_policies.base import BasePolicyOutput, BasePolicyProvider
from rl_garden.policies.sac_policy import SACPolicy

SACBaseEncoder = Literal["plain_conv", "resnet10", "resnet18"]


def image_encoder_factory_from_name(
    encoder: SACBaseEncoder,
    *,
    features_dim: int = 256,
) -> ImageEncoderFactory:
    if encoder == "plain_conv":
        return default_image_encoder_factory(features_dim=features_dim)
    return resnet_encoder_factory(name=encoder, features_dim=features_dim)


class SACBasePolicy(BasePolicyProvider):
    """Inference-only SAC policy loaded directly from an rl-garden checkpoint."""

    def __init__(
        self,
        policy: SACPolicy,
        *,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        deterministic: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(observation_space, action_space, device=device)
        self.policy = policy.eval()
        self.deterministic = deterministic
        self.to(device)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        deterministic: bool = True,
        image_encoder_factory: Optional[ImageEncoderFactory] = None,
        image_keys: Optional[tuple[str, ...]] = None,
        state_key: Optional[str] = None,
        use_proprio: Optional[bool] = None,
        proprio_latent_dim: Optional[int] = None,
        image_fusion_mode: Optional[str] = None,
        enable_stacking: Optional[bool] = None,
        strict: bool = True,
    ) -> "SACBasePolicy":
        device = torch.device(device)
        checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
        validate_checkpoint_metadata(
            checkpoint,
            algorithm_class="SACBasePolicy",
            compatible_algorithms=("SAC",),
            observation_space=observation_space,
            action_space=action_space,
            strict=strict,
        )
        hparams = checkpoint.get("metadata", {}).get("hyperparameters", {})
        policy = cls._build_policy(
            observation_space=observation_space,
            action_space=action_space,
            hparams=hparams,
            image_encoder_factory=image_encoder_factory,
            image_keys=image_keys,
            state_key=state_key,
            use_proprio=use_proprio,
            proprio_latent_dim=proprio_latent_dim,
            image_fusion_mode=image_fusion_mode,
            enable_stacking=enable_stacking,
        ).to(device)
        policy.load_state_dict(checkpoint["state"]["policy"], strict=strict)
        return cls(
            policy,
            observation_space=observation_space,
            action_space=action_space,
            deterministic=deterministic,
            device=device,
        )

    @staticmethod
    def _build_policy(
        *,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hparams: dict[str, Any],
        image_encoder_factory: Optional[ImageEncoderFactory],
        image_keys: Optional[tuple[str, ...]],
        state_key: Optional[str],
        use_proprio: Optional[bool],
        proprio_latent_dim: Optional[int],
        image_fusion_mode: Optional[str],
        enable_stacking: Optional[bool],
    ) -> SACPolicy:
        if isinstance(observation_space, spaces.Box):
            features_extractor = FlattenExtractor(observation_space)
        elif isinstance(observation_space, spaces.Dict):
            factory = image_encoder_factory or default_image_encoder_factory()
            features_extractor = CombinedExtractor(
                observation_space,
                image_keys=image_keys or tuple(hparams.get("image_keys", ("rgb", "depth"))),
                state_key=state_key or hparams.get("state_key", "state"),
                image_encoder_factory=factory,
                proprio_latent_dim=(
                    proprio_latent_dim
                    if proprio_latent_dim is not None
                    else int(hparams.get("proprio_latent_dim", 64))
                ),
                use_proprio=(
                    use_proprio
                    if use_proprio is not None
                    else bool(hparams.get("use_proprio", True))
                ),
                fusion_mode=image_fusion_mode or hparams.get("image_fusion_mode", "stack_channels"),
                enable_stacking=(
                    enable_stacking
                    if enable_stacking is not None
                    else bool(hparams.get("enable_stacking", False))
                ),
            )
        else:
            raise TypeError(f"SACBasePolicy supports Box or Dict observations, got {observation_space}.")

        net_arch: Sequence[int] | dict[str, Sequence[int]] = hparams.get(
            "net_arch", (256, 256, 256)
        )
        return SACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch=net_arch,
            n_critics=int(hparams.get("n_critics", 2)),
            critic_subsample_size=hparams.get("critic_subsample_size"),
        )

    @torch.no_grad()
    def select_action(self, obs: Obs) -> BasePolicyOutput:
        actions = self.policy.predict(
            self._obs_to_device(obs),
            deterministic=self.deterministic,
        )
        return BasePolicyOutput(actions=self._format_actions(actions))
