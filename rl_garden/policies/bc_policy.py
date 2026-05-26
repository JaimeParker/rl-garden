"""Behavioral Cloning policy with a shared feature extractor.

Actor-only: no critic, no value network. The encoder is trained end-to-end by
the actor loss, unlike RGBD SAC/IQL where critic updates own the encoder.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import BackboneType, KernelInit, SquashedGaussianActor
from rl_garden.policies.base import BasePolicy
from rl_garden.policies.sac_policy import LOG_STD_MAX, WSRL_LOG_STD_MIN


class BCPolicy(BasePolicy):
    """Actor-only policy for Behavioral Cloning."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] = (256, 256),
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        num_groups: int = 32,
        dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        log_std_mode: Literal["clamp", "tanh"] = "tanh",
        log_std_min: float = WSRL_LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "BCPolicy requires a Box action space."
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor

        fd = features_extractor.features_dim
        self.actor = SquashedGaussianActor(
            fd,
            action_space,
            hidden_dims=list(net_arch),
            use_layer_norm=use_layer_norm,
            use_group_norm=use_group_norm,
            num_groups=num_groups,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
            log_std_mode=log_std_mode,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def forward(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        return self.predict(obs, deterministic=deterministic)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)
        if deterministic:
            return self.actor.deterministic_action(features)
        action, _ = self.actor.action_log_prob(features)
        return action

    def behavior_log_prob(
        self, obs: Obs, actions: torch.Tensor, stop_gradient: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (log_prob, deterministic_action) for expert actions from a dataset.

        Matches IQLPolicy.behavior_log_prob signature so BC losses can be written
        in the same style as IQL actor losses.
        """
        features = self.extract_features(obs, stop_gradient=stop_gradient)
        log_prob = self.actor.evaluate_action_log_prob(features, actions)
        det_action = self.actor.deterministic_action(features)
        return log_prob, det_action

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        """All trainable parameters: encoder + actor trunk + heads."""
        return self.parameters()
