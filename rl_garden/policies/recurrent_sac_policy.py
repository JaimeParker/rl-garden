"""Recurrent variant of SACPolicy: an RNN latent stage between the encoder and
the actor/critic heads."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.obs_utils import flatten_leading_dims
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import BackboneType, CriticImpl, KernelInit, RecurrentLatentEncoder, RecurrentState
from rl_garden.policies.sac_policy import SACPolicy, LOG_STD_MAX, LOG_STD_MIN


class RecurrentSACPolicy(SACPolicy):
    """SACPolicy with a RecurrentLatentEncoder between features_extractor and
    the actor/critic heads. Always assumes a flat-latent features_extractor --
    the caller (RecurrentSAC._build_policy) is responsible for rejecting
    ``structured_feature_config() is not None`` before constructing this."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        recurrent_encoder: RecurrentLatentEncoder,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        *,
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        critic_impl: CriticImpl = "vmap",
        std_parameterization: str = "exp",
        log_std_mode: str = "tanh",
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        # recurrent_encoder.features_dim is read here (a plain property, safe
        # before nn.Module registration); self.recurrent_encoder is assigned
        # AFTER super().__init__() returns, since nn.Module.__setattr__ requires
        # nn.Module.__init__() (called inside super().__init__()) to have
        # already run first -- same constraint RecurrentPPOPolicy documents.
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            net_arch,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            critic_impl=critic_impl,
            std_parameterization=std_parameterization,
            log_std_mode=log_std_mode,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            features_dim=recurrent_encoder.features_dim,
        )
        self.recurrent_encoder = recurrent_encoder

    def act_recurrent_step(
        self,
        obs: Obs,
        hidden: RecurrentState,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Single rollout step: obs -> features -> RNN step -> action. Returns
        (action, new_hidden)."""
        raw = self._extract_features(obs, stop_gradient=False)
        latent, new_hidden = self.recurrent_encoder.step(raw, hidden, episode_starts)
        if deterministic:
            action = self.actor.deterministic_action(latent)
        else:
            action, _ = self.actor.action_log_prob(latent)
        return action, new_hidden

    def windowed_features(
        self,
        obs_window: Obs,
        initial_hidden: RecurrentState,
        episode_starts_window: torch.Tensor,
        burn_in_len: int,
        *,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        """Training-time windowed feature extraction: obs_window ->
        features_extractor (flattened over time) -> RNN burn-in+tail unroll.
        Returns the TAIL post-RNN features, shape (tail_len, B, hidden_size).
        """
        num_envs = episode_starts_window.shape[1]
        flat_obs = flatten_leading_dims(obs_window)
        raw = self._extract_features(flat_obs, stop_gradient=stop_gradient)
        raw = raw.reshape(-1, num_envs, raw.shape[-1])
        tail, _ = self.recurrent_encoder.forward_sequence_with_burn_in(
            raw, initial_hidden, episode_starts_window, burn_in_len
        )
        return tail
