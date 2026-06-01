"""Token-aware SAC policies for residual-style ViT features."""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.vit import ViTCombinedExtractor
from rl_garden.networks import (
    BackboneType,
    KernelInit,
    SquashedGaussianActor,
    create_mlp,
    get_actor_critic_arch,
)
from rl_garden.policies.base import BasePolicy
from rl_garden.policies.sac_policy import LOG_STD_MAX, LOG_STD_MIN


class TokenActorFeatures(nn.Module):
    """Residual default actor trunk: flatten ViT tokens, compress, append prop."""

    def __init__(
        self,
        extractor: ViTCombinedExtractor,
        *,
        output_dim: int = 128,
        base_action_dim: int = 0,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "extractor", extractor)
        self.base_action_dim = base_action_dim
        self.compress = nn.Sequential(
            nn.Linear(extractor.token_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
        self.features_dim = output_dim + extractor.prop_dim + base_action_dim

    def forward(
        self, features: torch.Tensor, base_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tokens, prop = self.extractor.split_features(features)
        out = [self.compress(tokens.flatten(1)), prop]
        if self.base_action_dim:
            if base_actions is None:
                raise ValueError("TokenActorFeatures requires base_actions.")
            out.append(base_actions)
        return torch.cat(out, dim=-1)


class SpatialEmbQHead(nn.Module):
    """One residual-style spatial Q head."""

    def __init__(
        self,
        extractor: ViTCombinedExtractor,
        action_dim: int,
        hidden_dims: Sequence[int],
        *,
        spatial_emb_dim: int = 1024,
        use_layer_norm: bool = True,
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "extractor", extractor)
        self.action_dim = action_dim
        self.spatial_emb_dim = spatial_emb_dim
        token_patch_input_dim = extractor.num_patches + extractor.prop_dim + action_dim
        if spatial_emb_dim > 0:
            self.spatial_proj = nn.Sequential(
                nn.Linear(token_patch_input_dim, spatial_emb_dim),
                nn.LayerNorm(spatial_emb_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
            )
            self.spatial_weight = nn.Parameter(
                torch.empty(1, extractor.patch_dim, spatial_emb_dim)
            )
            nn.init.xavier_uniform_(self.spatial_weight)
            head_input_dim = spatial_emb_dim + extractor.prop_dim + action_dim
        else:
            self.spatial_proj = None
            self.spatial_weight = None
            head_input_dim = extractor.token_dim + extractor.prop_dim + action_dim
        self.q = create_mlp(
            head_input_dim,
            1,
            hidden_dims,
            use_layer_norm=use_layer_norm,
            kernel_init=kernel_init,
        )

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        tokens, prop = self.extractor.split_features(features)
        if self.spatial_proj is None:
            z = torch.cat([tokens.flatten(1), prop, actions], dim=-1)
            return self.q(z)

        b, _p, d = tokens.shape
        token_channels = tokens.transpose(1, 2)
        action_rep = actions.unsqueeze(1).expand(b, d, self.action_dim)
        if self.extractor.prop_dim:
            prop_rep = prop.unsqueeze(1).expand(b, d, self.extractor.prop_dim)
            spatial_input = torch.cat([token_channels, prop_rep, action_rep], dim=-1)
        else:
            spatial_input = torch.cat([token_channels, action_rep], dim=-1)
        y = self.spatial_proj(spatial_input)
        assert self.spatial_weight is not None
        z = (self.spatial_weight * y).sum(dim=1)
        return self.q(torch.cat([z, prop, actions], dim=-1))


class SpatialEmbQEnsemble(nn.Module):
    def __init__(
        self,
        extractor: ViTCombinedExtractor,
        action_space: spaces.Box,
        hidden_dims: Sequence[int],
        *,
        n_critics: int = 2,
        spatial_emb_dim: int = 1024,
        use_layer_norm: bool = True,
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        action_dim = int(np.prod(action_space.shape))
        self.n_critics = n_critics
        self.q_nets = nn.ModuleList(
            [
                SpatialEmbQHead(
                    extractor,
                    action_dim,
                    hidden_dims,
                    spatial_emb_dim=spatial_emb_dim,
                    use_layer_norm=use_layer_norm,
                    kernel_init=kernel_init,
                )
                for _ in range(n_critics)
            ]
        )

    def forward(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        return tuple(q(features, actions) for q in self.q_nets)

    def forward_all(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([q(features, actions) for q in self.q_nets], dim=0)


class ViTSACPolicy(BasePolicy):
    """SAC policy using token-aware residual-style ViT critic."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        use_cql_alpha_lagrange: bool = False,
        cql_alpha_lagrange_init: float = 1.0,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        log_std_mode: Literal["clamp", "tanh"] = "tanh",
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        vit_actor_feature_dim: Optional[int] = None,
        vit_critic_spatial_emb_dim: Optional[int] = None,
        base_action_dim: int = 0,
    ) -> None:
        del (
            use_cql_alpha_lagrange,
            cql_alpha_lagrange_init,
            critic_use_group_norm,
            critic_dropout_rate,
        )
        super().__init__()
        if not isinstance(features_extractor, ViTCombinedExtractor):
            raise TypeError("ViTSACPolicy requires ViTCombinedExtractor.")
        if critic_subsample_size is not None and critic_subsample_size > n_critics:
            raise ValueError("critic_subsample_size must be <= n_critics.")

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size
        self.use_cql_alpha_lagrange = False
        self.cql_alpha_lagrange = None
        self.base_action_dim = base_action_dim
        actor_feature_dim = (
            vit_actor_feature_dim
            if vit_actor_feature_dim is not None
            else features_extractor.actor_feature_dim
        )
        critic_spatial_emb_dim = (
            vit_critic_spatial_emb_dim
            if vit_critic_spatial_emb_dim is not None
            else features_extractor.critic_spatial_emb_dim
        )

        if actor_hidden_dims is not None or critic_hidden_dims is not None:
            actor_arch = list(actor_hidden_dims if actor_hidden_dims is not None else ())
            critic_arch = list(
                critic_hidden_dims if critic_hidden_dims is not None else actor_arch
            )
        else:
            actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.actor_features = TokenActorFeatures(
            features_extractor,
            output_dim=actor_feature_dim,
            base_action_dim=base_action_dim,
        )
        self.actor = SquashedGaussianActor(
            self.actor_features.features_dim,
            action_space,
            hidden_dims=actor_arch,
            use_layer_norm=actor_use_layer_norm,
            use_group_norm=actor_use_group_norm,
            num_groups=num_groups,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
            log_std_mode=log_std_mode,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.critic = SpatialEmbQEnsemble(
            features_extractor,
            action_space,
            hidden_dims=critic_arch,
            n_critics=n_critics,
            spatial_emb_dim=critic_spatial_emb_dim,
            use_layer_norm=critic_use_layer_norm,
            kernel_init=kernel_init,
        )
        self.critic_target = SpatialEmbQEnsemble(
            features_extractor,
            action_space,
            hidden_dims=critic_arch,
            n_critics=n_critics,
            spatial_emb_dim=critic_spatial_emb_dim,
            use_layer_norm=critic_use_layer_norm,
            kernel_init=kernel_init,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    def prepare_train_batch(self, data) -> None:
        if isinstance(data.obs, dict):
            self.features_extractor.cache_features(data.obs, augment=True)
        if isinstance(data.next_obs, dict):
            with torch.no_grad():
                self.features_extractor.cache_features(data.next_obs, augment=True)

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def forward(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        return self.predict(obs, deterministic=deterministic)

    def _actor_input(
        self, features: torch.Tensor, base_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.actor_features(features, base_actions)

    def predict(
        self,
        obs: Obs,
        deterministic: bool = False,
        *,
        base_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.extract_features(obs)
        actor_features = self._actor_input(features, base_actions)
        if deterministic:
            return self.actor.deterministic_action(actor_features)
        action, _ = self.actor.action_log_prob(actor_features)
        return action

    def actor_action_log_prob(
        self,
        obs: Obs,
        stop_gradient: bool = False,
        detach_encoder: Optional[bool] = None,
        *,
        base_actions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if detach_encoder is not None:
            stop_gradient = detach_encoder
        features = self.extract_features(obs, stop_gradient=stop_gradient)
        actor_features = self._actor_input(features, base_actions)
        action, log_prob = self.actor.action_log_prob(actor_features)
        return action, log_prob, features

    def q_values(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> tuple[torch.Tensor, ...]:
        net = self.critic_target if target else self.critic
        return net(features, actions)

    def q_values_all(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        net = self.critic_target if target else self.critic
        return net.forward_all(features, actions)

    def q_values_subsampled(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        subsample_size: Optional[int] = None,
        target: bool = True,
    ) -> torch.Tensor:
        q_all = self.q_values_all(features, actions, target=target)
        if subsample_size is not None and subsample_size < self.n_critics:
            indices = torch.randint(
                0, self.n_critics, (subsample_size,), device=q_all.device
            )
            return q_all[indices]
        return q_all

    def min_q_value(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        subsample_size: Optional[int] = None,
        target: bool = True,
    ) -> torch.Tensor:
        return self.q_values_subsampled(
            features, actions, subsample_size=subsample_size, target=target
        ).min(dim=0).values

    def get_cql_alpha(self) -> torch.Tensor:
        raise ValueError("CQL alpha Lagrange multiplier is owned by algorithms.")

    def critic_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from self.features_extractor.parameters()

    def actor_parameters(self):
        yield from self.actor_features.parameters()
        yield from self.actor.parameters()

    def cql_alpha_lagrange_parameters(self):
        if False:
            yield


class ViTResidualSACPolicy(ViTSACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        action_dim = int(np.prod(action_space.shape))
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch=net_arch,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            base_action_dim=action_dim,
            log_std_mode="tanh",
            log_std_min=LOG_STD_MIN,
            log_std_max=LOG_STD_MAX,
            **kwargs,
        )

    def predict(
        self,
        obs,
        deterministic: bool = False,
        *,
        base_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_actions is None:
            raise ValueError("ViTResidualSACPolicy.predict requires base_actions.")
        return super().predict(obs, deterministic=deterministic, base_actions=base_actions)

    def actor_action_log_prob(
        self,
        obs,
        base_actions: torch.Tensor,
        stop_gradient: bool = False,
        detach_encoder: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().actor_action_log_prob(
            obs,
            stop_gradient=stop_gradient,
            detach_encoder=detach_encoder,
            base_actions=base_actions,
        )
