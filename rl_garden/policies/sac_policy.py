"""SAC actor/critic policy with a shared features extractor.

Mirrors the architecture used by ManiSkill's sac.py (state) and sac_rgbd.py
(RGBD), reorganized SB3-style: the ``SACPolicy`` owns:
  - ``features_extractor`` (shared encoder; can be ``FlattenExtractor`` for
    state obs, ``CombinedExtractor`` for RGBD)
  - ``actor`` (MLP -> mean/log_std heads, tanh-squashed Normal)
  - ``critic`` (ensemble of Q-nets that *reuse* the same extractor)
  - ``critic_target`` (same, no grad)

Key RGBD detail, preserved from hil-serl/ManiSkill visual SAC:
  - Critic optimizer owns the encoder params (encoder learns via Q-loss).
  - Actor update extracts visual features with ``stop_gradient=True`` so the
    image encoder sees no gradients from the policy loss.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    BackboneType,
    CriticImpl,
    EnsembleQCritic,
    KernelInit,
    SpatialEmbQEnsemble,
    SquashedGaussianActor,
    get_actor_critic_arch,
)
from rl_garden.policies.base import BasePolicy

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0
WSRL_LOG_STD_MIN = -20.0


Actor = SquashedGaussianActor
ContinuousCritic = EnsembleQCritic


class _TokenCompressor(nn.Module):
    """Compress flat token+prop features for the actor input.

    Splits the flat ``[tokens_flat, prop]`` vector into its two components,
    applies a Linear+LayerNorm+ReLU compression to the token block, then
    re-concatenates prop.

    Output dimension: ``output_dim + prop_dim``.
    """

    def __init__(self, token_dim: int, prop_dim: int, output_dim: int) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.prop_dim = prop_dim
        self.output_dim = output_dim
        self.compress = nn.Sequential(
            nn.Linear(token_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    @property
    def features_dim(self) -> int:
        return self.output_dim + self.prop_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens_flat = features[:, : self.token_dim]
        prop = features[:, self.token_dim :]
        return torch.cat([self.compress(tokens_flat), prop], dim=-1)


def _softplus_inverse(x: float) -> float:
    import numpy as np

    return float(np.log(np.expm1(x)))


def _dormant_ratio(activations: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Fraction of units with mean |activation| <= ``tau`` times the layer mean.

    Relative (Sokar et al. 2023) rather than absolute, so the result stays
    comparable across encoder configs with different activation scales (e.g.
    flatten vs. GAP pooling).
    """
    mean_abs = activations.abs().mean(dim=0)
    return (mean_abs <= tau * mean_abs.mean()).float().mean()


class CQLAlphaLagrange(nn.Module):
    """Lagrange multiplier for auto-tuning CQL alpha."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        if init_value <= 0:
            raise ValueError("CQL alpha Lagrange init value must be positive.")
        self.log_alpha = nn.Parameter(
            torch.tensor(_softplus_inverse(init_value), dtype=torch.float32)
        )

    def forward(self) -> torch.Tensor:
        return F.softplus(self.log_alpha)


class TemperatureLagrange(nn.Module):
    """SAC entropy coefficient via softplus."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        if init_value <= 0:
            raise ValueError("Temperature init value must be positive.")
        self.log_alpha = nn.Parameter(
            torch.tensor(_softplus_inverse(init_value), dtype=torch.float32)
        )

    def forward(self) -> torch.Tensor:
        return F.softplus(self.log_alpha)


class SACPolicy(BasePolicy):
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
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        critic_impl: CriticImpl = "vmap",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        log_std_mode: Literal["clamp", "tanh"] = "tanh",
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        log_std_multiplier_init: Optional[float] = None,
        log_std_offset_init: Optional[float] = None,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
        actor_feature_dim: Optional[int] = None,
        critic_spatial_emb_dim: int = 1024,
        features_dim: Optional[int] = None,
        critic_features_extractor: Optional[BaseFeaturesExtractor] = None,
        critic_backbone_type: Optional[BackboneType] = None,
    ) -> None:
        del use_cql_alpha_lagrange, cql_alpha_lagrange_init
        super().__init__()
        assert isinstance(action_space, spaces.Box), "SAC requires a Box action space."
        assert n_critics >= 2, f"n_critics must be >= 2, got {n_critics}"
        if critic_subsample_size is not None:
            assert critic_subsample_size <= n_critics, (
                f"critic_subsample_size ({critic_subsample_size}) must be <= "
                f"n_critics ({n_critics})"
            )
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        # Unset -> literally the same object as features_extractor (not just
        # equal config): this identity is the single source of truth for
        # "shared encoder" used by critic_features_for/prepare_batch_all/
        # critic_and_encoder_parameters below.
        self.critic_features_extractor = critic_features_extractor or features_extractor
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size

        if actor_hidden_dims is not None or critic_hidden_dims is not None:
            # Backward-compatible path for direct policy construction.
            actor_arch = list(actor_hidden_dims if actor_hidden_dims is not None else ())
            critic_arch = list(critic_hidden_dims if critic_hidden_dims is not None else actor_arch)
        else:
            actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        fd = features_dim if features_dim is not None else features_extractor.features_dim
        sc = features_extractor.structured_feature_config()

        # Critic's own fd/sc/backbone, computed independently so a critic
        # with a genuinely different (possibly structured) extractor is
        # dispatched correctly below. Identical to fd/sc/backbone_type when
        # critic_features_extractor wasn't overridden.
        if self.critic_features_extractor is features_extractor:
            critic_fd = fd
            critic_sc = sc
        else:
            critic_fd = self.critic_features_extractor.features_dim
            critic_sc = self.critic_features_extractor.structured_feature_config()
        critic_backbone_type = critic_backbone_type or backbone_type

        # --- Actor adapter (token compression) ---
        if actor_feature_dim is not None:
            if sc is None or sc.get("layout") != "token_and_prop":
                raise ValueError(
                    "actor_feature_dim requires the features extractor to declare a "
                    "'token_and_prop' structured_feature_config. "
                    f"Got: {sc!r}"
                )
            self._actor_adapter: Optional[_TokenCompressor] = _TokenCompressor(
                token_dim=sc["num_patches"] * sc["patch_dim"],
                prop_dim=sc["prop_dim"],
                output_dim=actor_feature_dim,
            )
            self._actor_fd: int = self._actor_adapter.features_dim
        else:
            self._actor_adapter = None
            self._actor_fd = fd

        # --- Critic --- (dispatched on critic_sc/critic_fd -- see above)
        if critic_sc is not None and critic_sc.get("layout") == "token_and_prop":
            self.critic = SpatialEmbQEnsemble(
                num_patches=critic_sc["num_patches"],
                patch_dim=critic_sc["patch_dim"],
                prop_dim=critic_sc["prop_dim"],
                action_space=action_space,
                hidden_dims=critic_arch,
                n_critics=n_critics,
                spatial_emb_dim=critic_spatial_emb_dim,
                use_layer_norm=critic_use_layer_norm,
                kernel_init=kernel_init,
                features_dim=critic_fd,
            )
            self.critic_target = SpatialEmbQEnsemble(
                num_patches=critic_sc["num_patches"],
                patch_dim=critic_sc["patch_dim"],
                prop_dim=critic_sc["prop_dim"],
                action_space=action_space,
                hidden_dims=critic_arch,
                n_critics=n_critics,
                spatial_emb_dim=critic_spatial_emb_dim,
                use_layer_norm=critic_use_layer_norm,
                kernel_init=kernel_init,
                features_dim=critic_fd,
            )
        elif critic_sc is None:
            self.critic = ContinuousCritic(
                critic_fd,
                action_space,
                hidden_dims=critic_arch,
                n_critics=n_critics,
                use_layer_norm=critic_use_layer_norm,
                use_group_norm=critic_use_group_norm,
                num_groups=num_groups,
                dropout_rate=critic_dropout_rate,
                kernel_init=kernel_init,
                backbone_type=critic_backbone_type,
                critic_impl=critic_impl,
            )
            self.critic_target = ContinuousCritic(
                critic_fd,
                action_space,
                hidden_dims=critic_arch,
                n_critics=n_critics,
                use_layer_norm=critic_use_layer_norm,
                use_group_norm=critic_use_group_norm,
                num_groups=num_groups,
                dropout_rate=critic_dropout_rate,
                kernel_init=kernel_init,
                backbone_type=critic_backbone_type,
                critic_impl=critic_impl,
            )
        else:
            raise ValueError(
                f"Unknown structured_feature_config layout {critic_sc.get('layout')!r}. "
                "Supported: 'token_and_prop'."
            )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # --- Actor ---
        self.actor = Actor(
            self._actor_fd,
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
            log_std_multiplier_init=log_std_multiplier_init,
            log_std_offset_init=log_std_offset_init,
        )

        self.use_cql_alpha_lagrange = False
        self.cql_alpha_lagrange = None

    # --- feature extraction helpers ---

    def extract_features(
        self,
        obs: Obs,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def extract_critic_features(
        self,
        obs: Obs,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        return self.critic_features_extractor.extract(obs, stop_gradient=stop_gradient)

    def critic_features_for(
        self,
        obs: Obs,
        actor_features: torch.Tensor,
        stop_gradient: bool = True,
    ) -> torch.Tensor:
        """Critic-role features for ``obs``, for callers that already hold
        actor-role features for the same ``obs`` (e.g. the actor loss's
        Q(s, pi(s)) term). Reuses ``actor_features`` -- zero extra
        compute, today's exact behavior -- when the encoder is shared;
        re-extracts through the critic's own encoder only when it's
        genuinely separate. ``stop_gradient`` applies only on that
        separate-encoder path; the shared path's gradient behavior is
        whatever ``actor_features`` already has."""
        if self.critic_features_extractor is self.features_extractor:
            return actor_features
        return self.extract_critic_features(obs, stop_gradient=stop_gradient)

    def prepare_batch_all(self, obs: Obs, next_obs: Optional[Obs] = None) -> None:
        """Call ``prepare_batch`` on every distinct extractor instance this
        policy owns (deduped by identity, so the shared default case still
        calls it exactly once)."""
        extractors = {id(self.features_extractor): self.features_extractor}
        extractors[id(self.critic_features_extractor)] = self.critic_features_extractor
        for extractor in extractors.values():
            extractor.prepare_batch(obs, next_obs)

    def _transform_features_for_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Apply token compressor adapter if configured; otherwise identity."""
        if self._actor_adapter is not None:
            return self._actor_adapter(features)
        return features

    # --- public inference API ---

    def forward(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        return self.predict(obs, deterministic=deterministic)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)
        actor_input = self._transform_features_for_actor(features)
        if deterministic:
            return self.actor.deterministic_action(actor_input)
        action, _ = self.actor.action_log_prob(actor_input)
        return action

    # --- helpers for SAC.train() ---

    def actor_action_log_prob(
        self,
        obs: Obs,
        stop_gradient: bool = False,
        detach_encoder: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if detach_encoder is not None:
            stop_gradient = detach_encoder
        features = self.extract_features(obs, stop_gradient=stop_gradient)
        actor_input = self._transform_features_for_actor(features)
        action, log_prob = self.actor.action_log_prob(actor_input)
        return action, log_prob, features

    def evaluate_action_log_prob(
        self,
        obs: Obs,
        actions: torch.Tensor,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        """Log-probability of an externally supplied action (e.g. an offline
        dataset action), for BC-style actor regularizers. Complements
        ``actor_action_log_prob`` the same way
        ``SquashedGaussianActor.evaluate_action_log_prob`` complements
        ``action_log_prob``."""
        features = self.extract_features(obs, stop_gradient=stop_gradient)
        actor_input = self._transform_features_for_actor(features)
        return self.actor.evaluate_action_log_prob(actor_input, actions)

    def actor_diagnostics(self, obs: Obs) -> dict[str, torch.Tensor]:
        """Diagnostic-only actor stats: entropy decomposition and saturation.

        Mirrors the two additive terms of ``SquashedGaussianActor._squashed_log_prob``
        (Gaussian log-density minus tanh-squash correction) so ``entropy`` can be
        read as ``entropy_gaussian + tanh_correction`` without a closed-form
        Gaussian-entropy estimate (which would require collapsing per-dimension,
        per-sample std into a single scalar and incurs a Jensen's-gap bias).
        """
        with torch.no_grad():
            features = self.extract_features(obs, stop_gradient=True)
            actor_input = self._transform_features_for_actor(features)
            mean, log_std = self.actor(actor_input)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            gaussian_term = normal.log_prob(x_t).sum(-1)
            tanh_correction = torch.log(
                self.actor.action_scale * (1 - y_t.pow(2)) + 1e-6
            ).sum(-1)
            return {
                "policy_std": std.mean(),
                "action_saturation": torch.tanh(mean).abs().mean(),
                "entropy_gaussian": -gaussian_term.mean(),
                "tanh_correction": tanh_correction.mean(),
            }

    def q_landscape_diagnostics(
        self,
        obs: Obs,
        *,
        num_actions: int = 8,
        batch_size: int = 64,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Diagnostic-only Q landscape stats for a replay batch.

        The caller owns RNG isolation. ``generator`` must be independent from
        the training RNG if random uniform actions are sampled.
        """
        with torch.enable_grad():
            actor_features = self.extract_features(obs, stop_gradient=True)
            features = self.critic_features_for(obs, actor_features, stop_gradient=True)
            if batch_size < features.shape[0]:
                features = features[:batch_size]
            features = features.detach()
            feature_norm = features.norm(dim=-1).mean()
            feature_dormant = _dormant_ratio(features)

            batch = features.shape[0]
            act_shape = self.action_space.shape
            low = (self.actor.action_bias - self.actor.action_scale).to(features.device)
            high = (self.actor.action_bias + self.actor.action_scale).to(features.device)
            rand = torch.rand(
                (num_actions, batch, *act_shape),
                device=features.device,
                generator=generator,
            )
            actions = low + (high - low) * rand
            flat_features = (
                features.unsqueeze(0)
                .expand(num_actions, *features.shape)
                .reshape(num_actions * batch, -1)
            )
            flat_actions = actions.reshape(num_actions * batch, -1).detach()
            flat_actions.requires_grad_(True)

            q_all = self.q_values_all(flat_features, flat_actions, target=False)
            q_min = q_all.min(dim=0).values.reshape(num_actions, batch, 1)
            q_var = q_min.squeeze(-1).var(dim=0, unbiased=False).mean()
            q_sum = q_min.sum()
            grad = torch.autograd.grad(
                q_sum,
                flat_actions,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            q_grad_norm = grad.norm(dim=-1).mean()

            diagnostics = {
                "q_uniform_var": q_var.detach(),
                "q_action_grad_norm": q_grad_norm.detach(),
                "feature_norm": feature_norm.detach(),
                "feature_dormant_ratio": feature_dormant.detach(),
            }
            if hasattr(self.critic, "trunk_features_first"):
                trunk = self.critic.trunk_features_first(
                    flat_features.detach(),
                    flat_actions.detach(),
                )
                diagnostics["critic_hidden_dormant_ratio"] = _dormant_ratio(trunk).detach()
            return diagnostics

    def _sample_actor_actions(
        self, features: torch.Tensor, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n actions from the actor using the full actor feature path."""
        actor_input = self._transform_features_for_actor(features)
        mean, log_std = self.actor(actor_input)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample(sample_shape=(n,))
        y_t = torch.tanh(x_t)
        action = y_t * self.actor.action_scale + self.actor.action_bias
        log_prob = self.actor._squashed_log_prob(normal, x_t, y_t).squeeze(-1)
        return action.permute(1, 0, 2), log_prob.permute(1, 0)

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

    # --- CQL alpha Lagrange multiplier ---

    def get_cql_alpha(self) -> torch.Tensor:
        raise ValueError(
            "CQL alpha Lagrange multiplier is owned by CQL/CalQL algorithms, "
            "not SACPolicy."
        )

    # --- parameter groups for optimizers ---

    def critic_and_encoder_parameters(self):
        # Encoder trained via Q-loss (matches sac_rgbd.py L581-L585).
        # critic_features_extractor is features_extractor in the (default)
        # shared case, so this is unchanged there.
        yield from self.critic.parameters()
        yield from self.critic_features_extractor.parameters()

    def actor_parameters(self):
        # Actor-only; RGBD actor path uses stop_gradient on image encodings
        # in the shared-encoder case, so features_extractor is deliberately
        # excluded there (it trains via critic_and_encoder_parameters'
        # Q-loss instead). When critic_features_extractor is genuinely
        # separate, features_extractor is actor-exclusive -- nothing else
        # would ever train it -- so it belongs on this optimizer instead.
        if self._actor_adapter is not None:
            yield from self._actor_adapter.parameters()
        if self.critic_features_extractor is not self.features_extractor:
            yield from self.features_extractor.parameters()
        yield from self.actor.parameters()

    def cql_alpha_lagrange_parameters(self):
        if self.use_cql_alpha_lagrange:
            assert self.cql_alpha_lagrange is not None
            yield from self.cql_alpha_lagrange.parameters()
