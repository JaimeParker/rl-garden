"""FINO policy: FQL's twin-Q critic + two-network flow-matching actor, with a
rejection-sampled (argmax/Boltzmann) inference-time policy extraction (Shin
et al., "Flow Matching with Injected Noise for Offline-to-Online RL", ICLR
2026, ``3rd_party/FINO/agents/fino.py``).

Extends ``FQLPolicy`` rather than duplicating it: no new networks are added
or deleted -- ``critic``/``critic_target``/``actor_bc_flow``/
``actor_onestep_flow`` are all inherited unchanged. The only new behavior is
``predict``/``sample_actions``: instead of FQL's plain one-step draw, FINO
samples ``num_samples`` one-step-student candidates per observation, scores
them with the live (non-target) critic, and either argmaxes (deterministic
eval) or Boltzmann-samples (stochastic online rollout) over them.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import Activation, BackboneType, KernelInit
from rl_garden.policies.fql_policy import EncoderSharing, FQLPolicy


class FINOPolicy(FQLPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] = (512, 512, 512, 512),
        *,
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = True,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        activation_fn: Optional[Activation] = None,
        encoder_sharing: EncoderSharing = "shared",
        actor_bc_flow_encoder: Optional[BaseFeaturesExtractor] = None,
        actor_onestep_flow_encoder: Optional[BaseFeaturesExtractor] = None,
        beta: float = 10.0,
        num_samples: Optional[int] = None,
        q_agg: Literal["mean", "min"] = "mean",
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            net_arch,
            n_critics=n_critics,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            activation_fn=activation_fn,
            encoder_sharing=encoder_sharing,
            actor_bc_flow_encoder=actor_bc_flow_encoder,
            actor_onestep_flow_encoder=actor_onestep_flow_encoder,
        )
        action_dim = int(np.prod(action_space.shape))
        self.beta = beta
        # TODO: the reference (fino.py:215-232, `update_entropy`) adapts
        # `beta` after every eval via a scikit-learn GaussianMixture entropy
        # estimate over sampled actions (`beta <- max(0, beta - 0.1*(-action_dim
        # - entropy))`); rl-garden has no post-eval algorithm hook today, so
        # `beta` is fixed here (see FINOCore's module docstring / the FINO
        # port plan's decision 2).
        self.num_samples = num_samples if num_samples is not None else min(10, (action_dim + 1) // 2)
        self.q_agg = q_agg

    def _aggregate_q(self, q_all: torch.Tensor) -> torch.Tensor:
        if self.q_agg == "min":
            return q_all.min(dim=0).values
        return q_all.mean(dim=0)

    def _sample_candidates(
        self, actor_features: torch.Tensor, q_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``self.num_samples`` one-step-student candidates per
        observation and score them with the live (non-target) critic.

        Returns ``(candidates, q)`` of shape ``(B, K, A)`` / ``(B, K)``.
        """
        batch_size = actor_features.shape[0]
        num_samples = self.num_samples
        action_dim = self.actor_onestep_flow.action_dim
        device, dtype = actor_features.device, actor_features.dtype

        actor_features_r = actor_features.repeat_interleave(num_samples, dim=0)
        q_features_r = q_features.repeat_interleave(num_samples, dim=0)
        noise = torch.randn(batch_size * num_samples, action_dim, device=device, dtype=dtype)
        candidates = self.actor_onestep_flow(actor_features_r, noise)
        candidates = candidates.clamp(self.action_low, self.action_high)

        q_all = self.q_values_all(q_features_r, candidates, target=False)
        q = self._aggregate_q(q_all).squeeze(-1)

        return (
            candidates.view(batch_size, num_samples, action_dim),
            q.view(batch_size, num_samples),
        )

    def sample_actions(self, obs: Obs, *, deterministic: bool) -> torch.Tensor:
        if self.encoder_sharing == "separate":
            actor_features = self.extract_actor_onestep_features(obs)
            q_features = self.extract_features(obs)
        else:
            features = self.extract_features(obs)
            actor_features = features
            q_features = features

        candidates, q = self._sample_candidates(actor_features, q_features)
        batch_size = candidates.shape[0]
        device = candidates.device

        if deterministic:
            # Per-observation argmax over K candidates. The reference
            # (fino.py:188-190, `to_eval`) instead does
            # `actions[jnp.argmax(q)]` with `q` of shape `(K, B)` and no
            # `axis` argument -- a flat index into the whole `K*B` array,
            # used to index `actions`' *first* axis only. For `B > 1` this
            # silently picks the wrong (or out-of-range-relative-to-intent)
            # candidate for most observations. This port implements the
            # intended per-observation argmax instead.
            idx = q.argmax(dim=-1)
        else:
            # Global scalar normalizer, matching the reference's
            # `jnp.abs(q).mean()` (fino.py:182), which reduces over the
            # entire (K, B) array to one scalar -- preserved as-is.
            norm = q.abs().mean().clamp_min(1e-8)
            logits = (q / norm) * self.beta
            idx = torch.distributions.Categorical(logits=logits).sample()

        return candidates[torch.arange(batch_size, device=device), idx]

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            return self.sample_actions(obs, deterministic=deterministic)
