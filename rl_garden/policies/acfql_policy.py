"""ACFQL policy: ``FQLPolicy`` plus QC's ``actor_type`` action-selection dispatch.

Network construction (both ``ActorVectorField`` instances, ``EnsembleQCritic``)
is entirely unchanged from ``FQLPolicy`` -- action dimensionality is already
generic (``action_dim = prod(action_space.shape)``), so passing a synthetic
flat ``(horizon_length * action_dim,)`` action space at construction time
(see ``ACFQLCore._setup_model``) is the whole chunking adaptation. The only
new behavior here is ``predict()`` dispatching on ``actor_type``
(``3rd_party/qc/agents/acfql.py:161-205``):

- ``"distill-ddpg"`` (default): identical to ``FQLPolicy.predict`` -- one
  ``actor_onestep_flow`` forward pass.
- ``"best-of-n"``: no trained ``actor_onestep_flow`` (QC's own reference sets
  ``distill_loss=q_loss=0`` in this mode, so it never receives gradient);
  instead draws ``actor_num_samples`` full Euler-integrated BC-flow samples
  and picks the argmax-Q one per observation.

Both training (``ACFQLCore.train``, for the critic's bootstrap next-action)
and rollout/eval (this class's ``predict``) go through the same dispatch, so
neither could silently diverge -- verified against ``acfql.py``'s
``critic_loss`` calling the same ``self.sample_actions`` used everywhere
else, not a training-only shortcut.

``best_of_n_action`` assumes ``encoder_sharing="shared"`` (uses
``extract_features``, the critic/shared encoder, for both the BC-flow
sampling and the Q evaluation) -- correct for v1's Box-only, shared-encoder
default; not exercised under ``encoder_sharing="separate"``.
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

ActorType = Literal["distill-ddpg", "best-of-n"]


class ACFQLPolicy(FQLPolicy):
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
        actor_type: ActorType = "distill-ddpg",
        actor_num_samples: int = 32,
        flow_steps: int = 10,
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
        if actor_type not in ("distill-ddpg", "best-of-n"):
            raise ValueError(f"actor_type must be 'distill-ddpg' or 'best-of-n', got {actor_type!r}.")
        self.actor_type = actor_type
        self.actor_num_samples = actor_num_samples
        self.flow_steps = flow_steps
        self.q_agg = q_agg

    def _aggregate_q(self, q_all: torch.Tensor) -> torch.Tensor:
        return q_all.mean(dim=0) if self.q_agg == "mean" else q_all.min(dim=0).values

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        del deterministic
        if self.actor_type == "distill-ddpg":
            return super().predict(obs)
        return self.best_of_n_action(obs)

    def best_of_n_action(self, obs: Obs) -> torch.Tensor:
        features = self.extract_features(obs)
        batch, n = features.shape[0], self.actor_num_samples
        rep_features = features.repeat_interleave(n, dim=0)
        noises = torch.randn(
            batch * n, self.actor_bc_flow.action_dim, device=features.device, dtype=features.dtype
        )
        actions = self.compute_flow_actions(rep_features, noises, self.flow_steps)
        q_all = self.q_values_all(rep_features, actions, target=False)
        q = self._aggregate_q(q_all).view(batch, n)
        best_idx = q.argmax(dim=1)
        actions = actions.view(batch, n, -1)
        return actions[torch.arange(batch, device=actions.device), best_idx]
