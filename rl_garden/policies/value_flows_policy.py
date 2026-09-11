"""Value Flows policy: FQL's two-network flow-matching actor, plus twin
flow-matching critics over the scalar-return axis (Dong et al., "Value
Flows", arXiv 2510.07650, ``3rd_party/value-flows/agents/value_flows.py``).

Extends ``FQLPolicy`` rather than duplicating it: the actor side (BC flow +
one-step distillation) is unchanged FQL. ``FQLPolicy``'s own ``critic``/
``critic_target`` (plain scalar Q-networks) are deleted after
``super().__init__`` -- Value Flows has no separate scalar critic at all; Q
is a one-Euler-step estimate read directly off the same flow networks used
for the TD loss (see ``one_step_q`` and
``rl_garden/algorithms/value_flows.py``).

The reference's twin ``critic_flow1``/``critic_flow2`` are generalized here to
an ``nn.ModuleList`` of ``n_critics`` plain ``ValueFlowVectorField`` instances
(default 2, matching the reference) -- no ``vmap`` ensembling, per this
port's design decisions.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import Activation, BackboneType, KernelInit
from rl_garden.networks.value_flow_field import ValueFlowVectorField
from rl_garden.policies.fql_policy import EncoderSharing, FQLPolicy


def aggregate(stacked: torch.Tensor, mode: str) -> torch.Tensor:
    """Aggregate a ``(K, ...)`` stack over its leading (ensemble) dim."""
    if mode == "min":
        return stacked.min(dim=0).values
    return stacked.mean(dim=0)


class ValueFlowsPolicy(FQLPolicy):
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
        num_samples: int = 16,
        policy_extraction: Literal["rs", "rpg"] = "rs",
        num_flow_steps: int = 10,
        q_agg: Literal["mean", "min"] = "mean",
        return_clip_range: Optional[tuple[float, float]] = None,
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
        del self.critic
        del self.critic_target

        fd = features_extractor.features_dim
        action_dim = int(np.prod(action_space.shape))
        net_arch = list(net_arch)

        flow_kwargs = dict(
            use_layer_norm=critic_use_layer_norm,
            kernel_init=kernel_init,
            activation_fn=activation_fn,
        )
        self.critic_flows = nn.ModuleList(
            [ValueFlowVectorField(fd, action_dim, net_arch, **flow_kwargs) for _ in range(n_critics)]
        )
        self.critic_flows_target = nn.ModuleList(
            [ValueFlowVectorField(fd, action_dim, net_arch, **flow_kwargs) for _ in range(n_critics)]
        )
        self.critic_flows_target.load_state_dict(self.critic_flows.state_dict())
        for p in self.critic_flows_target.parameters():
            p.requires_grad_(False)

        # `predict()` state -- kept on the policy (not just the owning
        # algorithm) so it is self-contained under the fixed
        # `policy.predict(obs, deterministic)` contract used everywhere in
        # the codebase (no extra kwargs at call sites). `policy_extraction`
        # is plain mutable state, not a buffer: `_ValueFlowsRolloutTrainingShell`
        # flips it to "rpg" in place at the offline->online switch.
        self.num_samples = num_samples
        self.policy_extraction = policy_extraction
        self.num_flow_steps = num_flow_steps
        self.q_agg = q_agg
        self.return_clip_range = return_clip_range

    def critic_and_encoder_parameters(self):
        yield from self.critic_flows.parameters()
        yield from self.features_extractor.parameters()

    def one_step_q(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        *,
        clip_range: Optional[tuple[float, float]] = None,
        agg: str = "mean",
    ) -> torch.Tensor:
        """One-Euler-step Q estimate off the (non-target) flow critics:
        ``q_k = (q_noise + flow_k(features, actions, q_noise, 0)).squeeze(-1)``,
        clipped to ``clip_range`` when given, aggregated over the ensemble by
        ``agg``. ``q_noise`` is fresh ``N(0,1)`` noise drawn on every call, per
        the reference (``agents/value_flows.py``'s ``q_noises``)."""
        batch_size = features.shape[0]
        q_noise = torch.randn(batch_size, 1, device=features.device, dtype=features.dtype)
        zeros = torch.zeros_like(q_noise)
        q_list = []
        for flow in self.critic_flows:
            q_k = (q_noise + flow(features, actions, q_noise, zeros)).squeeze(-1)
            if clip_range is not None:
                q_k = q_k.clamp(clip_range[0], clip_range[1])
            q_list.append(q_k)
        return aggregate(torch.stack(q_list, dim=0), agg)

    def sample_actions_rs(
        self,
        features_for_actor: torch.Tensor,
        features_for_q: torch.Tensor,
        *,
        num_samples: int,
        clip_range: Optional[tuple[float, float]] = None,
        agg: str = "mean",
    ) -> torch.Tensor:
        """Rejection-sampling policy extraction: the reference's
        ``sample_actions``, ``policy_extraction='rs'`` branch (its own
        default), ``agents/value_flows.py:343-371``. Draws ``num_samples``
        BC-flow candidates per observation -- the 'rs' branch queries
        ``compute_flow_actions``/``actor_flow`` (the multi-step teacher
        rollout), *not* the one-step student ``actor_onestep_flow`` used by
        ``'rpg'`` -- scores each with the one-step Q off the (non-target)
        critic flows (``one_step_q``, matching the reference's
        ``critic_flow1``/``critic_flow2``, not the target networks), and
        takes the per-observation argmax candidate.

        Shapes: ``features_for_actor``/``features_for_q``: ``(B, F)``.
        Candidates: ``(B, num_samples, A)``. Q: ``(B, num_samples)``. Output:
        ``(B, A)``.

        Caller note (``encoder_sharing="separate"`` only): both call sites
        (``predict`` and ``ValueFlowsCore._critic_update``) pass
        ``extract_actor_onestep_features``'s encoding as
        ``features_for_actor``, even though this method runs
        ``actor_bc_flow`` (via ``compute_flow_actions``), not
        ``actor_onestep_flow`` -- i.e. the BC-flow teacher is fed the
        one-step student's own encoder output, not
        ``actor_bc_flow_encoder``'s. In ``"shared"`` mode this is a no-op
        (one encoder, so the tensors are identical); in ``"separate"`` mode
        it is a deliberate simplification, not reference parity --
        ``extract_actor_loss_features`` would give the BC flow its own
        encoder's features instead, at the cost of a second encoder forward
        pass per call.
        """
        batch_size = features_for_actor.shape[0]
        action_dim = self.actor_onestep_flow.action_dim
        device, dtype = features_for_actor.device, features_for_actor.dtype

        actor_features = features_for_actor.repeat_interleave(num_samples, dim=0)
        q_features = features_for_q.repeat_interleave(num_samples, dim=0)
        noises = torch.randn(batch_size * num_samples, action_dim, device=device, dtype=dtype)
        candidates = self.compute_flow_actions(actor_features, noises, self.num_flow_steps)

        q = self.one_step_q(q_features, candidates, clip_range=clip_range, agg=agg)
        q = q.view(batch_size, num_samples)
        candidates = candidates.view(batch_size, num_samples, action_dim)
        best = q.argmax(dim=-1)
        return candidates[torch.arange(batch_size, device=device), best]

    def predict(self, obs, deterministic: bool = False) -> torch.Tensor:
        """Dispatches on ``self.policy_extraction`` (mirrors ``main.py``'s
        offline-phase eval / offline-critic-target use of ``'rs'`` vs.
        online rollout / online-phase eval's ``'rpg'``). FQL has no separate
        deterministic eval path (see ``FQLPolicy.predict``'s own note);
        ``deterministic`` is accepted for contract compatibility only."""
        del deterministic
        with torch.no_grad():
            if self.encoder_sharing == "separate":
                actor_features = self.extract_actor_onestep_features(obs)
                q_features = self.extract_features(obs)
            else:
                features = self.extract_features(obs)
                actor_features = features
                q_features = features

            if self.policy_extraction == "rs":
                return self.sample_actions_rs(
                    actor_features,
                    q_features,
                    num_samples=self.num_samples,
                    clip_range=self.return_clip_range,
                    agg=self.q_agg,
                )

            noise = self.sample_noise(
                actor_features.shape[0], device=actor_features.device, dtype=actor_features.dtype
            )
            action = self.actor_onestep_flow(actor_features, noise)
            return action.clamp(self.action_low, self.action_high)
