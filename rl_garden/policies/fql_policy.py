"""FQL policy: twin-Q critic + two coupled flow-matching actor networks.

Independent of SACPolicy/TD3BCPolicy's conventions (no entropy term, no
log_prob, deterministic-given-noise actor) -- mirrors TD3BCPolicy's
independence from BasePolicy directly, not SACFlowPolicy's "subclass and
swap one component" shape, since FQL's actor training itself (two networks,
three loss terms) diverges from every existing rl-garden actor, not just
its architecture.

``actor_bc_flow`` (teacher, time-conditioned, multi-step) and
``actor_onestep_flow`` (student, time-free, single forward pass -- its
output IS the action, not a velocity to integrate further) are both
``ActorVectorField`` instances with disjoint parameters.
``actor_parameters()`` yields from both: this is the entire mechanism that
lets FQL's three-term actor loss (bc_flow_loss + alpha*distill_loss +
q_loss) backprop through one ``actor_optimizer.step()`` call, the same way
TD3BC's two-term actor loss already does.

``encoder_sharing`` controls how the vision encoder is owned:

- ``"shared_critic_grad"`` (default) -- one encoder (``actor_extractor``,
  ``critic_extractor`` left ``None``), matching AGENTS.md's project
  convention ("RGBD actor and critic share the encoder; actor updates detach
  encoder features", also SACPolicy's own convention). The critic trains it
  via ``critic_loss``; the actor path gets detached features so neither
  actor network can leak gradient into it.
- ``"separate"`` -- matches FQL's own JAX reference, which builds three
  independent encoder instances (critic, ``actor_bc_flow``,
  ``actor_onestep_flow``, disjoint weights). ``critic_extractor`` (the
  ``BasePolicy`` slot) is the critic's own encoder; ``actor_extractor`` (the
  other ``BasePolicy`` slot) is ``actor_onestep_flow``'s own encoder;
  ``actor_bc_flow_encoder`` is a THIRD encoder, built directly by
  ``FQLCore`` and kept outside the ``BasePolicy`` actor/critic contract --
  the one documented exception this family takes (see the plan's "FQL
  family" note). No detach is used for the actor's q_loss term: the critic's
  own encoder is called with ``stop_gradient=False``, so gradient DOES
  accumulate into its ``.grad`` buffer during ``actor_loss.backward()`` --
  it is simply never applied, because ``actor_optimizer``'s parameter list
  never includes it (the same "isolation via optimizer grouping, not
  zero-gradient" mechanism already relied on for the critic's own Q-head
  weights in ``q_loss``). Do not mistake this for a zero-gradient guarantee;
  a correctness test here checks parameter-set disjointness between
  ``actor_optimizer`` and the critic's encoder, not the size of any
  particular ``.grad``.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    ActorVectorField,
    Activation,
    BackboneType,
    EnsembleQCritic,
    KernelInit,
)
from rl_garden.policies.base import BasePolicy

# This family only implements two of the mixin's three values (see
# rl_garden.algorithms._observation.EncoderSharing) -- "shared" (both losses
# train one encoder, no detach) has no implementation here; see __init__'s
# validation below. Kept as a local Literal, not imported from
# rl_garden.algorithms._observation: rl_garden/policies/*.py must not import
# from rl_garden.algorithms.* at module scope (importing any submodule of
# rl_garden.algorithms runs algorithms/__init__.py, which eagerly imports
# fql.py -> this module, i.e. import rl_garden.algorithms._observation from
# here IS a circular import, confirmed empirically -- not just theoretical).
EncoderSharing = Literal["shared_critic_grad", "separate"]


class FQLPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        actor_extractor: BaseFeaturesExtractor,
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
        encoder_sharing: EncoderSharing = "shared_critic_grad",
        critic_extractor: Optional[BaseFeaturesExtractor] = None,
        actor_bc_flow_encoder: Optional[BaseFeaturesExtractor] = None,
    ) -> None:
        assert isinstance(action_space, spaces.Box), "FQL requires a Box action space."
        if n_critics < 2:
            raise ValueError(f"n_critics must be >= 2, got {n_critics}.")
        if encoder_sharing not in ("shared_critic_grad", "separate"):
            raise ValueError(
                "encoder_sharing must be 'shared_critic_grad' or 'separate', got "
                f"{encoder_sharing!r}."
            )
        if encoder_sharing == "separate":
            if critic_extractor is None or actor_bc_flow_encoder is None:
                raise ValueError(
                    "encoder_sharing='separate' requires both critic_extractor "
                    "and actor_bc_flow_encoder."
                )
        elif critic_extractor is not None or actor_bc_flow_encoder is not None:
            raise ValueError(
                "encoder_sharing='shared_critic_grad' does not accept "
                "critic_extractor/actor_bc_flow_encoder -- pass "
                "encoder_sharing='separate'."
            )
        super().__init__(
            observation_space,
            action_space,
            actor_extractor=actor_extractor,
            critic_extractor=critic_extractor,
            encoder_sharing=encoder_sharing,
        )
        # actor_bc_flow_encoder is FQL's one documented exception to the
        # actor_extractor/critic_extractor contract: a third, independent
        # encoder for the BC-flow teacher, only present under "separate".
        if encoder_sharing == "separate":
            self.actor_bc_flow_encoder = actor_bc_flow_encoder

        actor_fd = self.actor_features_dim
        critic_fd = self.critic_features_dim
        action_dim = int(np.prod(action_space.shape))
        net_arch = list(net_arch)

        self.actor_bc_flow = ActorVectorField(
            actor_fd,
            action_dim,
            hidden_dims=net_arch,
            use_time_conditioning=True,
            use_layer_norm=actor_use_layer_norm,
            kernel_init=kernel_init,
            activation_fn=activation_fn,
        )
        self.actor_onestep_flow = ActorVectorField(
            actor_fd,
            action_dim,
            hidden_dims=net_arch,
            use_time_conditioning=False,
            use_layer_norm=actor_use_layer_norm,
            kernel_init=kernel_init,
            activation_fn=activation_fn,
        )

        self.critic = EnsembleQCritic(
            critic_fd,
            action_space,
            hidden_dims=net_arch,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            activation_fn=activation_fn,
        )
        self.critic_target = EnsembleQCritic(
            critic_fd,
            action_space,
            hidden_dims=net_arch,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            activation_fn=activation_fn,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def extract_actor_onestep_features(self, obs: Obs) -> torch.Tensor:
        """``actor_onestep_flow``'s own encoding of ``obs`` (``actor_extractor``
        -- the critic's own encoder too, in 'shared' mode). Used both at
        inference (``predict``) and for the critic-target next-action
        computation (always called under ``torch.no_grad()`` there)."""
        return self.actor_extractor.extract(obs, stop_gradient=False)

    def extract_actor_loss_features(
        self, obs: Obs, critic_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Features for the actor update: (bc_features, onestep_features,
        q_features).

        ``critic_features`` is the critic block's own already-computed,
        grad-enabled encoding of this same ``obs`` (from
        ``extract_critic_features`` earlier in the training step). In
        'shared' mode all three returned features alias
        ``critic_features.detach()`` -- reusing it (instead of a fresh
        forward pass) avoids doubling the encoder's compute cost every step.
        In 'separate' mode each is instead a fresh, grad-enabled forward
        through that network's own encoder -- q_features specifically *must*
        be freshly computed, not ``critic_features`` itself: PyTorch frees
        that tensor's graph after ``critic_loss.backward()`` runs (earlier in
        the same step), so a second backward through it would raise. This
        also matches the JAX reference's own behavior of re-encoding obs on
        every ``network.select(...)`` call rather than caching across the
        critic/actor loss functions."""
        if self.encoder_sharing == "separate":
            bc_features = self.actor_bc_flow_encoder.extract(obs, stop_gradient=False)
            onestep_features = self.actor_extractor.extract(obs, stop_gradient=False)
            q_features = self.critic_extractor.extract(obs, stop_gradient=False)
            return bc_features, onestep_features, q_features
        # Not "separate" -> one shared extractor for all three roles.
        # actor_features_detached (BasePolicy's single stop-gradient rule)
        # decides whether to detach, rather than hardcoding it: today FQL's
        # own encoder_sharing validation only ever allows
        # "shared_critic_grad" here (always detached), but this stays
        # correct if that were ever relaxed to "shared" too.
        features = (
            critic_features.detach() if self.actor_features_detached else critic_features
        )
        return features, features, features

    def sample_noise(self, batch_size: int, *, device, dtype) -> torch.Tensor:
        action_dim = self.actor_onestep_flow.action_dim
        return torch.randn(batch_size, action_dim, device=device, dtype=dtype)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        # FQL has no separate deterministic eval path: the reference always
        # samples fresh N(0,1) noise as the flow's generative latent input
        # (like a GAN's z), not exploration noise to zero out. `deterministic`
        # is accepted for contract compatibility but has no effect here.
        del deterministic
        features = self.extract_actor_onestep_features(obs)
        noise = self.sample_noise(features.shape[0], device=features.device, dtype=features.dtype)
        action = self.actor_onestep_flow(features, noise)
        return action.clamp(self.action_low, self.action_high)

    def compute_flow_actions(self, features: torch.Tensor, noises: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Multi-step teacher rollout used only as the distill-loss target."""
        return self.actor_bc_flow.integrate(
            features, noises, num_steps, low=self.action_low, high=self.action_high
        )

    def q_values_all(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        net = self.critic_target if target else self.critic
        return net.forward_all(features, actions)

    def actor_parameters(self):
        yield from self.actor_bc_flow.parameters()
        yield from self.actor_onestep_flow.parameters()
        if self.encoder_sharing == "separate":
            yield from self.actor_bc_flow_encoder.parameters()
            yield from self.actor_extractor.parameters()

    def critic_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from (self.critic_extractor or self.actor_extractor).parameters()
