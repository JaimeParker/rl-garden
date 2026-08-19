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

- ``"shared"`` (default) -- one ``features_extractor``, matching AGENTS.md's
  project convention ("RGBD actor and critic share the encoder; actor
  updates detach encoder features", also SACPolicy's own convention). The
  critic trains it via ``critic_loss``; the actor path gets detached
  features so neither actor network can leak gradient into it.
- ``"separate"`` -- matches FQL's own JAX reference, which builds three
  independent encoder instances (critic, ``actor_bc_flow``,
  ``actor_onestep_flow``, disjoint weights). No detach is used here: the
  critic's own encoder (``features_extractor``) is called with
  ``stop_gradient=False`` for the actor's q_loss term, so gradient DOES
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

EncoderSharing = Literal["shared", "separate"]


class FQLPolicy(BasePolicy):
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
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "FQL requires a Box action space."
        if n_critics < 2:
            raise ValueError(f"n_critics must be >= 2, got {n_critics}.")
        if encoder_sharing not in ("shared", "separate"):
            raise ValueError(
                f"encoder_sharing must be 'shared' or 'separate', got {encoder_sharing!r}."
            )
        if encoder_sharing == "separate":
            if actor_bc_flow_encoder is None or actor_onestep_flow_encoder is None:
                raise ValueError(
                    "encoder_sharing='separate' requires both actor_bc_flow_encoder "
                    "and actor_onestep_flow_encoder."
                )
        elif actor_bc_flow_encoder is not None or actor_onestep_flow_encoder is not None:
            raise ValueError(
                "encoder_sharing='shared' does not accept actor_bc_flow_encoder/"
                "actor_onestep_flow_encoder -- pass encoder_sharing='separate'."
            )

        self.observation_space = observation_space
        self.action_space = action_space
        self.encoder_sharing = encoder_sharing
        # features_extractor is always the critic's own encoder -- in
        # "shared" mode it is also the actor's encoder (with detach).
        self.features_extractor = features_extractor
        if encoder_sharing == "separate":
            self.actor_bc_flow_encoder = actor_bc_flow_encoder
            self.actor_onestep_flow_encoder = actor_onestep_flow_encoder

        fd = features_extractor.features_dim
        action_dim = int(np.prod(action_space.shape))
        net_arch = list(net_arch)

        self.actor_bc_flow = ActorVectorField(
            fd,
            action_dim,
            hidden_dims=net_arch,
            use_time_conditioning=True,
            use_layer_norm=actor_use_layer_norm,
            kernel_init=kernel_init,
            activation_fn=activation_fn,
        )
        self.actor_onestep_flow = ActorVectorField(
            fd,
            action_dim,
            hidden_dims=net_arch,
            use_time_conditioning=False,
            use_layer_norm=actor_use_layer_norm,
            kernel_init=kernel_init,
            activation_fn=activation_fn,
        )

        self.critic = EnsembleQCritic(
            fd,
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
            fd,
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

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        """Critic's own encoder -- also the actor's encoder in 'shared' mode."""
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def extract_actor_onestep_features(self, obs: Obs) -> torch.Tensor:
        """``actor_onestep_flow``'s own encoding of ``obs``. Used both at
        inference (``predict``) and for the critic-target next-action
        computation (always called under ``torch.no_grad()`` there)."""
        if self.encoder_sharing == "separate":
            return self.actor_onestep_flow_encoder.extract(obs, stop_gradient=False)
        return self.features_extractor.extract(obs, stop_gradient=False)

    def extract_actor_loss_features(
        self, obs: Obs, critic_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Features for the actor update: (bc_features, onestep_features,
        q_features).

        ``critic_features`` is the critic block's own already-computed,
        grad-enabled encoding of this same ``obs`` (from ``extract_features``
        earlier in the training step). In 'shared' mode all three returned
        features alias ``critic_features.detach()`` -- reusing it (instead of
        a fresh forward pass) avoids doubling the encoder's compute cost
        every step. In 'separate' mode each is instead a fresh, grad-enabled
        forward through that network's own encoder -- q_features specifically
        *must* be freshly computed, not ``critic_features`` itself: PyTorch
        frees that tensor's graph after ``critic_loss.backward()`` runs
        (earlier in the same step), so a second backward through it would
        raise. This also matches the JAX reference's own behavior of
        re-encoding obs on every ``network.select(...)`` call rather than
        caching across the critic/actor loss functions."""
        if self.encoder_sharing == "separate":
            bc_features = self.actor_bc_flow_encoder.extract(obs, stop_gradient=False)
            onestep_features = self.actor_onestep_flow_encoder.extract(obs, stop_gradient=False)
            q_features = self.features_extractor.extract(obs, stop_gradient=False)
            return bc_features, onestep_features, q_features
        features = critic_features.detach()
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
            yield from self.actor_onestep_flow_encoder.parameters()

    def critic_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from self.features_extractor.parameters()
