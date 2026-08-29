"""A2A ("Action-to-Action") flow-matching policy for BC pretraining.

Ports ``3rd_party/A2A_Flow_Matching``'s ``A2AImagePolicy``
(``roboverse_learn/il/policies/a2a/a2a_policy.py``): flow matching where the
source ``x_0`` is an encoded window of past proprio **states** (not noise --
``FlowBCPolicy``'s convention), the flow operates in a compressed latent
space via a learned action-chunk encoder/decoder (not raw action space), and
vision is encoded separately as the flow's **condition**. Everything
downstream of "encode history -> flow -> decode" reuses existing rl-garden
building blocks unmodified:

- ``ActorVectorField`` (``rl_garden/networks/actor_vector_field.py``) IS the
  flow network -- constructed with ``action_dim=latent_dim`` (not the env's
  actual action dim), so it operates on latent-space vectors. Its
  ``integrate()`` already accepts an arbitrary ``x_0``, so feeding it
  ``history_latents`` instead of noise requires no changes there.
- The CondOT loss (``t~U(0,1)``, ``x_t=(1-t)x_0+t*x_1``, target velocity
  ``x_1-x_0``) is the exact pattern from ``FlowBCPolicy.bc_flow_loss``, with
  ``x_0``/``x_1`` swapped for the learned ``history_latents``/
  ``future_action_latents`` instead of noise/raw actions.
- The "fold cond_steps into batch, encode once, reshape back" trick for
  running a single-frame ``CombinedExtractor`` over a history window is
  ``VisionDiffusionPolicy._encode_obs_history``'s trick, reused here for the
  vision-only conditioning branch (``CombinedExtractor(use_proprio=False)``).

New pieces: ``CNNSequenceEncoder`` (state-history and action-chunk encoding,
two disjoint-parameter instances) and ``ActionChunkDecoder``
(``rl_garden/networks/sequence_cnn.py``).

Design decisions carried over from the reference verbatim (not re-litigated
here): the encoder is deterministic, not a VAE (the reference config's
``use_variational``/``kl_weight`` are unused by ``A2AImagePolicy.compute_loss``);
no stop-gradients anywhere in the loss -- both ``x_0`` and ``x_1`` are learned
encoder outputs, so ``flow_loss`` backprops into both encoders and
``consistency_loss`` backprops through every Euler step into the flow net,
history encoder, and (via ``obs_latents``) the vision encoder; this is not
degenerate because ``enc_recon_weight``/``flow_recon_weight`` (both on by
default) force the shared latent space to stay decodable.

The contrastive losses (``enc_contrastive_weight``/``flow_contrastive_weight``,
both 0.0 by default, matching the reference config) use a standard symmetric
in-batch-negatives InfoNCE, not independently re-derived against the
reference's exact formulation -- a documented, inactive-by-default gap.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.common.obs_utils import flatten_leading_dims
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    ActionChunkDecoder,
    Activation,
    ActorVectorField,
    CNNSequenceEncoder,
    KernelInit,
)
from rl_garden.policies.base import BasePolicy


class A2APolicy(BasePolicy):
    """Actor-only policy for A2A flow-matching BC. No critic, no distillation."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        *,
        horizon_steps: int = 8,
        cond_steps: int = 8,
        state_key: str = "state",
        latent_dim: int = 512,
        cnn_num_layers: int = 3,
        cnn_hidden_channels: int = 512,
        cnn_kernel_size: int = 5,
        cnn_activation_fn: Optional[Activation] = "relu",
        decoder_net_arch: Sequence[int] = (512, 512, 512, 512),
        decoder_activation_fn: Optional[Activation] = None,
        decoder_kernel_init: Optional[KernelInit] = None,
        flow_hidden_dims: Sequence[int] = (512, 512, 512, 512),
        flow_use_layer_norm: bool = False,
        flow_kernel_init: Optional[KernelInit] = None,
        flow_activation_fn: Optional[Activation] = None,
        num_sampling_steps: int = 6,
        consistency_weight: float = 1.0,
        enc_recon_weight: float = 0.5,
        flow_recon_weight: float = 0.5,
        enc_contrastive_weight: float = 0.0,
        flow_contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "A2APolicy requires a Box action space."
        assert isinstance(observation_space, spaces.Dict), (
            "A2APolicy requires a Dict observation space."
        )
        if state_key not in observation_space.spaces:
            raise ValueError(
                f"A2APolicy requires state_key={state_key!r} in the observation "
                "space -- the state-history window is the flow's source (x_0), "
                "not optional."
            )
        self.observation_space = observation_space
        self.action_space = action_space
        self.state_key = state_key
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.latent_dim = latent_dim
        self.num_sampling_steps = num_sampling_steps
        self.consistency_weight = consistency_weight
        self.enc_recon_weight = enc_recon_weight
        self.flow_recon_weight = flow_recon_weight
        self.enc_contrastive_weight = enc_contrastive_weight
        self.flow_contrastive_weight = flow_contrastive_weight
        self.contrastive_temperature = contrastive_temperature

        self.state_dim = int(np.prod(observation_space[state_key].shape))
        self.action_dim = int(np.prod(action_space.shape))

        self.features_extractor = features_extractor
        self.obs_latent_proj = nn.Linear(features_extractor.features_dim * cond_steps, latent_dim)

        self.history_encoder = CNNSequenceEncoder(
            input_dim=self.state_dim,
            seq_len=cond_steps,
            latent_dim=latent_dim,
            num_layers=cnn_num_layers,
            hidden_channels=cnn_hidden_channels,
            kernel_size=cnn_kernel_size,
            activation_fn=cnn_activation_fn,
        )
        self.action_chunk_encoder = CNNSequenceEncoder(
            input_dim=self.action_dim,
            seq_len=horizon_steps,
            latent_dim=latent_dim,
            num_layers=cnn_num_layers,
            hidden_channels=cnn_hidden_channels,
            kernel_size=cnn_kernel_size,
            activation_fn=cnn_activation_fn,
        )
        self.action_decoder = ActionChunkDecoder(
            latent_dim=latent_dim,
            horizon=horizon_steps,
            output_dim=self.action_dim,
            net_arch=decoder_net_arch,
            activation_fn=decoder_activation_fn,
            kernel_init=decoder_kernel_init,
        )

        # ActorVectorField reused unmodified: action_dim IS latent_dim (flow
        # operates in the shared latent space, not raw action space);
        # features_dim IS latent_dim (obs_latents' width after obs_latent_proj).
        self.flow_net = ActorVectorField(
            latent_dim,
            latent_dim,
            hidden_dims=list(flow_hidden_dims),
            use_time_conditioning=True,
            use_layer_norm=flow_use_layer_norm,
            kernel_init=flow_kernel_init,
            activation_fn=flow_activation_fn,
        )

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def _encode_obs_latents(self, obs_history: Obs, stop_gradient: bool) -> torch.Tensor:
        """``obs_history``: Dict, each leaf ``(B, cond_steps, *leaf_shape)``.
        Returns ``(B, latent_dim)``. ``features_extractor`` is a vision-only
        (``use_proprio=False``) extractor, so ``state_key`` in ``obs_history``
        is dropped automatically -- no stripping needed here."""
        batch = obs_history[self.state_key].shape[0]
        flat_obs = flatten_leading_dims(obs_history)
        flat_features = self.features_extractor.extract(flat_obs, stop_gradient=stop_gradient)
        features = flat_features.reshape(batch, self.cond_steps, -1).flatten(1)
        return self.obs_latent_proj(features)

    def _info_nce(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        batch = anchor.shape[0]
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = anchor @ positive.T / self.contrastive_temperature
        labels = torch.arange(batch, device=logits.device)
        loss_a2p = F.cross_entropy(logits, labels)
        loss_p2a = F.cross_entropy(logits.T, labels)
        return (loss_a2p + loss_p2a) / 2

    def loss_with_metrics(
        self, obs_history: Obs, action_chunk: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch = action_chunk.shape[0]
        obs_latents = self._encode_obs_latents(obs_history, stop_gradient=False)
        history_latents = self.history_encoder(obs_history[self.state_key])
        future_action_latents = self.action_chunk_encoder(action_chunk)

        # CondOT flow loss -- exact FlowBCPolicy.bc_flow_loss pattern
        # (flow_bc_policy.py:91-103), x_0/x_1 swapped for the learned latents.
        t = torch.rand(batch, 1, device=action_chunk.device, dtype=action_chunk.dtype)
        x_t = (1 - t) * history_latents + t * future_action_latents
        vel_target = future_action_latents - history_latents
        pred_vel = self.flow_net(obs_latents, x_t, t)
        flow_loss = F.mse_loss(pred_vel, vel_target)
        total = flow_loss
        metrics = {"flow_loss": float(flow_loss.detach())}

        needs_sample = (
            self.consistency_weight > 0
            or self.flow_recon_weight > 0
            or self.flow_contrastive_weight > 0
        )
        if needs_sample:
            # No stop-gradient (matches the reference exactly -- see module
            # docstring): gradient flows through all num_sampling_steps Euler
            # steps into flow_net, history_encoder, and (via obs_latents) the
            # vision features_extractor.
            action_latents_pred = self.flow_net.integrate(
                obs_latents, history_latents, self.num_sampling_steps
            )

        if self.consistency_weight > 0:
            consistency_loss = F.mse_loss(action_latents_pred, future_action_latents)
            total = total + self.consistency_weight * consistency_loss
            metrics["consistency_loss"] = float(consistency_loss.detach())
        if self.flow_recon_weight > 0:
            flow_recon_loss = F.l1_loss(self.action_decoder(action_latents_pred), action_chunk)
            total = total + self.flow_recon_weight * flow_recon_loss
            metrics["flow_action_recon_loss"] = float(flow_recon_loss.detach())
        if self.enc_recon_weight > 0:
            enc_recon_loss = F.l1_loss(self.action_decoder(future_action_latents), action_chunk)
            total = total + self.enc_recon_weight * enc_recon_loss
            metrics["enc_action_recon_loss"] = float(enc_recon_loss.detach())
        if self.enc_contrastive_weight > 0:
            c = self._info_nce(obs_latents, future_action_latents)
            total = total + self.enc_contrastive_weight * c
            metrics["enc_contrastive_loss"] = float(c.detach())
        if self.flow_contrastive_weight > 0:
            c = self._info_nce(obs_latents, action_latents_pred)
            total = total + self.flow_contrastive_weight * c
            metrics["flow_contrastive_loss"] = float(c.detach())

        metrics["loss"] = float(total.detach())
        return total, metrics

    def loss(self, obs_history: Obs, action_chunk: torch.Tensor) -> torch.Tensor:
        return self.loss_with_metrics(obs_history, action_chunk)[0]

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        """``obs``: Dict, each leaf ``(B, *leaf_shape)`` (single frame,
        broadcast to ``cond_steps``) or ``(B, cond_steps, *leaf_shape)``
        (explicit history). Returns the full predicted action chunk,
        ``(B, horizon_steps, action_dim)`` -- chunk execution/slicing is the
        caller's concern (see ``RecedingHorizonPolicy``)."""
        del deterministic  # Euler-sampling always the same, matches FlowBCPolicy.predict's stance
        assert isinstance(obs, dict)
        leaf_ndim = len(self.observation_space[self.state_key].shape)
        is_single_frame = obs[self.state_key].dim() == leaf_ndim + 1
        if is_single_frame:
            obs_history = {
                key: value.unsqueeze(1).expand(-1, self.cond_steps, *([-1] * (value.dim() - 1)))
                for key, value in obs.items()
            }
        else:
            obs_history = obs
        obs_latents = self._encode_obs_latents(obs_history, stop_gradient=True)
        history_latents = self.history_encoder(obs_history[self.state_key])
        action_latents = self.flow_net.integrate(obs_latents, history_latents, self.num_sampling_steps)
        action_chunk = self.action_decoder(action_latents)
        return action_chunk.clamp(self.action_low, self.action_high)
