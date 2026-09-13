"""IDQL policy: an IQL-style critic ensemble + value head (network classes
reused verbatim from ``IQLPolicy``, `rl_garden/policies/iql_policy.py`) paired
with a diffusion actor (``DiffusionMLP`` + ``DiffusionProcess``, reused
verbatim from ``DiffusionPolicy``, `rl_garden/policies/diffusion_policy.py`)
in place of IQL's Gaussian actor.

Not built by subclassing ``IQLPolicy`` (whose actor half is
Gaussian-specific, e.g. ``behavior_log_prob``) or ``DiffusionPolicy`` (single
inheritance can't compose both) -- the critic/value and actor/diffusion
halves are each a direct, unmodified reuse of the *network classes*
(``EnsembleQCritic``, ``ValueNetwork``, ``DiffusionMLP``) those policies
already use, assembled fresh here. Box or Dict (CNN-based vision, via
``CombinedExtractor``) observations -- ``DiffusionMLP`` only ever sees
``actor_extractor.features_dim``-wide vectors through ``cond["state"]``, so
it does not care whether those features came from raw state or an image
encoder.

Actor/critic extractor contract (``rl_garden.policies.base.BasePolicy``):
value/critic/TD-target heads read through ``extract_critic_features``, the
diffusion actor (``net``/``target_net``) reads through
``extract_actor_features`` -- under the default ``encoder_sharing="shared"``
both resolve to the same one extractor with no stop-gradient (unchanged
behavior), and under ``"separate"`` each role gets its own extractor.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    Activation,
    BackboneType,
    DiffusionMLP,
    EnsembleQCritic,
    KernelInit,
    ValueNetwork,
)
from rl_garden.policies._diffusion_process import DiffusionProcess
from rl_garden.policies.base import BasePolicy, EncoderSharing


class IDQLPolicy(DiffusionProcess, BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Box | spaces.Dict,
        action_space: spaces.Box,
        actor_extractor: BaseFeaturesExtractor,
        *,
        critic_extractor: Optional[BaseFeaturesExtractor] = None,
        encoder_sharing: EncoderSharing = "shared",
        critic_hidden_dims: Sequence[int] = (256, 256),
        value_hidden_dims: Sequence[int] = (256, 256),
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        critic_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        diffusion_mlp_dims: Sequence[int] = (256, 256),
        diffusion_activation_fn: Optional[Activation] = "mish",
        diffusion_residual_style: bool = False,
        time_dim: int = 16,
        denoising_steps: int = 5,
        schedule: Literal["cosine", "vp", "linear"] = "vp",
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 10.0,
        final_action_clip_value: Optional[float] = None,
        min_sampling_denoising_std: float = 0.1,
        n_action_samples: int = 64,
        expectile: float = 0.7,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            actor_extractor=actor_extractor,
            critic_extractor=critic_extractor,
            encoder_sharing=encoder_sharing,
        )
        assert isinstance(action_space, spaces.Box), "IDQLPolicy requires a Box action space."
        if n_critics < 2:
            raise ValueError(f"n_critics must be >= 2, got {n_critics}.")
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size
        self.n_action_samples = n_action_samples
        self.expectile = expectile
        self.min_sampling_denoising_std = min_sampling_denoising_std

        actor_fd = actor_extractor.features_dim
        critic_fd = self.critic_features_dim
        action_dim = int(np.prod(action_space.shape))

        self.critic = EnsembleQCritic(
            critic_fd,
            action_space,
            hidden_dims=critic_hidden_dims,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.critic_target = EnsembleQCritic(
            critic_fd,
            action_space,
            hidden_dims=critic_hidden_dims,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.value = ValueNetwork(
            critic_fd,
            value_hidden_dims,
            use_layer_norm=value_use_layer_norm,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )

        net_kwargs = dict(
            action_dim=action_dim,
            horizon_steps=1,
            cond_dim=actor_fd,
            time_dim=time_dim,
            mlp_dims=diffusion_mlp_dims,
            activation_fn=diffusion_activation_fn,
            residual_style=diffusion_residual_style,
            kernel_init=kernel_init,
        )
        self.net = DiffusionMLP(**net_kwargs)
        self.target_net = DiffusionMLP(**net_kwargs)
        self.target_net.load_state_dict(self.net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self._init_diffusion_process(
            denoising_steps=denoising_steps,
            schedule=schedule,
            denoised_clip_value=denoised_clip_value,
            randn_clip_value=randn_clip_value,
            final_action_clip_value=final_action_clip_value,
        )

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        """Raw actor-extractor access with an explicit ``stop_gradient`` --
        an escape hatch for diagnostics/back-compat callers that need to
        pick the flag themselves. The training loss path
        (``IDQL._compute_losses``/``diffusion_loss``/``predict``) does not
        use this; it calls ``extract_critic_features``/``extract_actor_features``
        (``BasePolicy``), which apply the ``encoder_sharing`` rule
        automatically."""
        return self.actor_extractor.extract(obs, stop_gradient=stop_gradient)

    def q_values_all(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        net = self.critic_target if target else self.critic
        return net.forward_all(features, actions)

    def min_q_value(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        subsample_size: Optional[int] = None,
        target: bool = True,
    ) -> torch.Tensor:
        q_all = self.q_values_all(features, actions, target=target)
        if subsample_size is not None and subsample_size < self.n_critics:
            indices = torch.randint(0, self.n_critics, (subsample_size,), device=q_all.device)
            q_all = q_all[indices]
        return q_all.min(dim=0).values

    def diffusion_loss(
        self, obs: Obs, actions: torch.Tensor, *, weight: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample-weighted epsilon-MSE, ``sum`` over the action dim
        (matches ``ddpm_iql_learner.py``'s ``update_actor`` exactly -- not a
        mean). ``weight``: ``(B,)``, precomputed by the caller."""
        features = self.extract_actor_features(obs)
        batch = actions.shape[0]
        t = torch.randint(0, self.denoising_steps, (batch,), device=actions.device)
        x_start = actions.unsqueeze(1)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred = self.net(x_noisy, t, cond={"state": features})
        per_sample_loss = (pred.squeeze(1) - noise.squeeze(1)).pow(2).sum(dim=-1)
        return (per_sample_loss * weight).mean()

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        actor_features = self.extract_actor_features(obs)
        critic_features = self.extract_critic_features(obs)
        batch = actor_features.shape[0]
        n = self.n_action_samples
        actor_features_rep = actor_features.repeat_interleave(n, dim=0)
        cond = {"state": actor_features_rep}
        action_dim = int(self.action_low.shape[0])
        actions, _ = self.sample_chain(
            cond,
            horizon_steps=1,
            action_dim=action_dim,
            predict_noise=lambda x, t: self.target_net(x, t, cond=cond),
            deterministic=True,
            min_sampling_denoising_std=self.min_sampling_denoising_std,
            return_chain=False,
        )
        actions_flat = actions.squeeze(1)
        critic_features_rep = critic_features.repeat_interleave(n, dim=0)
        q = self.min_q_value(critic_features_rep, actions_flat, target=True).squeeze(-1).view(
            batch, n
        )
        if deterministic:
            idx = q.argmax(dim=1)
        else:
            v = self.value(critic_features)  # (B, 1), broadcasts against (B, n)
            adv = q - v
            weight = torch.where(adv > 0, self.expectile, 1.0 - self.expectile)
            probs = weight / weight.sum(dim=1, keepdim=True)
            idx = torch.multinomial(probs, 1).squeeze(1)
        actions_by_sample = actions_flat.view(batch, n, action_dim)
        selected = actions_by_sample[torch.arange(batch, device=actions.device), idx]
        return selected.clamp(self.action_low, self.action_high)

    def net_parameters(self):
        # Diffusion actor-only; under shared encoder_sharing the encoder
        # trains via critic_value_and_encoder_parameters's critic/value loss
        # instead (mirrors SACPolicy.actor_parameters -- actor_extractor is
        # deliberately excluded there). When critic_extractor is genuinely
        # separate, actor_extractor is actor-exclusive -- nothing else would
        # ever train it -- so it belongs on this optimizer instead.
        if self.critic_extractor is not None and self.critic_extractor is not self.actor_extractor:
            yield from self.actor_extractor.parameters()
        yield from self.net.parameters()

    def critic_value_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from self.value.parameters()
        yield from (self.critic_extractor or self.actor_extractor).parameters()
