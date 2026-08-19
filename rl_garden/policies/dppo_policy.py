"""DPPO policy: frozen ``actor`` + trainable ``actor_ft`` diffusion pair,
sharing the last ``ft_denoising_steps`` of the DDPM chain, plus an obs-only
critic.

Ported from ``3rd_party/dppo/model/diffusion/diffusion_vpg.py::VPGDiffusion``
(sampling/logprob machinery) verified against source directly. State-only
(Box observations, ``cond_steps=1`` only) -- see the module-level note on
``cond_steps`` below.

``cond_steps`` scope: the reference's observation-history conditioning
(``cond_steps > 1``) requires the online rollout loop to maintain a rolling
window of past observations; every reference example config actually used
(hopper pretrain/finetune) sets ``cond_steps: 1`` regardless, and no rolling
history mechanism exists anywhere in rl-garden's on-policy stack. This port
supports ``cond_steps`` generally on the BC pretraining side (see
``DiffusionPolicy``/``chunked_dataset.py``, which windows real history from
offline trajectories) but requires ``cond_steps == 1`` for online DPPO
fine-tuning -- the loaded BC checkpoint must also have been trained with
``cond_steps == 1``.
"""
from __future__ import annotations

import copy
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Normal

from rl_garden.common.types import Obs
from rl_garden.networks import Activation, DiffusionMLP, KernelInit, build_diffusion_mlp_head
from rl_garden.networks.mlp import _apply_kernel_init, resolve_activation
from rl_garden.policies._diffusion_process import DiffusionProcess
from rl_garden.policies.base import BasePolicy


class _CriticObs(nn.Module):
    """Obs-only value network matching
    ``3rd_party/dppo/model/common/critic.py::CriticObs`` (same
    ``MLP``/``ResidualMLP`` pair as ``DiffusionMLP``'s trunk, via
    ``build_diffusion_mlp_head``)."""

    def __init__(
        self,
        cond_dim: int,
        mlp_dims: Sequence[int],
        *,
        activation_fn: Optional[Activation],
        residual_style: bool,
        kernel_init: Optional[KernelInit],
    ) -> None:
        super().__init__()
        act = resolve_activation(activation_fn, default=nn.Mish)
        self.net = build_diffusion_mlp_head(
            [cond_dim] + list(mlp_dims) + [1], activation_fn=act, residual_style=residual_style
        )
        _apply_kernel_init(self.net, kernel_init)

    def forward(self, cond_state: torch.Tensor) -> torch.Tensor:
        batch = cond_state.shape[0]
        return self.net(cond_state.reshape(batch, -1))


class DPPOPolicy(DiffusionProcess, BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        *,
        horizon_steps: int,
        act_steps: int,
        denoising_steps: int,
        ft_denoising_steps: int,
        actor_mlp_dims: Sequence[int] = (512, 512, 512),
        actor_activation_fn: Optional[Activation] = "relu",
        actor_residual_style: bool = True,
        critic_mlp_dims: Sequence[int] = (256, 256, 256),
        critic_activation_fn: Optional[Activation] = "mish",
        critic_residual_style: bool = True,
        time_dim: int = 16,
        kernel_init: Optional[KernelInit] = None,
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 3.0,
        final_action_clip_value: Optional[float] = None,
        min_sampling_denoising_std: float = 0.1,
        min_logprob_denoising_std: float = 0.1,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "DPPOPolicy requires a Box action space."
        assert isinstance(
            observation_space, spaces.Box
        ), "DPPOPolicy is state-only (Box observations); vision is out of scope."
        if not (1 <= act_steps <= horizon_steps):
            raise ValueError(f"act_steps must be in [1, horizon_steps], got {act_steps}.")
        if not (1 <= ft_denoising_steps <= denoising_steps):
            raise ValueError(
                f"ft_denoising_steps must be in [1, denoising_steps], got {ft_denoising_steps}."
            )
        self.observation_space = observation_space
        self.action_space = action_space
        self.horizon_steps = horizon_steps
        self.act_steps = act_steps
        self.ft_denoising_steps = ft_denoising_steps
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std

        self.action_dim = int(np.prod(action_space.shape))
        self.obs_dim = int(np.prod(observation_space.shape))
        cond_dim = self.obs_dim  # cond_steps == 1 only, see module docstring

        actor = DiffusionMLP(
            action_dim=self.action_dim,
            horizon_steps=horizon_steps,
            cond_dim=cond_dim,
            time_dim=time_dim,
            mlp_dims=actor_mlp_dims,
            activation_fn=actor_activation_fn,
            residual_style=actor_residual_style,
            kernel_init=kernel_init,
        )
        self.actor = actor
        self.actor_ft = copy.deepcopy(actor)
        for p in self.actor.parameters():
            p.requires_grad_(False)

        self.critic = _CriticObs(
            cond_dim,
            critic_mlp_dims,
            activation_fn=critic_activation_fn,
            residual_style=critic_residual_style,
            kernel_init=kernel_init,
        )

        self._init_diffusion_process(
            denoising_steps=denoising_steps,
            denoised_clip_value=denoised_clip_value,
            randn_clip_value=randn_clip_value,
            final_action_clip_value=final_action_clip_value,
        )

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def load_actor_weights(self, net_state_dict: dict) -> None:
        """Loads a ``DiffusionBC``-produced ``ema_net_state_dict`` into both
        ``actor`` (frozen) and ``actor_ft`` (trainable starting point)."""
        self.actor.load_state_dict(net_state_dict)
        self.actor_ft.load_state_dict(net_state_dict)

    def _cond(self, obs: torch.Tensor) -> dict:
        return {"state": obs.unsqueeze(1)}

    def _predict_noise_mixed(
        self, x: torch.Tensor, t: torch.Tensor, cond: dict
    ) -> torch.Tensor:
        """Matches ``VPGDiffusion.p_mean_var``'s noise computation: frozen
        ``actor`` predicts every step, ``actor_ft`` overwrites the last
        ``ft_denoising_steps`` (``t < ft_denoising_steps``)."""
        noise = self.actor(x, t, cond=cond)
        ft_indices = torch.where(t < self.ft_denoising_steps)[0]
        if len(ft_indices) > 0:
            cond_ft = {key: cond[key][ft_indices] for key in cond}
            noise_ft = self.actor_ft(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise = noise.clone()
            noise[ft_indices] = noise_ft
        return noise

    def sample_rollout_chain(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """No-grad DDPM sampling for rollout collection. Returns
        ``(chain, action_chunk)``: ``chain`` is
        ``(N, ft_denoising_steps+1, horizon_steps, action_dim)``,
        ``action_chunk`` is ``(N, horizon_steps, action_dim)``."""
        cond = self._cond(obs)
        x, chain = self.sample_chain(
            cond,
            horizon_steps=self.horizon_steps,
            action_dim=self.action_dim,
            predict_noise=lambda x_t, t: self._predict_noise_mixed(x_t, t, cond),
            deterministic=False,
            min_sampling_denoising_std=self.min_sampling_denoising_std,
            return_chain=True,
            chain_start_t=self.ft_denoising_steps,
        )
        return chain, x

    def get_logprobs(self, cond: dict, chains: torch.Tensor) -> torch.Tensor:
        """Full (non-subsampled) per-denoising-step log-probs for one
        rollout step's chain. ``chains``: (B, ft_denoising_steps+1,
        horizon_steps, action_dim) -> (B, ft_denoising_steps, horizon_steps,
        action_dim). Matches ``VPGDiffusion.get_logprobs`` exactly."""
        batch = chains.shape[0]
        k = self.ft_denoising_steps
        cond_rep = {
            key: cond[key]
            .unsqueeze(1)
            .repeat(1, k, *([1] * (cond[key].dim() - 1)))
            .flatten(0, 1)
            for key in cond
        }
        t_single = torch.arange(k - 1, -1, -1, device=chains.device)
        t_all = t_single.repeat(batch)
        chains_prev = chains[:, :-1].reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains[:, 1:].reshape(-1, self.horizon_steps, self.action_dim)
        noise = self._predict_noise_mixed(chains_prev, t_all, cond_rep)
        mean, logvar = self.p_mean_var(chains_prev, t_all, noise)
        std = torch.exp(0.5 * logvar).clamp(min=self.min_logprob_denoising_std)
        log_prob = Normal(mean, std).log_prob(chains_next)
        return log_prob.reshape(batch, k, self.horizon_steps, self.action_dim)

    def get_logprobs_subsample(
        self,
        cond: dict,
        chains_prev: torch.Tensor,
        chains_next: torch.Tensor,
        denoising_inds: torch.Tensor,
    ) -> torch.Tensor:
        """Grad-enabled log-probs for a random subsample of
        ``(env-step, denoising-step)`` pairs -- the PPO update's
        recompute. Matches ``VPGDiffusion.get_logprobs_subsample``."""
        t_single = torch.arange(self.ft_denoising_steps - 1, -1, -1, device=chains_prev.device)
        t_all = t_single[denoising_inds]
        noise = self._predict_noise_mixed(chains_prev, t_all, cond)
        mean, logvar = self.p_mean_var(chains_prev, t_all, noise)
        std = torch.exp(0.5 * logvar).clamp(min=self.min_logprob_denoising_std)
        return Normal(mean, std).log_prob(chains_next)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._cond(obs)["state"]).view(-1)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        assert isinstance(obs, torch.Tensor)
        cond = self._cond(obs)
        x, _ = self.sample_chain(
            cond,
            horizon_steps=self.horizon_steps,
            action_dim=self.action_dim,
            predict_noise=lambda x_t, t: self._predict_noise_mixed(x_t, t, cond),
            deterministic=deterministic,
            min_sampling_denoising_std=self.min_sampling_denoising_std,
            return_chain=False,
        )
        return x[:, : self.act_steps]

    def clamp_action(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.clamp(self.action_low, self.action_high)
