"""Single-network diffusion policy for BC pretraining.

Ported from ``3rd_party/dppo/model/diffusion/diffusion.py::DiffusionModel``'s
supervised-training half: one ``DiffusionMLP``, no actor/actor_ft split, no
RL sampling extras (those belong to ``DPPOPolicy``, which reuses the same
``DiffusionProcess`` mixin). State-only (Box observations) -- vision is out
of scope for this port, matching ``DiffusionMLP``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.networks import Activation, DiffusionMLP, KernelInit
from rl_garden.policies._diffusion_process import DiffusionProcess
from rl_garden.policies.base import BasePolicy


class DiffusionPolicy(DiffusionProcess, BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        *,
        horizon_steps: int,
        cond_steps: int,
        denoising_steps: int = 20,
        mlp_dims: Sequence[int] = (512, 512, 512),
        activation_fn: Optional[Activation] = "relu",
        residual_style: bool = True,
        time_dim: int = 16,
        kernel_init: Optional[KernelInit] = None,
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 10.0,
        final_action_clip_value: Optional[float] = None,
        min_sampling_denoising_std: float = 0.1,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "DiffusionPolicy requires a Box action space."
        assert isinstance(
            observation_space, spaces.Box
        ), "DiffusionPolicy is state-only (Box observations); vision is out of scope."
        self.observation_space = observation_space
        self.action_space = action_space
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.min_sampling_denoising_std = min_sampling_denoising_std

        action_dim = int(np.prod(action_space.shape))
        obs_dim = int(np.prod(observation_space.shape))
        cond_dim = obs_dim * cond_steps

        self.net = DiffusionMLP(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=cond_dim,
            time_dim=time_dim,
            mlp_dims=mlp_dims,
            activation_fn=activation_fn,
            residual_style=residual_style,
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

    def loss(self, obs_history: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """``obs_history``: (B, cond_steps, obs_dim). ``action_chunk``:
        (B, horizon_steps, action_dim). Epsilon-prediction MSE at random t."""
        batch = action_chunk.shape[0]
        t = torch.randint(
            0, self.denoising_steps, (batch,), device=action_chunk.device
        )
        return self.p_losses(self.net, action_chunk, {"state": obs_history}, t)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        """Returns the full predicted action chunk, ``(B, horizon_steps,
        action_dim)`` -- chunk execution/slicing is the caller's concern."""
        assert isinstance(obs, torch.Tensor)
        state = obs if obs.dim() == 3 else obs.unsqueeze(1).expand(-1, self.cond_steps, -1)
        cond = {"state": state}
        action_chunk, _ = self.sample_chain(
            cond,
            horizon_steps=self.horizon_steps,
            action_dim=int(self.action_low.shape[0]),
            predict_noise=lambda x, t: self.net(x, t, cond=cond),
            deterministic=deterministic,
            min_sampling_denoising_std=self.min_sampling_denoising_std,
            return_chain=False,
        )
        return action_chunk.clamp(self.action_low, self.action_high)
