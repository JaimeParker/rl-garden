"""Vision-conditioned diffusion policy for BC pretraining.

A sibling of ``DiffusionPolicy`` (same ``DiffusionProcess``/``BasePolicy``
base classes), not a subclass -- ``DiffusionPolicy`` is Box-only by
assertion and is the network shape ``DPPOPolicy.load_actor_weights`` loads
``DiffusionBC`` checkpoints into (``dppo_policy.py:147-152``); changing that
class's conditioning shape would silently break DPPO's checkpoint contract.
This class reuses everything downstream of the conditioning vector unchanged
(``DiffusionMLP``, ``DiffusionProcess``'s DDPM math) -- only observation
encoding differs: raw flat state is replaced by a ``BaseFeaturesExtractor``
(``CombinedExtractor``, image+proprio fusion) run once per conditioning
frame, folding the ``cond_steps`` time axis into the batch dimension before
the encoder forward and reshaping back after -- the same trick
``real-stanford/diffusion_policy``'s own ``MultiImageObsEncoder`` uses for
its observation history.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.obs_utils import flatten_leading_dims
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import Activation, DiffusionMLP, KernelInit
from rl_garden.policies._diffusion_process import DiffusionProcess
from rl_garden.policies.base import BasePolicy


class VisionDiffusionPolicy(DiffusionProcess, BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
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
        assert isinstance(action_space, spaces.Box), (
            "VisionDiffusionPolicy requires a Box action space."
        )
        assert isinstance(observation_space, spaces.Dict), (
            "VisionDiffusionPolicy requires a Dict observation space; use "
            "DiffusionPolicy for Box (state-only) observations."
        )
        self.observation_space = observation_space
        self.action_space = action_space
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.features_extractor = features_extractor

        action_dim = int(np.prod(action_space.shape))
        cond_dim = features_extractor.features_dim * cond_steps

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

    def _encode_obs_history(self, obs_history: Obs, stop_gradient: bool = False) -> torch.Tensor:
        """``obs_history``: Dict of tensors each ``(B, cond_steps, *leaf_shape)``.
        Returns ``(B, cond_steps, features_dim)``."""
        assert isinstance(obs_history, dict)
        batch = next(iter(obs_history.values())).shape[0]
        flat_obs = flatten_leading_dims(obs_history)
        flat_features = self.features_extractor.extract(flat_obs, stop_gradient=stop_gradient)
        return flat_features.reshape(batch, self.cond_steps, -1)

    def loss(self, obs_history: Obs, action_chunk: torch.Tensor) -> torch.Tensor:
        """``obs_history``: Dict, each leaf ``(B, cond_steps, *leaf_shape)``.
        ``action_chunk``: (B, horizon_steps, action_dim). Epsilon-prediction
        MSE at random t."""
        batch = action_chunk.shape[0]
        t = torch.randint(
            0, self.denoising_steps, (batch,), device=action_chunk.device
        )
        features = self._encode_obs_history(obs_history, stop_gradient=False)
        return self.p_losses(self.net, action_chunk, {"state": features}, t)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        """``obs``: Dict, each leaf ``(B, *leaf_shape)`` (single frame,
        broadcast to ``cond_steps``) or ``(B, cond_steps, *leaf_shape)``
        (explicit history). Returns the full predicted action chunk,
        ``(B, horizon_steps, action_dim)`` -- chunk execution/slicing is the
        caller's concern (see ``RecedingHorizonPolicy``)."""
        assert isinstance(obs, dict)
        sample_key = next(iter(obs))
        leaf_ndim = len(self.observation_space[sample_key].shape)
        is_single_frame = obs[sample_key].dim() == leaf_ndim + 1
        if is_single_frame:
            # Single frame per key: broadcast to cond_steps, matching
            # DiffusionPolicy.predict()'s own Box single-frame handling.
            obs_history = {
                key: value.unsqueeze(1).expand(-1, self.cond_steps, *([-1] * (value.dim() - 1)))
                for key, value in obs.items()
            }
        else:
            obs_history = obs
        features = self._encode_obs_history(obs_history, stop_gradient=True)
        cond = {"state": features}
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
