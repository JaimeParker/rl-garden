"""Single-network diffusion policy for BC pretraining.

Ported from ``3rd_party/dppo/model/diffusion/diffusion.py::DiffusionModel``'s
supervised-training half: one denoiser network, no actor/actor_ft split, no
RL sampling extras (those belong to ``DPPOPolicy``, which reuses the same
``DiffusionProcess`` mixin).

Handles both Box (state-only) and Dict (vision) observations as an
``isinstance(observation_space, spaces.Box/Dict)`` branch, matching
``FlowBC``'s convention (``rl_garden/algorithms/flow_bc.py``) -- this class
absorbs the former standalone ``VisionDiffusionPolicy``. The Box branch never
builds or stores a ``features_extractor``: it conditions directly on the raw
flat obs tensor, exactly as before the merge, so ``net.*`` state-dict keys
and ``cond_dim`` stay byte-identical to the pre-merge Box-only class -- this
is required because ``DPPOPolicy.load_actor_weights``
(``rl_garden/policies/dppo_policy.py``) loads ``DiffusionBC`` Box checkpoints
directly into ``DPPOPolicy``'s own network. The Dict branch builds a
``BaseFeaturesExtractor`` (typically ``CombinedExtractor``, image+proprio
fusion), run once per conditioning frame, folding the ``cond_steps`` time
axis into the batch dimension before the encoder forward and reshaping back
after -- the same trick ``real-stanford/diffusion_policy``'s own
``MultiImageObsEncoder`` uses for its observation history. Both branches
route through the same ``build_diffusion_net`` dispatch below; only
``cond_dim`` (an input shape, not a key name) differs between them.

``net_cls`` defaults to ``DiffusionMLP`` (this class's original, only
network) and is a straight constructor swap -- ``net_cls`` must accept
``(action_dim, horizon_steps, cond_dim, *, time_dim, kernel_init, **net_kwargs)``
and implement ``forward(x, time, cond) -> (B, horizon_steps, action_dim)``, the
same interface every consumer of ``self.net`` already relies on.
``rl_garden.networks.diffusion_unet.DiffusionUNet1D`` is the other backbone
in this repo satisfying that interface. ``mlp_dims``/``activation_fn``/
``residual_style`` are ``DiffusionMLP``-specific and only forwarded when
``net_cls is DiffusionMLP``; non-default backbones take their own
architecture knobs via ``net_kwargs``. ``kernel_init`` is always forwarded
explicitly (by ``build_diffusion_net``, below) -- do not also put it inside
``net_kwargs``, which would raise "multiple values for keyword argument".
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.obs_utils import flatten_leading_dims
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import Activation, DiffusionMLP, KernelInit
from rl_garden.policies._diffusion_process import DiffusionProcess
from rl_garden.policies.base import BasePolicy


def build_diffusion_net(
    net_cls: type[nn.Module],
    *,
    action_dim: int,
    horizon_steps: int,
    cond_dim: int,
    time_dim: int,
    kernel_init: Optional[KernelInit],
    mlp_dims: Sequence[int],
    activation_fn: Optional[Activation],
    residual_style: bool,
    net_kwargs: Optional[dict[str, Any]],
) -> nn.Module:
    """Single source of truth for the ``DiffusionMLP``-vs-other-backbone
    dispatch, shared by ``DiffusionPolicy`` and ``ConsistencyDistillBC``.
    ``kernel_init`` is forwarded explicitly to every backbone -- callers must
    not also put ``kernel_init`` inside ``net_kwargs``."""
    if net_cls is DiffusionMLP:
        return DiffusionMLP(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=cond_dim,
            time_dim=time_dim,
            mlp_dims=mlp_dims,
            activation_fn=activation_fn,
            residual_style=residual_style,
            kernel_init=kernel_init,
        )
    return net_cls(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=cond_dim,
        time_dim=time_dim,
        kernel_init=kernel_init,
        **(net_kwargs or {}),
    )


class DiffusionPolicy(DiffusionProcess, BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Box | spaces.Dict,
        action_space: spaces.Box,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
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
        net_cls: type[nn.Module] = DiffusionMLP,
        net_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "DiffusionPolicy requires a Box action space."
        self.observation_space = observation_space
        self.action_space = action_space
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps
        self.min_sampling_denoising_std = min_sampling_denoising_std

        action_dim = int(np.prod(action_space.shape))

        if isinstance(observation_space, spaces.Box):
            if features_extractor is not None:
                raise TypeError(
                    "DiffusionPolicy with Box observation space does not accept a "
                    "features_extractor (no encoder is used for state-only "
                    "observations); pass features_extractor only for Dict "
                    "observations."
                )
            obs_dim = int(np.prod(observation_space.shape))
            cond_dim = obs_dim * cond_steps
        elif isinstance(observation_space, spaces.Dict):
            if features_extractor is None:
                raise TypeError(
                    "DiffusionPolicy with Dict observation space requires a "
                    "features_extractor."
                )
            self.features_extractor = features_extractor
            cond_dim = features_extractor.features_dim * cond_steps
        else:
            raise TypeError(
                "DiffusionPolicy supports Box or Dict observation spaces, got "
                f"{type(observation_space)}"
            )

        self.net = build_diffusion_net(
            net_cls,
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=cond_dim,
            time_dim=time_dim,
            kernel_init=kernel_init,
            mlp_dims=mlp_dims,
            activation_fn=activation_fn,
            residual_style=residual_style,
            net_kwargs=net_kwargs,
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

    def _cond_from_obs_history(
        self, obs_history: Obs, stop_gradient: bool = False
    ) -> torch.Tensor:
        """Box: ``obs_history`` is already ``(B, cond_steps, obs_dim)``, returned
        unchanged (no encoder). Dict: ``obs_history`` is a Dict of tensors each
        ``(B, cond_steps, *leaf_shape)``; returns ``(B, cond_steps,
        features_dim)`` by folding ``cond_steps`` into the batch dimension
        before the features-extractor forward and reshaping back after."""
        if isinstance(self.observation_space, spaces.Dict):
            assert isinstance(obs_history, dict)
            batch = next(iter(obs_history.values())).shape[0]
            flat_obs = flatten_leading_dims(obs_history)
            flat_features = self.features_extractor.extract(
                flat_obs, stop_gradient=stop_gradient
            )
            return flat_features.reshape(batch, self.cond_steps, -1)
        return obs_history

    def loss(self, obs_history: Obs, action_chunk: torch.Tensor) -> torch.Tensor:
        """``obs_history``: (B, cond_steps, obs_dim) for Box, or a Dict of
        tensors each ``(B, cond_steps, *leaf_shape)`` for Dict. ``action_chunk``:
        (B, horizon_steps, action_dim). Epsilon-prediction MSE at random t."""
        batch = action_chunk.shape[0]
        t = torch.randint(
            0, self.denoising_steps, (batch,), device=action_chunk.device
        )
        cond = self._cond_from_obs_history(obs_history, stop_gradient=False)
        return self.p_losses(self.net, action_chunk, {"state": cond}, t)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        """``obs``: a raw ``Tensor`` for Box (single frame ``(B, obs_dim)``,
        broadcast to ``cond_steps``, or explicit history ``(B, cond_steps,
        obs_dim)``), or a Dict for Dict observations (each leaf ``(B,
        *leaf_shape)`` single frame, broadcast to ``cond_steps``, or explicit
        history ``(B, cond_steps, *leaf_shape)``). Returns the full predicted
        action chunk, ``(B, horizon_steps, action_dim)`` -- chunk
        execution/slicing is the caller's concern."""
        if isinstance(obs, dict):
            sample_key = next(iter(obs))
            leaf_ndim = len(self.observation_space[sample_key].shape)
            is_single_frame = obs[sample_key].dim() == leaf_ndim + 1
            if is_single_frame:
                obs_history = {
                    key: value.unsqueeze(1).expand(
                        -1, self.cond_steps, *([-1] * (value.dim() - 1))
                    )
                    for key, value in obs.items()
                }
            else:
                obs_history = obs
        else:
            assert isinstance(obs, torch.Tensor)
            obs_history = (
                obs if obs.dim() == 3 else obs.unsqueeze(1).expand(-1, self.cond_steps, -1)
            )
        cond = {"state": self._cond_from_obs_history(obs_history, stop_gradient=True)}
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
