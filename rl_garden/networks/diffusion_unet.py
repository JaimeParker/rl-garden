"""1D temporal-convolution U-Net denoiser, an alternative backbone to
``DiffusionMLP`` for action-chunk diffusion policies.

Faithful port of ``3rd_party/diffusion_policy``'s ``ConditionalUnet1D``/
``ConditionalResidualBlock1D`` (``diffusion_policy/model/diffusion/
conditional_unet1d.py``) and ``Conv1dBlock``/``Downsample1d``/``Upsample1d``
(``conditional_unet1d.py``'s sibling ``conv1d_components.py``), verified
against source directly (MIT license, Columbia Artificial Intelligence and
Robotics Lab -- confirmed by reading ``3rd_party/diffusion_policy/LICENSE``).
Unlike ``DiffusionMLP``, which flattens the ``(horizon_steps, action_dim)``
action chunk into one MLP input, this treats it as a length-``horizon_steps``
sequence over ``action_dim`` channels and denoises it with 1D convolutions
and FiLM-style (Perez et al. 2017) conditioning at every residual block --
the architecture the original Diffusion Policy paper (Chi et al. 2023) uses
for its state-based/low-dim variant.

Only global conditioning is ported (``local_cond`` in the reference, used
for a receding-horizon "already observed" prefix, is never exercised by any
caller in this repo and is dropped rather than carried as dead code).

``forward(x, time, cond) -> (B, horizon_steps, action_dim)`` matches
``DiffusionMLP.forward``'s signature exactly, so this is a drop-in swap via
``DiffusionPolicy(net_cls=DiffusionUNet1D, net_kwargs={...})``
(``rl_garden/policies/diffusion_policy.py``).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from rl_garden.networks.mlp import KernelInit, _apply_kernel_init


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=x.device) * -scale)
        emb = x[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class _Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                _Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        # FiLM modulation (https://arxiv.org/abs/1709.07871): predicts a
        # per-channel scale+bias (cond_predict_scale=True) or bias only.
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            scale, bias = embed.chunk(2, dim=1)
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class DiffusionUNet1D(nn.Module):
    """``eps_theta(x_t, t, cond) -> (B, horizon_steps, action_dim)``, same
    interface as ``DiffusionMLP``. ``horizon_steps`` must be a multiple of
    ``2 ** (len(down_dims) - 1)`` -- each ``down_dims`` transition after the
    first halves the temporal length (mirrors the reference's own
    architecture, which has no padding/cropping fallback for a
    non-power-of-2-compatible horizon)."""

    def __init__(
        self,
        action_dim: int,
        horizon_steps: int,
        cond_dim: int,
        *,
        time_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        num_downsamples = len(down_dims) - 1
        if horizon_steps % (2**num_downsamples) != 0:
            raise ValueError(
                f"horizon_steps ({horizon_steps}) must be a multiple of "
                f"2**(len(down_dims)-1) ({2 ** num_downsamples}) for down_dims={tuple(down_dims)}."
            )
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.time_dim = time_dim

        self.diffusion_step_encoder = nn.Sequential(
            _SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        global_cond_dim = time_dim + cond_dim

        all_dims = [action_dim] + list(down_dims)
        start_dim = down_dims[0]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        def block(dim_in: int, dim_out: int) -> _ConditionalResidualBlock1D:
            return _ConditionalResidualBlock1D(
                dim_in,
                dim_out,
                cond_dim=global_cond_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([block(mid_dim, mid_dim), block(mid_dim, mid_dim)])

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        block(dim_in, dim_out),
                        block(dim_out, dim_out),
                        nn.Identity() if is_last else _Downsample1d(dim_out),
                    ]
                )
            )

        # Every up-stage upsamples (never `Identity`, unlike down_modules):
        # the reference computes `is_last` for this loop by comparing its
        # index (which only ranges over `len(in_out) - 1` up-stages) against
        # a threshold sized for the *down* loop's longer index range
        # (`len(in_out)`), so that threshold is structurally never reached
        # here. This is required, not a bug worth "fixing" -- down_modules
        # applies exactly `len(down_dims) - 1` downsamples (every stage
        # except its last), so up_modules must apply the same count of
        # upsamples across all of its `len(down_dims) - 1` stages for the
        # output length to match the input length.
        self.up_modules = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList([block(dim_out * 2, dim_in), block(dim_in, dim_in), _Upsample1d(dim_in)])
            )

        self.final_conv = nn.Sequential(
            _Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, action_dim, 1),
        )
        _apply_kernel_init(self, kernel_init)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: dict) -> torch.Tensor:
        batch = x.shape[0]
        state = cond["state"].reshape(batch, -1)
        time_emb = self.diffusion_step_encoder(time.reshape(batch))
        global_feature = torch.cat([time_emb, state], dim=-1)

        h = x.transpose(1, 2)  # (B, horizon, action_dim) -> (B, action_dim, horizon)
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            h = resnet(h, global_feature)
            h = resnet2(h, global_feature)
            skips.append(h)
            h = downsample(h)

        for mid_module in self.mid_modules:
            h = mid_module(h, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            h = torch.cat([h, skips.pop()], dim=1)
            h = resnet(h, global_feature)
            h = resnet2(h, global_feature)
            h = upsample(h)

        out = self.final_conv(h)
        return out.transpose(1, 2)
