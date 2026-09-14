"""``ObservationDecoder``: DreamerV3's reconstruction decoder (plan model-
based-base Part 2, section C), ported from r2dreamer's ``ConvDecoder``/
``MultiDecoder`` (``networks.py:144-189,237-310``; identical structure in
official JAX ``rssm.py:288-359``).

Mirrors ``rl_garden.encoders.dreamer_conv.DreamerConvEncoder`` in reverse:
one ``BlockLinear`` projects ``deter`` and an ``MLP`` projects ``stoch``,
both into the lowest-resolution spatial feature map, summed, then upsampled
back to the target image size (nearest-neighbour + same-pad conv, mirroring
the encoder's stride-2 maxpool stages); ``rgb_<cam>`` keys are decoded
jointly (one conv stack, sigmoid, channel-split per key) while every
``state``/``state_<name>`` key gets its own ``symlog_mse`` ``MLPHead``.
``depth_<cam>`` keys and stacked (``frame_stack > 1``) image keys are out of
scope, matching ``DreamerConvEncoder`` (``ObservationContractError``).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.dreamer_conv import (
    _DEFAULT_MULTS,
    _RMSNorm2D,
    _SamePadConv2d,
    validate_uniform_rgb_keys,
)
from rl_garden.networks.dreamer_nets import (
    BlockLinear,
    MLPHead,
    MSEDist,
    RMSNorm,
    SymlogMSEDist,
    dreamer_weight_init_,
)
from rl_garden.observations.schema import ObservationSchema


class _ImageDecoder(nn.Module):
    """``deter``/``stoch`` -> one multi-channel image (r2dreamer
    ``ConvDecoder``, ``networks.py:237-310``): ``BlockLinear(deter)`` +
    ``MLP(stoch)`` project into the lowest-resolution feature map, summed,
    then ``len(mults) - 1`` stages of ``Upsample(nearest, 2x) ->
    Conv2dSamePad -> RMSNorm -> act`` followed by one final upsample+conv
    (no norm/act -- matches upstream) back to the target ``(out_ch, H, W)``,
    then ``sigmoid`` (applied by the caller, ``MSEDist`` wraps the raw
    ``[0, 1]`` mode)."""

    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        out_ch: int,
        image_size: tuple[int, int],
        *,
        depth: int,
        units: int,
        mults: Sequence[int] = _DEFAULT_MULTS,
        kernel_size: int = 5,
        bspace: int = 8,
        act: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        depths = tuple(depth * m for m in mults)
        factor = 2 ** len(depths)
        h, w = image_size
        if h % factor != 0 or w % factor != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by 2**len(mults)={factor}."
            )
        min_h, min_w, min_c = h // factor, w // factor, depths[-1]
        self._min_shape = (min_h, min_w, min_c)
        self._bspace = bspace

        self.sp0 = BlockLinear(deter_dim, min_h * min_w * min_c, bspace)
        self.sp1 = nn.Sequential(nn.Linear(stoch_dim, 2 * units), RMSNorm(2 * units), act())
        self.sp2 = nn.Linear(2 * units, min_h * min_w * min_c)
        self.sp_norm = nn.Sequential(RMSNorm(min_c), act())

        layers: list[nn.Module] = []
        in_ch = min_c
        for stage_depth in reversed(depths[:-1]):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(_SamePadConv2d(in_ch, stage_depth, kernel_size, stride=1, bias=True))
            layers.append(_RMSNorm2D(stage_depth))
            layers.append(act())
            in_ch = stage_depth
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(_SamePadConv2d(in_ch, out_ch, kernel_size, stride=1, bias=True))
        self.layers = nn.Sequential(*layers)

        self.sp1.apply(dreamer_weight_init_)
        self.sp2.apply(dreamer_weight_init_)
        self.layers.apply(dreamer_weight_init_)
        # BlockLinear self-initializes at construction (see its docstring).

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        min_h, min_w, min_c = self._min_shape
        x0 = self.sp0(deter)  # (N, min_h*min_w*min_c), block-diagonal in min_c
        x0 = x0.reshape(-1, self._bspace, min_h, min_w, min_c // self._bspace)
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, min_h, min_w, min_c)

        x1 = self.sp2(self.sp1(stoch)).reshape(-1, min_h, min_w, min_c)

        x = self.sp_norm(x0 + x1)  # (N, H, W, C), norm over channel (last) dim
        x = x.permute(0, 3, 1, 2)  # -> NCHW
        x = self.layers(x)
        return x.permute(0, 2, 3, 1)  # -> NHWC


class ObservationDecoder(nn.Module):
    """Per-observation-key reconstruction heads from an RSSM state.

    ``decode(deter, stoch) -> dict[key, dist]`` where each ``dist`` exposes
    ``.log_prob(target)``/``.mode`` (``MSEDist`` for every ``rgb_<cam>`` key
    -- r2dreamer's own default, NOT ``symlog_mse``, see
    ``dreamer-code-survey.md`` section 2 -- ``SymlogMSEDist`` via
    ``MLPHead(output="symlog_mse")`` for every ``state``/``state_<name>``
    key). Any leading ``(...,)`` batch shape on ``deter``/``stoch`` is
    supported (flattened for the conv stack, restored after).
    """

    def __init__(
        self,
        schema: ObservationSchema,
        deter_dim: int,
        stoch_dim: int,
        *,
        depth: int = 16,
        units: int = 256,
        state_layers: int = 3,
        mults: Sequence[int] = _DEFAULT_MULTS,
        kernel_size: int = 5,
        bspace: int = 8,
    ) -> None:
        super().__init__()
        rgb_keys, image_hw = validate_uniform_rgb_keys(schema)
        state_keys = schema.state_keys

        if not rgb_keys and not state_keys:
            raise ValueError(
                f"ObservationDecoder has nothing to decode: schema has no rgb_<cam> keys "
                f"and no state keys ({schema.keys!r})."
            )

        self.rgb_keys = rgb_keys
        self.state_keys = tuple(state_keys)
        self._rgb_channel_splits = (3,) * len(rgb_keys)

        self.image_decoder: Optional[_ImageDecoder] = None
        if rgb_keys:
            self.image_decoder = _ImageDecoder(
                deter_dim,
                stoch_dim,
                out_ch=3 * len(rgb_keys),
                image_size=image_hw,
                depth=depth,
                units=units,
                mults=mults,
                kernel_size=kernel_size,
                bspace=bspace,
            )

        self.state_head: Optional[MLPHead] = None
        self._state_dims: tuple[int, ...] = ()
        if state_keys:
            # `observation_space` isn't available here (schema only) -- state
            # key dims come from the schema entries themselves.
            self._state_dims = tuple(math.prod(schema.entries[k].shape) for k in state_keys)
            self.state_head = MLPHead(
                deter_dim + stoch_dim,
                sum(self._state_dims),
                output="symlog_mse",
                layers=state_layers,
                units=units,
                outscale=1.0,
            )

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> dict[str, object]:
        dists: dict[str, object] = {}
        if self.image_decoder is not None:
            leading = deter.shape[:-1]
            flat_deter = deter.reshape(-1, deter.shape[-1])
            flat_stoch = stoch.reshape(-1, stoch.shape[-1])
            image = self.image_decoder(flat_deter, flat_stoch)  # (N, H, W, C_total)
            image = torch.sigmoid(image)
            h, w, c_total = image.shape[-3:]
            image = image.reshape(*leading, h, w, c_total)
            splits = torch.split(image, self._rgb_channel_splits, dim=-1)
            dists.update({key: MSEDist(value) for key, value in zip(self.rgb_keys, splits)})
        if self.state_head is not None:
            feat = torch.cat([deter, stoch], dim=-1)
            # Raw symlog-space linear output (not MLPHead.forward()'s
            # SymlogMSEDist wrapper) so the concatenated per-key output can be
            # split before wrapping -- log_prob is per-element and additive,
            # so a per-key SymlogMSEDist over its own slice gives exactly the
            # same per-key log_prob the combined head would.
            raw = self.state_head.last(self.state_head.mlp(feat))
            offset = 0
            for key, dim in zip(self.state_keys, self._state_dims):
                dists[key] = SymlogMSEDist(raw[..., offset : offset + dim])
                offset += dim
        return dists
