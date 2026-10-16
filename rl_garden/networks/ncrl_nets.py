"""NCRL's own network backbone (``RSSM(backbone="ncrl")``), ported from the
official ``3rd_party/ncrl/src/dreamer/{nets,rssm}.py``.

Differences from the DreamerV3 backbone in ``rl_garden.networks.dreamer_nets``
/ ``rl_garden.encoders.dreamer_conv`` / ``rl_garden.world_models.decoder``:

- ``LayerNorm`` everywhere (``eps=1e-3`` for the conv channel norms, torch's
  default elsewhere), not RMSNorm.
- torch's default initialization everywhere (no truncated normal, no output
  rescale). The block-diagonal layer keeps the official ``(blocks, in/blocks,
  out/blocks)`` weight layout and its ``kaiming_uniform`` reset.
- Encoder: 4 stride-2 kernel-4 convs without padding (64 -> 31 -> 14 -> 6 ->
  2), channels ``depth * 2**i``. Decoder: a 1x1 feature map upsampled by 4
  stride-2 transposed convs with kernels ``(5, 5, 6, 6)`` (1 -> 5 -> 13 -> 30
  -> 64), linear output (no sigmoid).
- Block-GRU: no ``a / max(|a|, 1)`` action normalization, and the
  reset/candidate/update gates are split from the flat output (official
  ``rssm.py:49-50``) rather than per block.
- Heads read ``deter`` through a block-diagonal branch and ``stoch`` through a
  linear branch, summed, then the MLP (official ``MLPs``).

All modules accept any number of leading batch dims.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.dreamer_conv import validate_uniform_rgb_keys
from rl_garden.networks.dreamer_nets import MLPHead, MSEDist
from rl_garden.observations.schema import ObservationContractError, ObservationSchema

ENCODER_KERNELS: tuple[int, ...] = (4, 4, 4, 4)
DECODER_KERNELS: tuple[int, ...] = (5, 5, 6, 6)
NCRL_IMAGE_SIZE = (64, 64)
CHANNEL_NORM_EPS = 1e-3


class NCRLBlockLinear(nn.Module):
    """Official ``nets.BlockLinear`` (``nets.py:56-100``): ``blocks`` independent
    ``(in / blocks) -> (out / blocks)`` maps, weight ``(blocks, in/blocks,
    out/blocks)`` with ``nn.Linear``'s default ``kaiming_uniform`` reset (fan-in
    taken from that layout, as in the official code)."""

    def __init__(self, in_features: int, out_features: int, blocks: int) -> None:
        super().__init__()
        if in_features % blocks or out_features % blocks:
            raise ValueError(f"in/out features {in_features}/{out_features} must be divisible by blocks={blocks}.")
        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.weight = nn.Parameter(torch.empty(blocks, in_features // blocks, out_features // blocks))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, self.blocks, self.in_features // self.blocks)
        x = torch.einsum("...ki,kio->...ko", x, self.weight)
        return x.reshape(*batch_shape, self.out_features) + self.bias


class ChannelNorm(nn.LayerNorm):
    """LayerNorm over the channel dim of an NCHW tensor (official
    ``ChannelNorm``, ``nets.py:103-121``)."""

    def __init__(self, channels: int, eps: float = CHANNEL_NORM_EPS) -> None:
        super().__init__(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _image_only_rgb_keys(schema: ObservationSchema, owner: str) -> tuple[tuple[str, ...], tuple[int, int]]:
    rgb_keys, image_hw = validate_uniform_rgb_keys(schema)
    if schema.state_keys or not rgb_keys:
        raise ObservationContractError(
            f"{owner} is image-only (official NCRL): needs rgb_<cam> keys and no state keys, got {schema.keys!r}."
        )
    return rgb_keys, image_hw


class NCRLConvEncoder(BaseFeaturesExtractor):
    """Official ``nets.Encoder`` (``nets.py:124-158``): every ``rgb_<cam>`` key
    channel-concatenated, ``/255 - 0.5``, then ``len(kernels)`` stages of
    ``Conv2d(k, stride 2, no padding) -> ChannelNorm -> act`` with channels
    ``depth * 2**i``, flattened (C, H, W order). ``features_dim`` comes from a
    dummy forward."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        schema: ObservationSchema,
        *,
        depth: int = 48,
        kernels: Sequence[int] = ENCODER_KERNELS,
        act: type[nn.Module] = nn.SiLU,
    ) -> None:
        rgb_keys, (h, w) = _image_only_rgb_keys(schema, type(self).__name__)
        in_ch = 3 * len(rgb_keys)
        layers: list[nn.Module] = []
        for i, kernel in enumerate(kernels):
            out_ch = depth * 2**i
            layers += [nn.Conv2d(in_ch, out_ch, kernel, stride=2), ChannelNorm(out_ch), act()]
            in_ch = out_ch
        cnn = nn.Sequential(*layers)
        with torch.no_grad():
            features_dim = int(cnn(torch.zeros(1, 3 * len(rgb_keys), h, w)).numel())
        super().__init__(observation_space, features_dim)
        self.cnn = cnn
        self.rgb_keys = rgb_keys

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        images = torch.cat([obs[k] for k in self.rgb_keys], dim=-1)  # (..., H, W, C)
        leading = images.shape[:-3]
        x = images.reshape(-1, *images.shape[-3:]).float() / 255.0 - 0.5
        x = self.cnn(x.permute(0, 3, 1, 2))
        return x.reshape(*leading, self.features_dim)


class NCRLConvDecoder(nn.Module):
    """Official ``nets.Decoder`` (``nets.py:161-216``). ``deter`` goes through
    ``NCRLBlockLinear(deter -> 2 units) -> LayerNorm -> act -> Linear(-> 32 depth)``,
    ``stoch`` through ``Linear(-> 2 units) -> LayerNorm -> act -> Linear(-> 32 depth)``;
    the sum is a ``(32 depth, 1, 1)`` map upsampled by transposed convs (channel
    norm + act on all but the last, linear output).

    Same interface as ``ObservationDecoder``: ``forward(deter, stoch_flat)`` ->
    ``{rgb_key: MSEDist}`` in NHWC. The network output targets ``obs / 255 -
    0.5`` (official); ``MSEDist`` wraps ``output + 0.5`` so the RSSM's ``obs /
    255`` target gives exactly the official ``MSE.sum(C, H, W)``."""

    def __init__(
        self,
        schema: ObservationSchema,
        deter_dim: int,
        stoch_dim: int,
        *,
        depth: int = 48,
        units: int = 1024,
        blocks: int = 8,
        kernels: Sequence[int] = DECODER_KERNELS,
        act: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        rgb_keys, image_hw = _image_only_rgb_keys(schema, type(self).__name__)
        feature_dim = 32 * depth
        self.rgb_keys = rgb_keys
        self._feature_dim = feature_dim
        self.fc_deter = nn.Sequential(
            NCRLBlockLinear(deter_dim, 2 * units, blocks),
            nn.LayerNorm(2 * units),
            act(),
            nn.Linear(2 * units, feature_dim),
        )
        self.fc_stoch = nn.Sequential(
            nn.Linear(stoch_dim, 2 * units), nn.LayerNorm(2 * units), act(), nn.Linear(2 * units, feature_dim)
        )
        layers: list[nn.Module] = []
        in_ch, num = feature_dim, len(kernels)
        size = 1
        for i, kernel in enumerate(kernels):
            last = i == num - 1
            out_ch = 3 * len(rgb_keys) if last else depth * 2 ** (num - i - 2)
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=2))
            if not last:
                layers += [ChannelNorm(out_ch), act()]
            in_ch = out_ch
            size = (size - 1) * 2 + kernel
        if (size, size) != tuple(image_hw):
            raise ObservationContractError(
                f"NCRLConvDecoder with kernels {tuple(kernels)} produces {size}x{size} images, "
                f"but the rgb keys are {tuple(image_hw)}."
            )
        self.cnn = nn.Sequential(*layers)

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> dict[str, Any]:
        leading = deter.shape[:-1]
        x = self.fc_deter(deter.reshape(-1, deter.shape[-1])) + self.fc_stoch(stoch.reshape(-1, stoch.shape[-1]))
        x = self.cnn(x.reshape(-1, self._feature_dim, 1, 1)).permute(0, 2, 3, 1)  # NHWC
        image = x.reshape(*leading, *x.shape[1:]) + 0.5
        splits = torch.split(image, 3, dim=-1)
        return {key: MSEDist(value) for key, value in zip(self.rgb_keys, splits)}


class NCRLBlockGRU(nn.Module):
    """Official ``BlockGRUCell`` (``rssm.py:17-55``): ``deter``/``stoch``/
    ``action`` each ``Linear -> LayerNorm -> act`` to ``hidden``, broadcast over
    ``blocks`` and concatenated with the per-block ``deter``, ``dyn_layers``
    ``NCRLBlockLinear -> LayerNorm -> act`` layers, then one ``NCRLBlockLinear``
    to ``3 * deter`` split (flat) into reset/candidate/update gates. The action
    enters unnormalized. ``forward(stoch_flat, deter, action)`` matches the
    DreamerV3 core's signature."""

    def __init__(
        self,
        deter: int,
        flat_stoch: int,
        action_dim: int,
        hidden: int,
        blocks: int,
        dyn_layers: int = 1,
        act: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.in_deter = nn.Sequential(nn.Linear(deter, hidden), nn.LayerNorm(hidden), act())
        self.in_stoch = nn.Sequential(nn.Linear(flat_stoch, hidden), nn.LayerNorm(hidden), act())
        self.in_action = nn.Sequential(nn.Linear(action_dim, hidden), nn.LayerNorm(hidden), act())
        layers: list[nn.Module] = []
        in_ch = deter + 3 * hidden * blocks
        for _ in range(dyn_layers):
            layers += [NCRLBlockLinear(in_ch, deter, blocks), nn.LayerNorm(deter), act()]
            in_ch = deter
        self.dyn_layers = nn.Sequential(*layers)
        self.gru = NCRLBlockLinear(in_ch, 3 * deter, blocks)

    def forward(self, stoch_flat: torch.Tensor, deter: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.in_deter(deter), self.in_stoch(stoch_flat), self.in_action(action)], dim=-1)
        x = x.unsqueeze(-2).expand(*x.shape[:-1], self.blocks, x.shape[-1])
        grouped_deter = deter.reshape(*deter.shape[:-1], self.blocks, -1)
        x = torch.cat([grouped_deter, x], dim=-1).reshape(*deter.shape[:-1], -1)
        reset, cand, update = torch.chunk(self.gru(self.dyn_layers(x)), 3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter


def ncrl_logits_mlp(
    inp_dim: int, hidden: int, layers: int, out_dim: int, act: type[nn.Module] = nn.SiLU
) -> nn.Sequential:
    """Prior/posterior logits head (official ``rssm.py:110-124``): ``layers`` x
    ``Linear -> LayerNorm -> act``, then ``Linear(-> out_dim)``."""
    modules: list[nn.Module] = []
    for _ in range(layers):
        modules += [nn.Linear(inp_dim, hidden), nn.LayerNorm(hidden), act()]
        inp_dim = hidden
    modules.append(nn.Linear(inp_dim, out_dim))
    return nn.Sequential(*modules)


class _NCRLHeadTrunk(nn.Module):
    """Official ``MLPs`` trunk (``nets.py:219-290``, ``blocks > 0``): splits the
    RSSM feature ``cat[deter, stoch_flat]`` at ``deter_dim``; ``deter`` ->
    ``NCRLBlockLinear(-> 2 units) -> LayerNorm -> act -> Linear(-> units)``,
    ``stoch`` -> ``Linear(-> units)``, summed, then ``layers`` x ``Linear ->
    LayerNorm -> act``."""

    def __init__(
        self, deter_dim: int, stoch_dim: int, layers: int, units: int, blocks: int, act: type[nn.Module]
    ) -> None:
        super().__init__()
        self.deter_dim = deter_dim
        self.fc_deter = nn.Sequential(
            NCRLBlockLinear(deter_dim, 2 * units, blocks), nn.LayerNorm(2 * units), act(), nn.Linear(2 * units, units)
        )
        self.fc_stoch = nn.Linear(stoch_dim, units)
        modules: list[nn.Module] = []
        for _ in range(layers):
            modules += [nn.Linear(units, units), nn.LayerNorm(units), act()]
        self.layers = nn.Sequential(*modules)
        self.out_dim = units

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        deter, stoch = feat[..., : self.deter_dim], feat[..., self.deter_dim :]
        return self.layers(self.fc_deter(deter) + self.fc_stoch(stoch))


class NCRLHead(MLPHead):
    """Reward/critic/actor head of the NCRL backbone: an ``MLPHead`` (same
    output kinds, ``.mlp``/``.last`` surface and distributions) whose trunk is
    the official ``MLPs`` (:class:`_NCRLHeadTrunk`) and whose output layer is a
    default-initialized ``nn.Linear`` (no outscale). Consumes the RSSM feature
    ``cat[deter, stoch_flat]``."""

    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        out_dim: int,
        *,
        output: str,
        layers: int,
        units: int,
        blocks: int = 8,
        act: type[nn.Module] = nn.SiLU,
        **kwargs: Any,
    ) -> None:
        super().__init__(deter_dim + stoch_dim, out_dim, output=output, layers=0, units=units, act=act, **kwargs)
        last_dim = self.last.out_features
        self.mlp = _NCRLHeadTrunk(deter_dim, stoch_dim, layers, units, blocks, act)
        self.last = nn.Linear(units, last_dim)
