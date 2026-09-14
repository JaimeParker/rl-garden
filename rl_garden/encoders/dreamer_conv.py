"""``DreamerConvEncoder``: DreamerV3's observation encoder (plan model-based-
base Part 2, section C), ported from r2dreamer's ``ConvEncoder``/``MLP``
(``networks.py:192-234,313-336``; identical structure in official JAX
``rssm.py:210-250``).

Two extractors share the conv-stack builder below (``_build_dreamer_cnn``):

- ``DreamerConvEncoder``, the one described by the plan -- unlike every
  other entry in ``rl_garden.encoders.registry`` (each an image-only factory
  plugged into ``CombinedExtractor`` alongside a separate ``ProprioEncoder``
  for state keys), this consumes the WHOLE Dict observation itself (images
  AND state keys together, into one embedding): DreamerV3's world model owns
  representation learning end to end (encoder, RSSM, decoder all trained by
  one loss), so ``rl_garden.world_models.rssm.RSSM`` constructs this
  directly as its own ``self.encoder``, never through
  ``EncoderConfig.image_encoder_factory()``. Its ``forward()`` takes the raw
  obs Dict (uint8 HWC images) and does its own ``/255 - 0.5`` normalization.
- ``_DreamerConvImageOnly``, registered as the ``"dreamer_conv"`` backbone
  (``registry.py``, ``EncoderConfig.backbone``) so any algorithm can also
  select this CNN for its image branch through the ordinary
  ``CombinedExtractor`` path. That path's contract is fundamentally
  different from ``DreamerConvEncoder``'s -- a single already-normalized
  ``[0, 1]`` channels-first tensor in, not a raw-uint8 Dict -- so it is a
  separate, tiny adapter class rather than a second mode bolted onto
  ``DreamerConvEncoder.forward``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks.dreamer_nets import MLP, dreamer_weight_init_
from rl_garden.observations.schema import Modality, ObservationContractError, ObservationSchema

_DEFAULT_MULTS: tuple[int, ...] = (2, 3, 4, 4)


class _RMSNorm2D(nn.RMSNorm):
    """RMSNorm applied over the channel dim of an NCHW tensor (r2dreamer
    ``RMSNorm2D``, ``networks.py:88-96``): permute to channels-last, norm,
    permute back."""

    def __init__(self, channels: int, eps: float = 1e-4) -> None:
        super().__init__(channels, eps=eps, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _SamePadConv2d(nn.Conv2d):
    """``Conv2d`` emulating TensorFlow 'SAME' padding (r2dreamer
    ``Conv2dSamePad``, ``networks.py:59-85``) -- stride is always 1 here
    (downsampling is a separate ``MaxPool2d``), so padding only needs to
    keep H/W unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2:]
        pad_h = max((ih - 1) * self.stride[0] + (self.kernel_size[0] - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((iw - 1) * self.stride[1] + (self.kernel_size[1] - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h or pad_w:
            x = nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return self._conv_forward(x, self.weight, self.bias)


def validate_uniform_rgb_keys(
    schema: ObservationSchema,
) -> tuple[tuple[str, ...], Optional[tuple[int, int]]]:
    """Shared by ``DreamerConvEncoder`` and ``rl_garden.world_models.decoder
    .ObservationDecoder``: collects ``rgb_<cam>`` keys, rejects
    ``depth_<cam>`` keys and ``frame_stack > 1``, and asserts every ``rgb``
    key shares one ``(H, W)``. Returns ``(rgb_keys, image_hw)`` --
    ``image_hw`` is ``None`` when there are no rgb keys."""
    rgb_keys = [k for k in schema.keys if schema.entries[k].modality == Modality.RGB]
    depth_keys = [k for k in schema.keys if schema.entries[k].modality == Modality.DEPTH]
    if depth_keys:
        raise ObservationContractError(f"depth_<cam> keys are not supported, got {depth_keys!r}.")
    for key in rgb_keys:
        if schema.entries[key].stacked:
            raise ObservationContractError(
                f"frame_stack > 1 is not supported (key {key!r} is stacked) -- Dreamer "
                "carries history in its recurrent state, not a stacked-frame channel axis."
            )
    image_hw: Optional[tuple[int, int]] = None
    for key in rgb_keys:
        hw = schema.image_size(key)
        if image_hw is None:
            image_hw = hw
        elif hw != image_hw:
            raise ObservationContractError(
                f"every rgb_<cam> key must share one (H, W); {key!r} is {hw} but a "
                f"previous key was {image_hw}."
            )
    return tuple(rgb_keys), image_hw


def _build_dreamer_cnn(
    in_ch: int, h: int, w: int, depth: int, mults: Sequence[int], kernel_size: int
) -> tuple[nn.Sequential, int]:
    """Shared conv-stack builder (r2dreamer ``ConvEncoder``,
    ``networks.py:192-234``): ``len(mults)`` stages of
    ``Conv2dSamePad(kernel_size) -> MaxPool2d(2,2) -> RMSNorm -> SiLU``,
    channel depth ``depth * mults[i]``. Returns ``(module, flattened_out_dim)``.
    Shared by ``DreamerConvEncoder`` (whole-Dict, raw uint8 HWC input) and
    ``_DreamerConvImageOnly`` (the ``registry.py`` per-image adapter, already-
    normalized CHW input) -- the two differ only in what happens to the
    tensor BEFORE/AFTER this stack, not in the stack itself.
    """
    depths = tuple(depth * m for m in mults)
    layers: list[nn.Module] = []
    for stage_depth in depths:
        layers.append(_SamePadConv2d(in_ch, stage_depth, kernel_size, stride=1, bias=True))
        layers.append(nn.MaxPool2d(2, 2))
        layers.append(_RMSNorm2D(stage_depth))
        layers.append(nn.SiLU())
        in_ch = stage_depth
        h, w = h // 2, w // 2
    cnn = nn.Sequential(*layers)
    cnn.apply(dreamer_weight_init_)
    return cnn, depths[-1] * h * w


class _DreamerConvImageOnly(BaseFeaturesExtractor):
    """``rl_garden.encoders.registry``'s per-image adapter: plugged into
    ``CombinedExtractor`` exactly like ``PlainConv``/a ResNet -- input is a
    single ``(B, C, H, W)`` tensor already normalized to ``[0, 1]`` and
    channels-first (``CombinedExtractor._prepare_image``/``_to_nchw``), NOT
    the raw uint8 HWC Dict ``DreamerConvEncoder.forward`` expects. Applies
    only the extra ``-0.5`` shift Dreamer's own convention wants on top of
    that existing ``[0, 1]`` normalization, then the same conv stack.
    """

    def __init__(
        self,
        image_space: spaces.Box,
        *,
        depth: int = 16,
        mults: Sequence[int] = _DEFAULT_MULTS,
        kernel_size: int = 5,
    ) -> None:
        c, h, w = image_space.shape
        cnn, out_dim = _build_dreamer_cnn(c, h, w, depth, mults, kernel_size)
        super().__init__(image_space, out_dim)
        self.cnn = cnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x - 0.5).flatten(1)


class DreamerConvEncoder(BaseFeaturesExtractor):
    """Multi-``rgb_<cam>``-key CNN + symlog state MLP, concatenated into one
    embedding.

    - Every ``rgb_<cam>`` key is channel-concatenated into one multi-channel
      image (all cameras must share one ``(H, W)``; ``ObservationContractError``
      otherwise) and run through ``len(mults)`` conv stages
      (``Conv2dSamePad(kernel_size) -> MaxPool2d(2,2) -> RMSNorm -> SiLU``,
      channel depth ``depth * mults[i]``), then flattened.
    - Any ``depth_<cam>`` key is out of scope (``ObservationContractError``,
      mirrors ``ObservationDecoder``).
    - A stacked (``frame_stack > 1``) image key is out of scope
      (``ObservationContractError``): Dreamer is recurrent, it carries
      history in ``deter``/``stoch``, not in a stacked-frame channel axis.
    - Every ``state``/``state_<name>`` key is concatenated (schema order),
      ``symlog``-transformed, and run through an ``MLP`` (``state_layers``
      layers of width ``units``).

    Any number of leading batch dims are supported (``(B, ...)`` at rollout
    time, ``(T, B, ...)`` for a training window) -- flattened to one batch
    dim for the conv stack, then restored.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        schema: ObservationSchema,
        *,
        depth: int = 16,
        units: int = 256,
        state_layers: int = 3,
        mults: Sequence[int] = _DEFAULT_MULTS,
        kernel_size: int = 5,
    ) -> None:
        rgb_keys, image_hw = validate_uniform_rgb_keys(schema)
        state_keys = schema.state_keys
        features_dim = 0

        cnn: Optional[nn.Sequential] = None
        cnn_out_dim = 0
        if rgb_keys:
            in_ch = 3 * len(rgb_keys)
            h, w = image_hw
            cnn, cnn_out_dim = _build_dreamer_cnn(in_ch, h, w, depth, mults, kernel_size)
            features_dim += cnn_out_dim

        state_mlp: Optional[MLP] = None
        if state_keys:
            state_dim = sum(int(observation_space.spaces[k].shape[0]) for k in state_keys)
            state_mlp = MLP(state_dim, units, state_layers, symlog_inputs=True)
            state_mlp.apply(dreamer_weight_init_)
            features_dim += state_mlp.out_dim

        if features_dim == 0:
            raise ObservationContractError(
                f"DreamerConvEncoder produced 0-dim output: schema has no rgb_<cam> keys "
                f"and no state keys ({schema.keys!r})."
            )

        # nn.Module submodule assignment requires nn.Module.__init__() (run
        # inside BaseFeaturesExtractor.__init__ below) to have already run --
        # everything above is built into locals for exactly this reason.
        super().__init__(observation_space, features_dim)
        self.cnn = cnn
        self._cnn_out_dim = cnn_out_dim
        self.state_mlp = state_mlp
        self.rgb_keys = rgb_keys
        self.state_keys = tuple(state_keys)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.cnn is not None:
            images = torch.cat([obs[k] for k in self.rgb_keys], dim=-1)  # (..., H, W, C_total)
            leading = images.shape[:-3]
            h, w, c = images.shape[-3:]
            x = images.reshape(-1, h, w, c).float() / 255.0 - 0.5
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
            x = self.cnn(x)
            x = x.reshape(x.shape[0], -1)
            parts.append(x.reshape(*leading, self._cnn_out_dim))
        if self.state_mlp is not None:
            state = torch.cat([obs[k] for k in self.state_keys], dim=-1)
            parts.append(self.state_mlp(state))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
