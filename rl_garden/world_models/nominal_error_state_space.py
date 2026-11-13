"""``NominalErrorStateSpaceWorldModel``: a ``StateSpaceWorldModel`` whose
prediction target and error metric are separate, as in the official SuperMimic
implementation (github.com/HybridRobotics/supermimic, ``WorldModelLearner``).

The dynamics MLP predicts the codec's raw ``increment`` (no output
normalization, ``normalize_output = False``); the analytic ``integrate`` turns
it into the next state. The k-step autoregressive loss does not score the
increment: it scores the codec's ``LossBlockStateCodec.loss_blocks`` of the
predicted against the real states, each block weighted by Bryson's rule
``w_g = 1 / nominal_g**2`` and by its width ``W_g``::

    diff_g = loss_blocks(s_hat)[g] - loss_blocks(s_real)[g]   # (k, B, W_g)
    mse_g  = mean(diff_g ** 2)                                # over k, B, W_g
    loss   = sum_g mse_g * w_g * W_g / sum_g W_g

So a codec can predict velocity-level deltas while the loss is measured on the
integrated pose. Channels outside every block do not enter the loss, and the
codec's ``error_floor`` is not used. Unlike the base class, the state term is
averaged (not summed) over the ``k`` steps. Every other part of
``model_loss`` (rollout, heads, posterior) is the base class's.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from rl_garden.world_models.state_space import StateSpaceWorldModel


class NominalErrorStateSpaceWorldModel(StateSpaceWorldModel):
    """Raw-increment dynamics scored in the codec's loss blocks with fixed
    nominal errors. Needs a ``LossBlockStateCodec``; takes the same keyword
    arguments as ``StateSpaceWorldModel``.

    Construction raises ``ValueError`` unless the codec has ``loss_blocks``
    and finite, positive ``nominal_errors`` whose keys equal the block names.
    Block names and widths ``W_g`` come from one probe,
    ``codec.loss_blocks(torch.zeros(1, state_dim))`` on the CPU, so
    ``loss_blocks`` must accept a CPU tensor.
    """

    normalize_output = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        codec = self.codec
        nominal_errors = getattr(codec, "nominal_errors", None)
        if nominal_errors is None or not callable(getattr(codec, "loss_blocks", None)):
            raise ValueError(
                f"{type(self).__name__} needs a LossBlockStateCodec (with nominal_errors and "
                f"loss_blocks); got {type(codec).__name__}."
            )
        if not nominal_errors:
            raise ValueError("codec.nominal_errors must name at least one loss block.")
        for name, nominal in nominal_errors.items():
            if not (math.isfinite(float(nominal)) and float(nominal) > 0):
                raise ValueError(f"nominal error of loss block {name!r} must be finite and > 0, got {nominal}.")
        # Probe once on CPU for the block names and widths.
        with torch.no_grad():
            probe = codec.loss_blocks(torch.zeros(1, self.state_dim))
        if set(probe) != set(nominal_errors):
            raise ValueError(
                f"codec.loss_blocks keys {sorted(probe)} must equal codec.nominal_errors keys "
                f"{sorted(nominal_errors)}."
            )
        self.block_names: tuple[str, ...] = tuple(nominal_errors)
        self.block_weights: dict[str, float] = {name: 1.0 / float(nominal_errors[name]) ** 2 for name in self.block_names}
        self.block_widths: dict[str, int] = {name: int(probe[name].shape[-1]) for name in self.block_names}
        self._total_block_width = sum(self.block_widths.values())

    def _state_loss(
        self, s_real: torch.Tensor, s_hat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        real_blocks = self.codec.loss_blocks(s_real[1:])
        predicted_blocks = self.codec.loss_blocks(s_hat[1:])
        loss = per_step_error = 0.0
        extra: dict[str, torch.Tensor] = {}
        for name in self.block_names:
            squared = (predicted_blocks[name] - real_blocks[name]).square()
            scale = self.block_weights[name] * self.block_widths[name] / self._total_block_width
            block_mse = squared.mean()
            loss = loss + block_mse * scale
            per_step_error = per_step_error + squared.mean(-1) * scale
            extra[f"state_block/{name}"] = (block_mse * self.block_weights[name]).detach()
        return loss, per_step_error, extra
