"""PPO with a GTrXL (Gated Transformer-XL) latent stage between the encoder
and actor/critic heads -- Transformer sibling of ``RecurrentPPO``."""

from __future__ import annotations

from typing import Any, Mapping

from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.ppo import TransformerPPOParams
from rl_garden.algorithms.sequence_ppo import SequencePPO
from rl_garden.networks.gtrxl import GTrXLLatentEncoder


class TransformerPPO(SequencePPO):
    """PPO whose policy carries a GTrXL segment-recurrent memory across rollout
    steps and trains it via BPTT over sequence-preserving minibatches.

    See ``rl_garden.networks.gtrxl.GTrXLLatentEncoder`` for the memory/masking
    design and why burn-in is unimplemented (on-policy, no replay staleness).
    Encoder-agnostic rollout/BPTT logic lives in ``SequencePPO``; this class
    only adds GTrXL-specific construction (``RecurrentPPO`` is the RNN-based
    sibling)."""

    params_class = TransformerPPOParams

    _compatible_checkpoint_algorithms = ("TransformerPPO",)

    def __init__(
        self,
        env: Any,
        *,
        params: TransformerPPOParams | None = None,
        **ppo_kwargs: Any,
    ) -> None:
        parameter_names = self.params_class.__dataclass_fields__
        overrides = {name: ppo_kwargs.pop(name) for name in tuple(ppo_kwargs)
                     if name in parameter_names}
        params = resolve_params(self.params_class, params, overrides)
        super().__init__(env, params=params, **ppo_kwargs)

    def _build_sequence_encoder(self, actor_extractor) -> GTrXLLatentEncoder:
        return GTrXLLatentEncoder(
            input_dim=actor_extractor.features_dim,
            embed_dim=self.embed_dim,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            num_layers=self.num_transformer_layers,
            mlp_num=self.mlp_num,
            memory_len=self.memory_len,
            dropout_rate=self.dropout_rate,
            gru_bias=self.gru_bias,
        )

    @classmethod
    def configured_sequence_stage(cls, settings: Mapping[str, Any]) -> tuple[str, str]:
        del settings
        return "gtrxl", f"{cls.__name__} builds a GTrXL stage"

    def _builtin_stage_params(self) -> dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_transformer_layers,
            "mlp_num": self.mlp_num,
            "memory_len": self.memory_len,
            "dropout_rate": self.dropout_rate,
            "gru_bias": self.gru_bias,
        }

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
        }
