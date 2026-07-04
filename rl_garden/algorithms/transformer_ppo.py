"""PPO with a GTrXL (Gated Transformer-XL) latent stage between the encoder
and actor/critic heads -- Transformer sibling of ``RecurrentPPO``."""

from __future__ import annotations

from typing import Any

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

    _compatible_checkpoint_algorithms = ("TransformerPPO",)

    def __init__(
        self,
        env: Any,
        *,
        embed_dim: int = 256,
        head_dim: int = 64,
        num_heads: int = 4,
        num_transformer_layers: int = 3,
        mlp_num: int = 2,
        memory_len: int = 64,
        dropout_rate: float = 0.0,
        gru_bias: float = 2.0,
        **ppo_kwargs: Any,
    ) -> None:
        # Must be set before super().__init__(), which ends by calling
        # self._setup_model() -- our _build_sequence_encoder() override below
        # reads these attributes.
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.mlp_num = mlp_num
        self.memory_len = memory_len
        self.dropout_rate = dropout_rate
        self.gru_bias = gru_bias
        super().__init__(env, **ppo_kwargs)

    def _build_sequence_encoder(self, features_extractor) -> GTrXLLatentEncoder:
        return GTrXLLatentEncoder(
            input_dim=features_extractor.features_dim,
            embed_dim=self.embed_dim,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            num_layers=self.num_transformer_layers,
            mlp_num=self.mlp_num,
            memory_len=self.memory_len,
            dropout_rate=self.dropout_rate,
            gru_bias=self.gru_bias,
        )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "embed_dim": self.embed_dim,
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "num_transformer_layers": self.num_transformer_layers,
            "mlp_num": self.mlp_num,
            "memory_len": self.memory_len,
            "dropout_rate": self.dropout_rate,
            "gru_bias": self.gru_bias,
        }
