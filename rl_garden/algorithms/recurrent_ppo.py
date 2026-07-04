"""PPO with an LSTM/GRU latent stage between the encoder and actor/critic heads."""

from __future__ import annotations

from typing import Any

from rl_garden.algorithms.sequence_ppo import SequencePPO
from rl_garden.networks.recurrent import RecurrentLatentEncoder, RNNType


class RecurrentPPO(SequencePPO):
    """PPO whose policy carries a persistent LSTM/GRU hidden state across rollout
    steps and trains it via BPTT over sequence-preserving minibatches.

    See ``rl_garden.networks.recurrent.RecurrentLatentEncoder`` for why this is
    on-policy-only (off-policy/replay-based recurrence needs stored hidden state
    and burn-in, neither of which exists here). Encoder-agnostic rollout/BPTT
    logic lives in ``SequencePPO``; this class only adds RNN-specific
    construction (``TransformerPPO`` is the GTrXL-based sibling)."""

    _compatible_checkpoint_algorithms = ("RecurrentPPO",)

    def __init__(
        self,
        env: Any,
        *,
        rnn_type: RNNType = "lstm",
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        **ppo_kwargs: Any,
    ) -> None:
        # Must be set before super().__init__(), which ends by calling
        # self._setup_model() -- our _build_sequence_encoder() override below
        # reads these attributes.
        self.rnn_type: RNNType = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        super().__init__(env, **ppo_kwargs)

    def _build_sequence_encoder(self, features_extractor) -> RecurrentLatentEncoder:
        return RecurrentLatentEncoder(
            input_dim=features_extractor.features_dim,
            hidden_size=self.rnn_hidden_size,
            rnn_type=self.rnn_type,
            num_layers=self.rnn_num_layers,
        )

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "rnn_type": self.rnn_type,
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_num_layers": self.rnn_num_layers,
        }
