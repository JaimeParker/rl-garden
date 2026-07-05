"""SAC with an LSTM/GRU latent stage and an R2D2-style (stored hidden state +
burn-in + n-step + priority replay) recurrent replay buffer.

See ``rl_garden.networks.recurrent.RecurrentLatentEncoder`` for the original
on-policy-only rationale this buffer removes, and
``rl_garden.buffers.recurrent_replay_buffer`` for the buffer's checkpoint-
aligned sampling design. Encoder-agnostic off-policy-sequence logic (rollout
hidden-state carry, burn-in BPTT critic/actor loss, diagnostics no-ops,
checkpoint-aligned priority replay hyperparameters) lives in ``SequenceSAC``;
this class only adds RNN-specific construction (a future ``TransformerSAC``
would be the GTrXL-based sibling)."""
from __future__ import annotations

from typing import Any

from rl_garden.algorithms.sequence_sac import SequenceSAC
from rl_garden.buffers.recurrent_replay_buffer import RecurrentReplayBuffer
from rl_garden.networks.recurrent import RecurrentLatentEncoder, RecurrentState, RNNType


class RecurrentSAC(SequenceSAC):
    _compatible_checkpoint_algorithms = ("RecurrentSAC",)

    def __init__(
        self,
        env: Any,
        *,
        rnn_type: RNNType = "lstm",
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        **sequence_sac_kwargs: Any,
    ) -> None:
        # Must be set before super().__init__(), which ends by calling
        # self._setup_model() -- our _build_sequence_encoder/_build_replay_buffer
        # overrides below read these attributes.
        self.rnn_type: RNNType = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        super().__init__(env, **sequence_sac_kwargs)

    def _build_sequence_encoder(self, features_extractor) -> RecurrentLatentEncoder:
        return RecurrentLatentEncoder(
            input_dim=features_extractor.features_dim,
            hidden_size=self.rnn_hidden_size,
            rnn_type=self.rnn_type,
            num_layers=self.rnn_num_layers,
        )

    def _build_replay_buffer(self) -> RecurrentReplayBuffer:
        return RecurrentReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            burn_in_len=self.burn_in_len,
            learning_len=self.learning_len,
            forward_len=self.forward_len,
            rnn_type=self.rnn_type,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_num_layers=self.rnn_num_layers,
            gamma=self.gamma,
            prio_exponent=self.prio_exponent,
            importance_sampling_exponent=self.importance_sampling_exponent,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _initial_state_from_sample(self, data) -> RecurrentState:
        """Sample dataclasses store hidden state batch-dim-first (B, num_layers,
        H) so _slice_batch needs no override; transpose to the encoder's native
        (num_layers, B, H) here, right before the one call site that needs it."""
        h = data.initial_hidden_h.transpose(0, 1).contiguous()
        if data.initial_hidden_c is None:
            return h
        c = data.initial_hidden_c.transpose(0, 1).contiguous()
        return h, c

    # --- checkpointing ---

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "rnn_type": self.rnn_type,
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_num_layers": self.rnn_num_layers,
        }
