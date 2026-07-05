"""SAC with a GTrXL (Gated Transformer-XL) latent stage and a dense,
burn-in-from-zero replay buffer -- Transformer sibling of ``RecurrentSAC``."""
from __future__ import annotations

from typing import Any

from rl_garden.algorithms.sequence_sac import SequenceSAC
from rl_garden.buffers.transformer_replay_buffer import TransformerReplayBuffer
from rl_garden.networks.gtrxl import GTrXLLatentEncoder, GTrXLState


class TransformerSAC(SequenceSAC):
    """SAC whose policy carries a GTrXL segment-recurrent memory across rollout
    steps and trains it via zero-init-burn-in-refreshed BPTT over densely
    (not checkpoint-aligned) sampled, priority-sampled replay windows.

    See ``rl_garden.networks.gtrxl.GTrXLLatentEncoder.forward_sequence_with_burn_in``
    for why no stored per-transition hidden state is needed (unlike
    ``RecurrentSAC``'s R2D2 scheme) and for the approximate-reconstruction
    fidelity tradeoff burn_in_len/memory_len/num_transformer_layers control.
    Encoder-agnostic off-policy-sequence logic lives in ``SequenceSAC``; this
    class only adds GTrXL-specific construction (``RecurrentSAC`` is the
    RNN-based sibling)."""

    _compatible_checkpoint_algorithms = ("TransformerSAC",)

    def __init__(
        self,
        env: Any,
        *,
        embed_dim: int = 256,
        head_dim: int = 64,
        num_heads: int = 4,
        num_transformer_layers: int = 3,
        mlp_num: int = 2,
        memory_len: int = 16,
        dropout_rate: float = 0.0,
        gru_bias: float = 2.0,
        **sequence_sac_kwargs: Any,
    ) -> None:
        # Must be set before super().__init__(), which ends by calling
        # self._setup_model() -- our _build_sequence_encoder/_build_replay_buffer
        # overrides below read these attributes.
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.mlp_num = mlp_num
        self.memory_len = memory_len
        self.dropout_rate = dropout_rate
        self.gru_bias = gru_bias
        super().__init__(env, **sequence_sac_kwargs)

    def _build_sequence_encoder(self, features_extractor) -> GTrXLLatentEncoder:
        if self.burn_in_len < self.memory_len:
            raise ValueError(
                f"TransformerSAC requires burn_in_len ({self.burn_in_len}) >= "
                f"memory_len ({self.memory_len}) -- below this floor the memory "
                "window isn't even fully populated once. This is a floor, not a "
                "guarantee of exact reconstruction: full fidelity for all "
                "num_transformer_layers layers needs burn_in_len >= "
                "num_transformer_layers * memory_len (deeper layers see a "
                "progressively-more-truncated effective context otherwise, "
                "matching Transformer-XL's depth x segment-length compounding -- "
                "see GTrXLLatentEncoder.forward_sequence_with_burn_in). Treat this "
                "the same way R2D2 treats partial burn-in: an accepted, tunable "
                "approximation."
            )
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

    def _build_replay_buffer(self) -> TransformerReplayBuffer:
        return TransformerReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            burn_in_len=self.burn_in_len,
            learning_len=self.learning_len,
            forward_len=self.forward_len,
            gamma=self.gamma,
            prio_exponent=self.prio_exponent,
            importance_sampling_exponent=self.importance_sampling_exponent,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _initial_state_from_sample(self, data) -> GTrXLState:
        # GTrXL's memory is a bounded sliding window of raw activations, not a
        # compressed RNN-style state -- unlike RecurrentSAC, no stored
        # per-transition hidden state is needed at all; burn-in always starts
        # from zero (see _build_sequence_encoder's burn_in_len/memory_len floor).
        batch_size = data.episode_starts.shape[1]
        return self.policy.recurrent_encoder.get_initial_state(batch_size, self.device)

    def _replay_buffer_add_kwargs(
        self, action_context, obs, next_obs, real_next_obs, infos, need_final_obs
    ) -> dict[str, Any]:
        # GTrXL never stores a per-transition hidden checkpoint (see
        # _initial_state_from_sample) -- override SequenceSAC's RNN-oriented
        # default ({"hidden": ...}) since TransformerReplayBuffer.add() has no
        # `hidden` parameter.
        del action_context, obs, next_obs, real_next_obs, infos, need_final_obs
        return {}

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
