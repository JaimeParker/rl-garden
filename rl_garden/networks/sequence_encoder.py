"""Shared structural contract for stateful sequence-latent encoders.

`RecurrentLatentEncoder` (LSTM/GRU) and `GTrXLLatentEncoder` (Transformer) both
implement this shape, letting `RecurrentPPOPolicy`/`RecurrentRolloutBuffer` stay
encoder-agnostic. `Protocol` is structural: neither encoder needs to subclass
this to satisfy it, this is a documented contract, not a base class in the
inheritance sense. A future SSM/Mamba encoder is expected to satisfy the same
Protocol without requiring any changes here.

State shape is deliberately NOT part of the contract -- RNN state is a compact
`(num_layers, B, hidden_size)` vector, GTrXL state is a much larger
`(memory, memory_valid)` pair holding a window of raw past activations. Only
the requirement that batch is tensor-dim 1 (for every tensor in the state,
tuple or not) is shared, since `RecurrentRolloutBuffer._index_recurrent_state`
and `mask_state`/`index_state` both index/broadcast along dim 1.
"""
from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

import torch

SequenceState = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


@runtime_checkable
class SequenceLatentEncoder(Protocol):
    @property
    def features_dim(self) -> int: ...

    def get_initial_state(self, batch_size: int, device: torch.device) -> SequenceState: ...

    def mask_state(self, state: SequenceState, keep_mask: torch.Tensor) -> SequenceState: ...

    def index_state(self, state: SequenceState, indices: torch.Tensor) -> SequenceState: ...

    def step(
        self,
        latent: torch.Tensor,
        state: SequenceState,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, SequenceState]: ...

    def forward_sequence(
        self,
        latent: torch.Tensor,
        state: SequenceState,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, SequenceState]: ...

    def forward_sequence_with_burn_in(
        self,
        latent: torch.Tensor,
        state: SequenceState,
        episode_starts: torch.Tensor,
        burn_in_len: int,
    ) -> tuple[torch.Tensor, SequenceState]: ...
