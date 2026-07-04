"""GTrXL (Gated Transformer-XL) latent encoder -- Transformer counterpart to
`RecurrentLatentEncoder`, both satisfying `SequenceLatentEncoder`.

Core attention/gating math is ported from DI-engine's
`ding/torch_utils/network/gtrxl.py` (Apache-2.0, see `3rd_party/DI-engine`):
relative positional encoding (Transformer-XL `_rel_shift`) + GRU gating (the
"Gated" in GTrXL, replacing residual-adds with a GRU-style gate for training
stability, per Parisotto et al. 2020, https://arxiv.org/abs/1910.06764).

Two things are NOT ported from DI-engine, by design:
  1. DI-engine's `Memory`/`GTrXL` keep `self.memory` as internal mutable state
     and have no notion of per-env episode boundaries -- their attention
     `mask` is shape `(cur_seq, full_seq, 1)`, identical across the whole
     batch. Real vectorized rollouts need per-env masking (env A's memory may
     be mid-episode while env B just reset), so this module's mask is shaped
     `(cur_seq, full_seq, B)` instead, and every call is explicit
     state-in/state-out (matching `RecurrentLatentEncoder`'s existing
     stateless-per-call contract) rather than a `self.memory` object.
  2. Nothing here calls `.detach()`. Gradient isolation across segments comes
     for free from how `initial_hidden`/`initial_memory` is produced (a
     `torch.no_grad()` rollout snapshot -- see `RecurrentPPO`/`SequencePPO`),
     exactly like `RecurrentLatentEncoder`. Detaching per-step here would
     confine BPTT to a single step and defeat the point of the memory.

State = `(memory, memory_valid)`:
  - `memory`: `(num_layers+1, B, memory_len, embed_dim)` -- slot 0 is the
    input embedding, slot i is the input to attention layer i. Batch is dim 1
    (not DI-engine's dim 2) so `RecurrentRolloutBuffer._index_recurrent_state`
    (`h[:, indices]` per tuple element) and `mask_state`/`index_state` work
    unmodified.
  - `memory_valid`: `(1, B, memory_len)` -- per-env validity of each memory
    slot (False for slots before the current episode started). The leading
    dummy dim exists only so dim 1 is batch, matching `memory`.

`forward_sequence()` loops `step()` T times, same structural shape as
`RecurrentLatentEncoder.forward_sequence()`, but here the loop is NOT just a
slow stand-in for a batched call -- it is required for correctness, not an
accepted-but-suboptimal perf characteristic:

  - `step()`'s FIFO gives every position a *sliding* window of exactly the
    last `memory_len` activations.
  - A batched segment forward over the whole window at once (what DI-engine's
    native `GTrXL.forward()` does, with a single `(cur_seq, full_seq, B)`
    causal+validity mask) gives position *t* within the segment a *growing*
    window of `memory_len + t` positions -- it still attends to pre-segment
    memory that the sliding version has already evicted by the time it
    reaches position *t*. These are different computations with different
    outputs, not two ways of computing the same thing.
  - Rollout necessarily calls `step()` one token at a time (env steps arrive
    one at a time), so rollout always sees the sliding-window behavior.
    `forward_sequence()` must match that exactly for the PPO ratio to be 1 at
    the first training epoch -- looping `step()` is what guarantees train/
    rollout parity. A batched-window `forward_sequence()` would be a
    *different model* from the one that produced the rollout data, not a
    drop-in optimization of this one.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

GTrXLState = tuple[torch.Tensor, torch.Tensor]


class PositionalEmbedding(nn.Module):
    """Sinusoidal embedding of *relative* distance (ported from DI-engine, itself
    adapted from https://github.com/kimiyoung/transformer-xl). Depends only on
    relative position, never absolute episode step -- unlike an absolute/learned
    encoding, this needs no a-priori max-episode-length."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq: torch.Tensor) -> torch.Tensor:
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_embedding = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_embedding.unsqueeze(1)


class GRUGatingUnit(nn.Module):
    """GRU-style gate combining a residual stream `x` with new signal `y`, in
    place of a plain residual add -- the mechanism that makes GTrXL "Gated"."""

    def __init__(self, input_dim: int, bias: float = 2.0) -> None:
        super().__init__()
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bias = nn.Parameter(torch.full([input_dim], bias))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bias)
        h = torch.tanh(self.Wg(y) + self.Ug(r * x))
        return (1 - z) * x + z * h


class RelativeAttention(nn.Module):
    """Transformer-XL relative-position multi-head attention. Shapes are
    seq-major throughout (`(seq, B, dim)`), matching rl-garden's `(T, N, ...)`
    convention."""

    def __init__(self, input_dim: int, head_dim: int, num_heads: int, dropout: nn.Module) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_kv = nn.Linear(input_dim, head_dim * num_heads * 2)
        self.attention_q = nn.Linear(input_dim, head_dim * num_heads)
        self.project = nn.Linear(head_dim * num_heads, input_dim)
        self.project_pos = nn.Linear(input_dim, head_dim * num_heads)
        self.scale = 1 / (head_dim**0.5)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        # x: (bs, head_num, cur_seq, full_seq) -- see DI-engine's `_rel_shift`
        # docstring (rl_garden/../3rd_party/DI-engine/.../gtrxl.py) for the
        # step-by-step derivation of this pad/reshape/slice trick.
        x_padded = F.pad(x, [1, 0])
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        return x_padded[:, :, 1:].view_as(x)

    def forward(
        self,
        inputs: torch.Tensor,
        pos_embedding: torch.Tensor,
        full_input: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # inputs: (cur_seq, B, input_dim) raw (pre-LN) query source.
        # full_input: (full_seq, B, input_dim) LN'd cat(memory, inputs).
        # pos_embedding: (full_seq, 1, input_dim).
        # mask: (cur_seq, full_seq, B) bool, True = masked out.
        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]

        kv = self.attention_kv(full_input)
        key, value = torch.chunk(kv, 2, dim=-1)
        query = self.attention_q(inputs)
        r = self.project_pos(pos_embedding)

        key = key.view(full_seq, bs, self.num_heads, self.head_dim)
        query = query.view(cur_seq, bs, self.num_heads, self.head_dim)
        value = value.view(full_seq, bs, self.num_heads, self.head_dim)
        r = r.view(full_seq, self.num_heads, self.head_dim)

        content_attn = (query + u).permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0)
        position_attn = (query + v).permute(1, 2, 0, 3) @ r.permute(1, 2, 0)
        position_attn = self._rel_shift(position_attn)

        attn = (content_attn + position_attn) * self.scale
        if mask is not None and mask.any():
            attn = attn.masked_fill(mask.permute(2, 0, 1).unsqueeze(1), -float("inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ value.permute(1, 2, 0, 3)
        out = out.permute(2, 0, 1, 3).contiguous().view(cur_seq, bs, self.num_heads * self.head_dim)
        return self.dropout(self.project(out))


class GatedTransformerXLLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        hidden_dim: int,
        num_heads: int,
        mlp_num: int,
        dropout: nn.Module,
        activation: nn.Module,
        gru_bias: float = 2.0,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.gate1 = GRUGatingUnit(input_dim, gru_bias)
        self.gate2 = GRUGatingUnit(input_dim, gru_bias)
        self.attention = RelativeAttention(input_dim, head_dim, num_heads, dropout)
        dims = [input_dim] + [hidden_dim] * (mlp_num - 1) + [input_dim]
        layers: list[nn.Module] = []
        for i in range(mlp_num):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
            layers.append(dropout)
        self.mlp = nn.Sequential(*layers)
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.layernorm2 = nn.LayerNorm(input_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        pos_embedding: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # inputs: (cur_seq, B, input_dim); memory: (memory_len, B, input_dim).
        full_input = torch.cat([memory, inputs], dim=0)
        x1 = self.layernorm1(full_input)
        a1 = self.dropout(self.attention(inputs, pos_embedding, x1, u, v, mask=mask))
        a1 = self.activation(a1)
        o1 = self.gate1(inputs, a1)
        x2 = self.layernorm2(o1)
        m2 = self.dropout(self.mlp(x2))
        return self.gate2(o1, m2)


class GTrXLLatentEncoder(nn.Module):
    """Explicit-state GTrXL encoder satisfying `SequenceLatentEncoder`. See
    module docstring for the two deliberate departures from DI-engine's
    reference (per-env masking, no internal `.detach()`)."""

    def __init__(
        self,
        input_dim: int,
        *,
        embed_dim: int = 256,
        head_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_num: int = 2,
        memory_len: int = 64,
        dropout_rate: float = 0.0,
        gru_bias: float = 2.0,
    ) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even (sinusoidal encoding), got {embed_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if memory_len < 1:
            raise ValueError(f"memory_len must be >= 1, got {memory_len}")

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.memory_len = memory_len

        activation = nn.ReLU()
        dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.embedding = nn.Sequential(nn.Linear(input_dim, embed_dim), activation)
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                GatedTransformerXLLayer(
                    embed_dim, head_dim, embed_dim, num_heads, mlp_num, dropout, activation, gru_bias
                )
                for _ in range(num_layers)
            ]
        )
        self.u = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.v = nn.Parameter(torch.zeros(num_heads, head_dim))

        pos_embedding_module = PositionalEmbedding(embed_dim)
        full_seq = memory_len + 1
        pos_seq = torch.arange(full_seq - 1, -1, -1.0)
        # (full_seq, 1, embed_dim); fixed given memory_len/embed_dim, so cached
        # once as a buffer (moves with .to(device), never recomputed per step).
        self.register_buffer("_pos_embedding", pos_embedding_module(pos_seq))

    @property
    def features_dim(self) -> int:
        return self.embed_dim

    def get_initial_state(self, batch_size: int, device: torch.device) -> GTrXLState:
        memory = torch.zeros(self.num_layers + 1, batch_size, self.memory_len, self.embed_dim, device=device)
        memory_valid = torch.zeros(1, batch_size, self.memory_len, device=device)
        return memory, memory_valid

    @staticmethod
    def mask_state(state: GTrXLState, keep_mask: torch.Tensor) -> GTrXLState:
        memory, memory_valid = state
        keep_mask = keep_mask.to(dtype=memory.dtype, device=memory.device)
        memory = memory * keep_mask.reshape(1, -1, 1, 1)
        memory_valid = memory_valid * keep_mask.reshape(1, -1, 1)
        return memory, memory_valid

    @staticmethod
    def index_state(state: GTrXLState, indices: torch.Tensor) -> GTrXLState:
        memory, memory_valid = state
        return memory[:, indices], memory_valid[:, indices]

    def step(
        self,
        latent: torch.Tensor,
        state: GTrXLState,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, GTrXLState]:
        """Single rollout/BPTT step. ``latent``: (B, input_dim); ``episode_starts``: (B,)."""
        if episode_starts is not None:
            state = self.mask_state(state, 1.0 - episode_starts.float())
        memory, memory_valid = state
        batch_size = memory.shape[1]

        # Internal computation uses DI-engine's seq-major-per-layer layout
        # (memory_len, B, embed_dim); external state keeps batch at dim 1 for
        # buffer/mask_state/index_state compatibility -- transpose at the
        # boundary only.
        memory_seq_major = memory.transpose(1, 2)  # (num_layers+1, memory_len, B, embed_dim)
        valid_seq_major = memory_valid[0].transpose(0, 1)  # (memory_len, B)

        # Matches DI-engine's GTrXL.forward(): dropout is applied to the
        # embedded input and to the positional embedding (freshly each call,
        # not baked into the cached buffer), in addition to the final output
        # dropout below.
        x = self.dropout(self.embedding(latent)).unsqueeze(0)  # (1, B, embed_dim)
        pos_embedding = self.dropout(self._pos_embedding)
        valid_full = torch.cat([valid_seq_major, torch.ones(1, batch_size, device=latent.device)], dim=0)
        # mask: (cur_seq=1, full_seq, B), True = masked out. No causal term
        # needed for cur_seq=1 (a single new query has nothing "in the
        # future" within memory+itself) -- only per-env memory validity.
        mask = (valid_full < 0.5).unsqueeze(0)  # (1, full_seq, B)

        hidden_states = [x]
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out, pos_embedding, self.u, self.v, memory=memory_seq_major[i], mask=mask)
            hidden_states.append(out)
        out = self.dropout(out)

        new_memory_slot = torch.stack(hidden_states, dim=0)  # (num_layers+1, 1, B, embed_dim)
        new_memory_seq_major = torch.cat([memory_seq_major, new_memory_slot], dim=1)[:, 1:]
        new_valid_seq_major = torch.cat([valid_seq_major, torch.ones(1, batch_size, device=latent.device)], dim=0)[1:]

        new_memory = new_memory_seq_major.transpose(1, 2)  # back to (num_layers+1, B, memory_len, embed_dim)
        new_memory_valid = new_valid_seq_major.transpose(0, 1).unsqueeze(0)  # (1, B, memory_len)
        return out.squeeze(0), (new_memory, new_memory_valid)

    def forward_sequence(
        self,
        latent: torch.Tensor,
        state: GTrXLState,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, GTrXLState]:
        """BPTT update call. ``latent``: (T, B, input_dim); ``episode_starts``: (T, B)."""
        num_steps = latent.shape[0]
        outputs = []
        for t in range(num_steps):
            step_starts = episode_starts[t] if episode_starts is not None else None
            output, state = self.step(latent[t], state, step_starts)
            outputs.append(output)
        return torch.stack(outputs, dim=0), state

    def forward_sequence_with_burn_in(
        self,
        latent: torch.Tensor,
        state: GTrXLState,
        episode_starts: torch.Tensor,
        burn_in_len: int,
    ) -> tuple[torch.Tensor, GTrXLState]:
        raise NotImplementedError(
            "GTrXLLatentEncoder does not support burn-in replay this round -- "
            "on-policy TransformerPPO never calls this (no replay staleness to "
            "correct for, same reason RecurrentPPO never implemented burn-in). "
            "Only relevant if/when a TransformerSAC is built."
        )
