"""``BCGuide``: NCRL's execution-guidance prior policy (``3rd_party/ncrl/src/behavior_cloning.py``).

A small deterministic behaviour-cloning policy over a 3-frame stack, trained
once on the retrieved prior episodes before online fine-tuning and frozen
afterwards. ``NCRL._rollout_action`` substitutes its action for the actor's
inside a guidance window.

Architecture (official ``BCAgent`` with ``nets.Encoder`` defaults): 4 conv
layers (kernel 4, stride 2, no padding, depth ``48 * 2**i``), each followed by
a LayerNorm over channels and ELU; then ``Linear-LN-ELU`` x2 (512 units) and a
linear output of ``padded_action_dim``. Input frames are ``/255 - 0.5``;
training applies ``RandomShiftsAug(pad=3)`` to the stacked channels. The loss
is MSE multiplied by the action mask (padded dims carry no loss), averaged
over every dim like the official ``(mse * mask).mean()``.

Training windows follow rl-garden's row alignment (``actions[t]`` is executed
at ``obs[t]``): frames ``obs[t], obs[t+1], obs[t+2]`` of one episode, target
``actions[t+2]``. This is the official chunk-4 target ``action[3]`` (the
action that led from ``o_2`` to ``o_3`` in Dreamer alignment).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.common.obs_utils import index_obs
from rl_garden.encoders.augment import RandomShiftsAug

BC_FRAME_STACK = 3


class _ChannelLayerNorm(nn.LayerNorm):
    """LayerNorm over the channel dim of an NCHW tensor (official
    ``ChannelNorm``, eps 1e-3)."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class BCGuide(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        action_mask: torch.Tensor,
        *,
        frame_stack: int = BC_FRAME_STACK,
        depth: int = 48,
        units: int = 512,
        shift_pad: int = 3,
    ) -> None:
        super().__init__()
        height, width, channels = image_shape
        self.frame_stack = frame_stack
        self.register_buffer("action_mask", action_mask.float().clone())

        layers: list[nn.Module] = []
        in_ch = frame_stack * channels
        for i in range(4):
            out_ch = depth * 2**i
            layers += [nn.Conv2d(in_ch, out_ch, 4, stride=2), _ChannelLayerNorm(out_ch), nn.ELU()]
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)
        with torch.no_grad():
            embed_dim = self.encoder(torch.zeros(1, frame_stack * channels, height, width)).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(embed_dim, units), nn.LayerNorm(units), nn.ELU(),
            nn.Linear(units, units), nn.LayerNorm(units), nn.ELU(),
            nn.Linear(units, action_mask.numel()),
        )
        self.aug = RandomShiftsAug(shift_pad)

    def forward(self, frames: torch.Tensor, *, augment: bool = False) -> torch.Tensor:
        """``frames``: uint8 ``(B, frame_stack, H, W, C)``, oldest first.
        Returns the raw (unmasked) ``(B, padded_action_dim)`` prediction."""
        batch, stack, height, width, channels = frames.shape
        x = frames.float().div(255.0).permute(0, 1, 4, 2, 3).reshape(batch, stack * channels, height, width)
        if augment:
            x = self.aug(x)
        return self.head(self.encoder(x - 0.5).flatten(1))

    @torch.no_grad()
    def act(self, frames: torch.Tensor) -> torch.Tensor:
        """Deterministic guidance action, padded dims zeroed."""
        return self.forward(frames) * self.action_mask

    def loss(self, frames: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
        prediction = self.forward(frames, augment=True)
        return ((prediction - target_action) ** 2 * self.action_mask).mean()


def bc_window_starts(buffer: SequenceReplayBuffer, frame_stack: int = BC_FRAME_STACK) -> tuple[torch.Tensor, torch.Tensor]:
    """``(t, env)`` of every ``frame_stack``-row window inside one episode of
    ``buffer`` (no ``episode_end`` before the last row, no read across the
    ring-buffer write position)."""
    size = buffer.size
    if size < frame_stack:
        raise ValueError(f"buffer holds {size} rows per env, fewer than frame_stack={frame_stack}.")
    t = torch.arange(size - frame_stack + 1, device=buffer.storage_device)
    valid = torch.ones(t.numel(), buffer.num_envs, dtype=torch.bool, device=buffer.storage_device)
    for offset in range(frame_stack - 1):
        valid &= ~buffer.episode_ends[t + offset]
    if buffer.full:
        crosses_write_position = (t < buffer.pos) & (t + frame_stack - 1 >= buffer.pos)
        valid &= ~crosses_write_position.unsqueeze(1)
    t_idx, env_idx = valid.nonzero(as_tuple=True)
    if t_idx.numel() == 0:
        raise ValueError("buffer contains no episode with at least frame_stack rows.")
    return t[t_idx], env_idx


def sample_bc_batch(
    buffer: SequenceReplayBuffer,
    starts: tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames ``(B, frame_stack, H, W, C)`` and target actions ``(B, A)``."""
    window_t, window_env = starts
    pick = torch.randint(0, window_t.numel(), (batch_size,), device=window_t.device)
    t0, env = window_t[pick], window_env[pick]
    (obs_key,) = buffer.observation_space.spaces.keys()
    frame_stack = BC_FRAME_STACK
    rows = t0.unsqueeze(1) + torch.arange(frame_stack, device=t0.device)
    frames = index_obs(buffer.obs, (rows, env.unsqueeze(1).expand_as(rows)))[obs_key]
    action = buffer.actions[t0 + frame_stack - 1, env]
    return frames.to(device), action.to(device)


def train_bc_guide(
    bc_guide: BCGuide,
    prior_buffer: SequenceReplayBuffer,
    *,
    steps: int = 150_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    log_every: Optional[int] = None,
) -> list[float]:
    """Trains ``bc_guide`` with Adam on ``prior_buffer`` (a filled
    uniform-mode ``SequenceReplayBuffer``) and puts it in eval mode. The
    optimizer is local: the guide is never updated again. Returns the loss
    of every step."""
    if bc_guide.frame_stack != BC_FRAME_STACK:
        raise ValueError(f"train_bc_guide samples {BC_FRAME_STACK}-frame windows, got frame_stack={bc_guide.frame_stack}.")
    device = bc_guide.action_mask.device
    starts = bc_window_starts(prior_buffer)
    optimizer = torch.optim.Adam(bc_guide.parameters(), lr=lr)
    bc_guide.train()
    losses = torch.zeros(steps, device=device)
    for step in range(steps):
        frames, action = sample_bc_batch(prior_buffer, starts, batch_size, device)
        loss = bc_guide.loss(frames, action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses[step] = loss.detach()
        if log_every and (step + 1) % log_every == 0:
            print(f"[bc_guide] step={step + 1}/{steps} loss={float(losses[step]):.5f}", flush=True)
    bc_guide.eval()
    for parameter in bc_guide.parameters():
        parameter.requires_grad_(False)
    return losses.tolist()
