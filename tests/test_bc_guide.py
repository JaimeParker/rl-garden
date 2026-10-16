"""``BCGuide``: output shape and action mask, RandomShift keeps the shape,
3-frame windows stay inside one episode with the action of the last frame as
target, and 50 training steps lower the loss."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers.rehearsal_sequence_buffer import RehearsalSequenceBuffer, SequenceEpisode
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.encoders.augment import RandomShiftsAug
from rl_garden.policies.action_mask import action_mask
from rl_garden.policies.bc_guide import BCGuide, bc_window_starts, sample_bc_batch, train_bc_guide

IMAGE = (64, 64, 3)
PADDED = 5
REAL = 3


def _prior(episode_lengths=(9, 7, 11)) -> SequenceReplayBuffer:
    """Frame pixels = ``20 * ep + t`` and action = ``(ep * 100 + t) / 1000``
    in every real dim, so a window's target identifies its last frame."""
    total = sum(episode_lengths)
    buffer = SequenceReplayBuffer(
        spaces.Dict({"rgb_default": spaces.Box(0, 255, IMAGE, dtype=np.uint8)}),
        spaces.Box(-1.0, 1.0, (PADDED,), dtype=np.float32),
        1,
        total,
        cross_episode=True,
        priority=False,
        horizon=4,
        carry_spec={"deter": (2,)},
        storage_device="cpu",
        sample_device="cpu",
    )
    episodes = []
    for ep, length in enumerate(episode_lengths):
        t = torch.arange(length)
        episodes.append(
            SequenceEpisode(
                key=("walker", ep),
                obs=(20 * ep + t).to(torch.uint8).reshape(length, 1, 1, 1).expand(length, *IMAGE).clone(),
                action=((ep * 100 + t).float() / 1000.0).unsqueeze(1).repeat(1, REAL),
                reward=torch.zeros(length),
            )
        )
    RehearsalSequenceBuffer(buffer, prior=buffer).load_prior_episodes(episodes)
    return buffer


def test_output_shape_and_mask():
    mask = action_mask(REAL, PADDED)
    guide = BCGuide(IMAGE, mask)
    frames = torch.randint(0, 256, (4, 3, *IMAGE), dtype=torch.uint8)
    raw = guide(frames)
    assert raw.shape == (4, PADDED)
    act = guide.act(frames)
    assert act.shape == (4, PADDED)
    assert torch.all(act[:, REAL:] == 0)
    torch.testing.assert_close(act[:, :REAL], raw[:, :REAL])


def test_random_shift_keeps_shape():
    x = torch.rand(6, 9, 64, 64)
    assert RandomShiftsAug(3)(x).shape == x.shape
    guide = BCGuide((64, 64, 3), action_mask(REAL, PADDED))
    frames = torch.randint(0, 256, (2, 3, 64, 64, 3), dtype=torch.uint8)
    assert guide(frames, augment=True).shape == (2, PADDED)


def test_windows_stay_inside_one_episode_and_target_last_frame():
    buffer = _prior()
    starts = bc_window_starts(buffer)
    assert starts[0].numel() == (9 - 2) + (7 - 2) + (11 - 2)
    frames, action = sample_bc_batch(buffer, starts, 64, torch.device("cpu"))
    assert frames.shape == (64, 3, *IMAGE) and action.shape == (64, PADDED)
    pixel = frames[:, :, 0, 0, 0].long()  # (B, 3): 20 * ep + t
    episode, step = pixel // 20, pixel % 20
    assert (episode == episode[:, :1]).all(), "a window crossed an episode boundary"
    assert ((step[:, 1:] - step[:, :-1]) == 1).all()
    expected = (episode[:, 2] * 100 + step[:, 2]).float() / 1000.0
    torch.testing.assert_close(action[:, 0], expected)
    assert torch.all(action[:, REAL:] == 0)


def test_training_lowers_loss():
    torch.manual_seed(0)
    buffer = _prior()
    guide = BCGuide(IMAGE, action_mask(REAL, PADDED), depth=8, units=64)
    losses = train_bc_guide(guide, buffer, steps=50, batch_size=32, lr=1e-3)
    assert len(losses) == 50
    assert np.mean(losses[-10:]) < np.mean(losses[:10])
    assert not guide.training
    assert not any(p.requires_grad for p in guide.parameters())
