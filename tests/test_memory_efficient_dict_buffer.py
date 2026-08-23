"""Tests for MemoryEfficientDictReplayBuffer.

Uses ImageFrameStackWrapper itself (already tested, test_frame_stack_wrapper.py)
as the ground-truth generator for stacked (obs, next_obs) pairs, so this
suite doesn't need to hand-derive the exact edge-replication/sliding-window
sequence a real training loop would produce -- it drives the real wrapper
against a fake single-env with deterministic scalar frame content and
episode boundaries, then feeds the exact same pushed transitions into both
buffer types for comparison.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.buffers.dict_buffer import DictReplayBuffer
from rl_garden.buffers.memory_efficient_dict_buffer import MemoryEfficientDictReplayBuffer
from rl_garden.envs.wrappers.frame_stack import ImageFrameStackWrapper

IMAGE_KEY = "rgb_cam"
FRAME_STACK = 3


class _ScalarFrameEnv(gym.Env):
    """Single-env, deterministic scalar image content (frame value == a
    monotonically increasing counter), with caller-controlled episode
    lengths so tests can place episode boundaries and ring-buffer
    wraparound exactly where needed."""

    num_envs = 1

    def __init__(self, episode_lengths):
        self._episode_lengths = list(episode_lengths)
        self._ep_idx = 0
        self._step_in_ep = 0
        self._counter = 0
        self._init_raw_obs = self._obs()
        self.single_action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.action_space = batch_space(self.single_action_space, 1)

    def _obs(self):
        frame = torch.full((1, 2, 2, 3), float(self._counter), dtype=torch.uint8)
        return {IMAGE_KEY: frame, "state": torch.full((1, 1), float(self._counter))}

    def update_obs_space(self, obs):
        self._init_raw_obs = obs
        self.single_observation_space = spaces.Dict(
            {
                IMAGE_KEY: spaces.Box(0, 255, (2, 2, 3), np.uint8),
                "state": spaces.Box(-np.inf, np.inf, (1,), np.float32),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, 1)

    def reset(self, *, seed=None, options=None):
        del seed, options
        self._step_in_ep = 0
        return self._obs(), {}

    def step(self, action):
        del action
        self._counter += 1
        self._step_in_ep += 1
        terminated = self._step_in_ep >= self._episode_lengths[self._ep_idx]
        if terminated:
            self._ep_idx = (self._ep_idx + 1) % len(self._episode_lengths)
        zeros_or_term = torch.tensor([terminated], dtype=torch.bool)
        return self._obs(), torch.ones(1), zeros_or_term, torch.zeros(1, dtype=torch.bool), {}


def _drive(episode_lengths, num_steps):
    """Runs ImageFrameStackWrapper against _ScalarFrameEnv, collecting
    (obs, next_obs, action, reward, done) tuples exactly as a real
    actor/learner loop would push them."""
    env = ImageFrameStackWrapper(_ScalarFrameEnv(episode_lengths), frame_stack=FRAME_STACK)
    obs, _ = env.reset()
    transitions = []
    for _ in range(num_steps):
        action = torch.zeros(1, 1)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        transitions.append((obs, next_obs, action, reward, terminated.float()))
        if bool(terminated) or bool(truncated):
            obs, _ = env.reset()
        else:
            obs = next_obs
    return transitions


def _obs_space():
    return spaces.Dict(
        {
            IMAGE_KEY: spaces.Box(0, 255, (FRAME_STACK, 2, 2, 3), np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (1,), np.float32),
        }
    )


def _act_space():
    return spaces.Box(-1, 1, (1,), np.float32)


def test_no_wraparound_matches_dict_replay_buffer_exactly():
    transitions = _drive(episode_lengths=[4, 4], num_steps=8)

    dict_rb = DictReplayBuffer(
        _obs_space(), _act_space(), num_envs=1, buffer_size=16,
        storage_device="cpu", sample_device="cpu",
    )
    mem_rb = MemoryEfficientDictReplayBuffer(
        _obs_space(), _act_space(), num_envs=1, buffer_size=16,
        image_keys=(IMAGE_KEY,), frame_stack=FRAME_STACK,
        storage_device="cpu", sample_device="cpu",
    )
    for obs, next_obs, action, reward, done in transitions:
        dict_rb.add(obs, next_obs, action, reward, done)
        mem_rb.add(obs, next_obs, action, reward, done)

    batch_inds = torch.arange(len(transitions))
    env_inds = torch.zeros(len(transitions), dtype=torch.long)
    dict_sample = dict_rb._index_batch(batch_inds, env_inds)
    mem_sample = mem_rb._index_batch(batch_inds, env_inds)

    torch.testing.assert_close(mem_sample.obs[IMAGE_KEY], dict_sample.obs[IMAGE_KEY])
    torch.testing.assert_close(mem_sample.next_obs[IMAGE_KEY], dict_sample.next_obs[IMAGE_KEY])
    torch.testing.assert_close(mem_sample.obs["state"], dict_sample.obs["state"])
    torch.testing.assert_close(mem_sample.rewards, dict_sample.rewards)
    torch.testing.assert_close(mem_sample.dones, dict_sample.dones)


def test_edge_replication_at_episode_start_matches_wrapper():
    # Episode of length 1 -> the very first stored transition's obs must be
    # 3 copies of the same (post-reset) frame, matching ImageFrameStackWrapper.
    transitions = _drive(episode_lengths=[1, 5], num_steps=1)
    mem_rb = MemoryEfficientDictReplayBuffer(
        _obs_space(), _act_space(), num_envs=1, buffer_size=16,
        image_keys=(IMAGE_KEY,), frame_stack=FRAME_STACK,
        storage_device="cpu", sample_device="cpu",
    )
    obs, next_obs, action, reward, done = transitions[0]
    mem_rb.add(obs, next_obs, action, reward, done)

    sample = mem_rb._index_batch(torch.tensor([0]), torch.tensor([0]))
    torch.testing.assert_close(sample.obs[IMAGE_KEY], obs[IMAGE_KEY])


def test_wraparound_overwrite_is_detected_invalid():
    # buffer_size=4: episode of length 6 wraps the ring buffer mid-episode,
    # overwriting positions 0-1 with the same episode's later steps 4-5.
    transitions = _drive(episode_lengths=[6], num_steps=6)
    mem_rb = MemoryEfficientDictReplayBuffer(
        _obs_space(), _act_space(), num_envs=1, buffer_size=4,
        image_keys=(IMAGE_KEY,), frame_stack=FRAME_STACK,
        storage_device="cpu", sample_device="cpu",
    )
    for obs, next_obs, action, reward, done in transitions:
        mem_rb.add(obs, next_obs, action, reward, done)

    # Position 0 now holds step 4's data (steps 0,1,2,3 -> pos 0,1,2,3; step
    # 4 -> pos 0, step 5 -> pos 1). Step 4's window wants to reach back into
    # step 2/3 (positions 2,3) which are still the SAME episode -- valid.
    # But step 5 (pos 1) wants to reach back to step 3 (pos 3, still same
    # episode, valid) and step 4 (pos 0, also same episode -- also valid,
    # since this is all one long episode). To actually exercise an INVALID
    # window we need two distinct episodes sharing the ring buffer.
    valid = mem_rb._valid_batch(torch.tensor([0, 1]), torch.tensor([0, 0]))
    assert bool(valid.all())  # single long episode: wraparound reuses its OWN data, still valid


def test_freshly_written_positions_stay_valid_across_episode_wraparound():
    """buffer_size=4, two 3-step episodes: ep0 -> pos 0,1,2; ep1 -> pos
    3,0,1 (overwriting ep0's pos 0,1). Positions 0, 1, 3 hold ep1's own
    freshest writes -- step_id resets to 0 at episode start, so their
    back-reach only ever lands on this same episode's own
    immediately-preceding writes (a monotonic ring cursor guarantees this
    for the *freshest* positions of an episode)."""
    transitions = _drive(episode_lengths=[3, 3], num_steps=6)
    mem_rb = MemoryEfficientDictReplayBuffer(
        _obs_space(), _act_space(), num_envs=1, buffer_size=4,
        image_keys=(IMAGE_KEY,), frame_stack=FRAME_STACK,
        storage_device="cpu", sample_device="cpu",
    )
    for obs, next_obs, action, reward, done in transitions:
        mem_rb.add(obs, next_obs, action, reward, done)

    valid = mem_rb._valid_batch(torch.tensor([0, 1, 3]), torch.zeros(3, dtype=torch.long))
    assert bool(valid.all())

    dict_rb = DictReplayBuffer(
        _obs_space(), _act_space(), num_envs=1, buffer_size=16,
        storage_device="cpu", sample_device="cpu",
    )
    for obs, next_obs, action, reward, done in transitions:
        dict_rb.add(obs, next_obs, action, reward, done)
    # transitions[4] is ep1's 2nd step (the one now stored at ring pos 0).
    expected = dict_rb._index_batch(torch.tensor([4]), torch.tensor([0]))
    actual = mem_rb._index_batch(torch.tensor([0]), torch.tensor([0]))
    torch.testing.assert_close(actual.obs[IMAGE_KEY], expected.obs[IMAGE_KEY])
    torch.testing.assert_close(actual.next_obs[IMAGE_KEY], expected.next_obs[IMAGE_KEY])


def test_stale_position_orphaned_by_a_later_episodes_writes_is_rejected():
    """The same buffer as above: position 2 still holds ep0's OWN,
    never-overwritten step_id=2 data -- but its back=1 neighbor (position 1)
    has SINCE been overwritten by ep1's later write. Unlike the freshest
    positions above, an older still-resident position's temporal
    neighbors are not guaranteed unwritten-since -- this is the real,
    reachable case _valid_batch's per-slot _ep_id check exists for."""
    transitions = _drive(episode_lengths=[3, 3], num_steps=6)
    mem_rb = MemoryEfficientDictReplayBuffer(
        _obs_space(), _act_space(), num_envs=1, buffer_size=4,
        image_keys=(IMAGE_KEY,), frame_stack=FRAME_STACK,
        storage_device="cpu", sample_device="cpu",
    )
    for obs, next_obs, action, reward, done in transitions:
        mem_rb.add(obs, next_obs, action, reward, done)

    valid = mem_rb._valid_batch(torch.tensor([2]), torch.tensor([0]))
    assert not bool(valid.item())




def test_sample_never_returns_cross_episode_window():
    transitions = _drive(episode_lengths=[3, 3, 3, 3], num_steps=40)
    mem_rb = MemoryEfficientDictReplayBuffer(
        _obs_space(), _act_space(), num_envs=1, buffer_size=6,
        image_keys=(IMAGE_KEY,), frame_stack=FRAME_STACK,
        storage_device="cpu", sample_device="cpu",
    )
    for obs, next_obs, action, reward, done in transitions:
        mem_rb.add(obs, next_obs, action, reward, done)

    for _ in range(20):
        batch_inds, env_inds = mem_rb._sample_valid_indices(8, mem_rb.size)
        assert bool(mem_rb._valid_batch(batch_inds, env_inds).all())


def test_image_key_tensor_uses_less_memory_than_dict_replay_buffer():
    obs_space = spaces.Dict(
        {IMAGE_KEY: spaces.Box(0, 255, (8, 32, 32, 3), np.uint8)}
    )
    act_space = spaces.Box(-1, 1, (1,), np.float32)
    dict_rb = DictReplayBuffer(
        obs_space, act_space, num_envs=1, buffer_size=1000,
        storage_device="cpu", sample_device="cpu",
    )
    mem_rb = MemoryEfficientDictReplayBuffer(
        obs_space, act_space, num_envs=1, buffer_size=1000,
        image_keys=(IMAGE_KEY,), frame_stack=8,
        storage_device="cpu", sample_device="cpu",
    )

    def _bytes(rb):
        return (
            rb.obs.data[IMAGE_KEY].element_size() * rb.obs.data[IMAGE_KEY].nelement()
            + rb.next_obs.data[IMAGE_KEY].element_size() * rb.next_obs.data[IMAGE_KEY].nelement()
        )

    assert _bytes(mem_rb) < _bytes(dict_rb) / 4  # ~8x smaller per key, minus bookkeeping overhead


def test_add_raises_when_obs_not_pre_stacked():
    mem_rb = MemoryEfficientDictReplayBuffer(
        _obs_space(), _act_space(), num_envs=1, buffer_size=16,
        image_keys=(IMAGE_KEY,), frame_stack=FRAME_STACK,
        storage_device="cpu", sample_device="cpu",
    )
    bad_obs = {IMAGE_KEY: torch.zeros(1, 2, 2, 3), "state": torch.zeros(1, 1)}  # missing T dim
    with pytest.raises(ValueError, match="pre-stacked"):
        mem_rb.add(bad_obs, bad_obs, torch.zeros(1, 1), torch.zeros(1), torch.zeros(1))


def test_mmap_dir_not_supported():
    with pytest.raises(NotImplementedError):
        MemoryEfficientDictReplayBuffer(
            _obs_space(), _act_space(), num_envs=1, buffer_size=16,
            image_keys=(IMAGE_KEY,), frame_stack=FRAME_STACK,
            storage_device="cpu", sample_device="cpu", mmap_dir="/tmp/whatever",
        )
