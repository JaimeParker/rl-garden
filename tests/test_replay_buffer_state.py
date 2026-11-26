"""Replay-buffer snapshots: save, build a fresh buffer, load, keep sampling.

Every buffer class goes through the file helpers of
``rl_garden.common.checkpoint`` and must then draw the same samples as the
buffer it was saved from, also after both store the same further transitions.
"""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers.chunked_replay_buffer import ChunkedReplayBuffer
from rl_garden.buffers.mc_buffer import MCReplayBuffer, MCReplayBufferMixin
from rl_garden.buffers.memory_efficient_buffer import MemoryEfficientReplayBuffer
from rl_garden.buffers.mmap_multitask_episode_buffer import MmapMultitaskEpisodeBuffer
from rl_garden.buffers.nstep_buffer import LazyNextNStepReplayBuffer, NStepReplayBuffer
from rl_garden.buffers.rebrac_replay_buffer import ReBRACReplayBuffer
from rl_garden.buffers.recurrent_replay_buffer import RecurrentReplayBuffer
from rl_garden.buffers.rehearsal_sequence_buffer import RehearsalSequenceBuffer
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.buffers.sarsa_buffer import SarsaMCReplayBuffer
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.buffers.state_sequence_replay_buffer import StateSequenceReplayBuffer
from rl_garden.buffers.transformer_replay_buffer import TransformerReplayBuffer
from rl_garden.common.checkpoint import (
    load_replay_buffer_file,
    load_replay_buffer_state_dict,
    replay_buffer_state_dict,
    save_replay_buffer_file,
)

NUM_ENVS = 2
PER_ENV = 24
EPISODE_LENGTHS = (7, 9)  # per env, so episode ends do not line up
FRAME_STACK = 3
CARRY_SPEC = {"deter": (4,), "stoch": (2, 3)}

OBS_SPACE = spaces.Dict(
    {
        "state": spaces.Box(-1.0, 1.0, (3,), np.float32),
        "rgb_cam": spaces.Box(0, 255, (4, 4, 3), np.uint8),
    }
)
STACKED_SPACE = spaces.Dict(
    {
        "state": OBS_SPACE["state"],
        "rgb_cam": spaces.Box(0, 255, (FRAME_STACK, 4, 4, 3), np.uint8),
    }
)
ACT_SPACE = spaces.Box(-1.0, 1.0, (2,), np.float32)
CPU = {"storage_device": "cpu", "sample_device": "cpu"}


@pytest.fixture(autouse=True)
def restored_rng():
    with torch.random.fork_rng():
        yield


def _args(observation_space=OBS_SPACE):
    return (observation_space, ACT_SPACE, NUM_ENVS, NUM_ENVS * PER_ENV)


def _uniform():
    return SequenceReplayBuffer(
        *_args(), cross_episode=True, priority=False, horizon=4, carry_spec=CARRY_SPEC, **CPU
    )


WINDOW = {"burn_in_len": 4, "learning_len": 3, "forward_len": 2}

BUFFERS = {
    "plain": lambda: ReplayBuffer(*_args(), **CPU),
    "grasp_penalty": lambda: ReplayBuffer(*_args(), store_grasp_penalty=True, **CPU),
    "rebrac": lambda: ReBRACReplayBuffer(*_args(), **CPU),
    "mc": lambda: MCReplayBuffer(*_args(), gamma=0.9, **CPU),
    "mc_sparse": lambda: MCReplayBuffer(
        *_args(), gamma=0.9, sparse_reward_mc=True, sparse_negative_reward=-1.0, **CPU
    ),
    "sarsa_mc": lambda: SarsaMCReplayBuffer(*_args(), gamma=0.9, **CPU),
    "memory_efficient": lambda: MemoryEfficientReplayBuffer(
        *_args(STACKED_SPACE), image_keys=["rgb_cam"], frame_stack=FRAME_STACK,
        store_grasp_penalty=True, **CPU,
    ),
    "nstep": lambda: NStepReplayBuffer(*_args(), nstep=3, gamma=0.9, **CPU),
    "lazy_nstep": lambda: LazyNextNStepReplayBuffer(
        *_args(), nstep=3, gamma=0.9, final_obs_capacity=2, **CPU
    ),
    "chunked": lambda: ChunkedReplayBuffer(*_args(), horizon_length=3, **CPU),
    "strict_sequence": lambda: SequenceReplayBuffer(*_args(), horizon=3, **CPU),
    "state_sequence": lambda: StateSequenceReplayBuffer(*_args(), horizon=3, **CPU),
    "uniform_sequence": _uniform,
    "rehearsal": lambda: RehearsalSequenceBuffer(_uniform(), online_ratio=0.75),
    "recurrent_lstm": lambda: RecurrentReplayBuffer(*_args(), rnn_hidden_size=5, **WINDOW, **CPU),
    "recurrent_gru": lambda: RecurrentReplayBuffer(
        *_args(), rnn_type="gru", rnn_hidden_size=5, **WINDOW, **CPU
    ),
    "transformer": lambda: TransformerReplayBuffer(*_args(), **WINDOW, **CPU),
}


def _random_obs(buffer) -> dict[str, torch.Tensor]:
    obs = {}
    for key, storage in buffer.obs.data.items():
        shape = storage.shape[1:]
        if key in getattr(buffer, "image_keys", ()):
            shape = (shape[0], buffer.frame_stack, *shape[1:])
        obs[key] = (
            torch.randint(0, 256, shape, dtype=torch.uint8)
            if storage.dtype == torch.uint8
            else torch.randn(shape)
        )
    return obs


def _add_random(buffer, step: int) -> None:
    buffer = getattr(buffer, "online", buffer)
    n = buffer.num_envs
    ends = torch.tensor(
        [step % EPISODE_LENGTHS[env % 2] == EPISODE_LENGTHS[env % 2] - 1 for env in range(n)]
    )
    done = (ends & (torch.rand(n) < 0.5)).float()
    extras = {
        "episode_end": ends,
        "grasp_penalty": torch.randn(n) if getattr(buffer, "store_grasp_penalty", False) else None,
        "is_first": torch.tensor([step % EPISODE_LENGTHS[env % 2] == 0 for env in range(n)]),
        "carry": {
            key: torch.randn(n, *shape) for key, shape in getattr(buffer, "_carry_spec", {}).items()
        },
        "hidden": None,
    }
    if isinstance(buffer, RecurrentReplayBuffer):
        shape = (buffer.rnn_num_layers, n, buffer.rnn_hidden_size)
        hidden = torch.randn(shape)
        extras["hidden"] = (hidden, torch.randn(shape)) if buffer._has_cell_state else hidden
    accepted = set(inspect.signature(buffer.add).parameters)
    strict_sequence = isinstance(buffer, SequenceReplayBuffer) and not buffer.cross_episode
    if strict_sequence or isinstance(buffer, MCReplayBufferMixin):
        accepted = {"episode_end"}
    buffer.add(
        _random_obs(buffer),
        _random_obs(buffer),
        torch.randn(n, *buffer.actions.shape[2:]),
        torch.randn(n),
        done,
        **{name: value for name, value in extras.items() if name in accepted},
    )


def fill_randomly(buffer, steps: int, *, first_step: int = 0) -> None:
    for step in range(first_step, first_step + steps):
        _add_random(buffer, step)


def assert_tree_equal(left, right, path: str = "") -> None:
    if dataclasses.is_dataclass(left):
        left, right = vars(left), vars(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_tree_equal(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_tree_equal(a, b, f"{path}/{index}")
    elif isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right), path
    else:
        assert left == right, path


def _draw(buffer, seed: int) -> list:
    torch.manual_seed(seed)
    samples = [buffer.sample(6) for _ in range(3)]
    if hasattr(buffer, "sample_without_repeat"):
        samples += [buffer.sample_without_repeat(5) for _ in range(2)]
    if hasattr(buffer, "sample_history"):
        samples.append(buffer.sample_history(4, 3))
    for sample in samples[:3]:
        if hasattr(sample, "priority_indices"):
            buffer.update_priorities(sample.priority_indices, torch.randn(6))
        if hasattr(sample, "indices"):
            buffer.write_back_carry(
                sample.indices,
                {key: torch.randn(*sample.action.shape[:2], *s) for key, s in CARRY_SPEC.items()},
            )
    return samples


def _round_trip(source, target, path):
    save_replay_buffer_file(path, source)
    load_replay_buffer_file(path, target)


def assert_samples_continue(source, target) -> None:
    for seed in range(3):
        assert_tree_equal(_draw(source, seed), _draw(target, seed))


@pytest.mark.parametrize("steps", [PER_ENV - 5, 2 * PER_ENV + 5], ids=["partial", "wrapped"])
@pytest.mark.parametrize("name", BUFFERS)
def test_saved_buffer_continues_sampling(name, steps, tmp_path):
    torch.manual_seed(0)
    source = BUFFERS[name]()
    fill_randomly(source, steps)
    _draw(source, 7)  # non-default priorities, carry and without-replacement cursor

    target = BUFFERS[name]()
    _round_trip(source, target, tmp_path / "replay_buffer.pt")
    assert (target.pos, target.full, target.size) == (source.pos, source.full, source.size)
    assert_samples_continue(source, target)

    for buffer in (source, target):
        torch.manual_seed(1)
        fill_randomly(buffer, 11, first_step=steps)
    assert_samples_continue(source, target)


def _multitask(mmap_dir):
    return MmapMultitaskEpisodeBuffer(
        obs_dim=3, action_dim=2, buffer_size=40, horizon=2, mmap_dir=mmap_dir, sample_device="cpu"
    )


def test_multitask_buffer_continues_sampling(tmp_path):
    source = _multitask(tmp_path / "source")
    for task in range(3):
        source.load_episode(torch.randn(9, 3), torch.randn(9, 2), torch.randn(9), task)
    target = _multitask(tmp_path / "target")
    _round_trip(source, target, tmp_path / "replay_buffer.pt")
    assert_samples_continue(source, target)


def test_final_obs_table_grown_past_its_initial_capacity_is_restored(tmp_path):
    source = BUFFERS["lazy_nstep"]()
    fill_randomly(source, PER_ENV)
    assert source._final_obs.shape[0] > 2
    target = BUFFERS["lazy_nstep"]()
    _round_trip(source, target, tmp_path / "replay_buffer.pt")
    assert target._final_obs.shape == source._final_obs.shape
    assert_samples_continue(source, target)


def test_without_replacement_epoch_continues_after_loading(tmp_path):
    source = BUFFERS["plain"]()
    fill_randomly(source, 10)
    for _ in range(2):
        source.sample_without_repeat(5)
    target = BUFFERS["plain"]()
    _round_trip(source, target, tmp_path / "replay_buffer.pt")
    # 10 of the epoch's 20 transitions are left: no reshuffle, so no RNG use.
    for _ in range(2):
        assert_tree_equal(source.sample_without_repeat(5), target.sample_without_repeat(5))


@pytest.mark.parametrize("name", BUFFERS)
def test_mismatched_files_are_rejected(name):
    source = BUFFERS[name]()
    fill_randomly(source, 10)
    state = replay_buffer_state_dict(source)

    for changed, message in (
        ({"buffer_class": "OtherBuffer"}, "buffer_class mismatch"),
        ({"buffer_size": state["buffer_size"] + 2}, "buffer_size mismatch"),
        ({"num_envs": 3}, "num_envs mismatch"),
        ({"format_version": -1}, "format_version mismatch"),
        ({"actions": state["actions"][..., :1]}, "shape mismatch for 'actions'"),
        ({"rewards": state["rewards"][:1]}, "shape mismatch for 'rewards'"),
        ({"obs": {"state": state["obs"]["state"]}}, "'obs' keys do not match"),
    ):
        with pytest.raises(ValueError, match=message):
            load_replay_buffer_state_dict(BUFFERS[name](), {**state, **changed})


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("strict_sequence", "step_id"),
        ("uniform_sequence", "is_first"),
        ("recurrent_lstm", "hidden_checkpoints_c"),
        ("lazy_nstep", "final_slot_ids"),
        ("plain", "next_obs"),
    ],
)
def test_file_without_required_state_is_rejected(name, key):
    source = BUFFERS[name]()
    fill_randomly(source, 10)
    state = replay_buffer_state_dict(source)
    del state[key]
    with pytest.raises(ValueError, match=f"missing '{key}'"):
        load_replay_buffer_state_dict(BUFFERS[name](), state)


def test_sequence_modes_do_not_load_each_other():
    strict = BUFFERS["strict_sequence"]()
    fill_randomly(strict, 10)
    with pytest.raises(ValueError, match="is_first"):
        load_replay_buffer_state_dict(_uniform(), replay_buffer_state_dict(strict))


OLD_KEYS = {
    "format_version", "buffer_class", "num_envs", "buffer_size", "per_env_buffer_size",
    "pos", "full", "storage_device", "sample_device",
    "obs", "next_obs", "actions", "rewards", "dones", "gamma",
}


def _old_file_state(source) -> dict:
    return {k: v for k, v in replay_buffer_state_dict(source).items() if k in OLD_KEYS}


WINDOWED = ["nstep", "chunked", "memory_efficient"]


@pytest.mark.parametrize("name", WINDOWED)
def test_old_file_without_window_state_is_rejected_by_a_fresh_buffer(name):
    source = BUFFERS[name]()
    fill_randomly(source, 10)
    with pytest.raises(ValueError, match="cannot rebuild"):
        load_replay_buffer_state_dict(BUFFERS[name](), _old_file_state(source))


@pytest.mark.parametrize("name", WINDOWED)
def test_old_file_loads_over_the_dataset_it_was_saved_from(name):
    torch.manual_seed(0)
    source = BUFFERS[name]()
    fill_randomly(source, 10)
    torch.manual_seed(0)
    target = BUFFERS[name]()
    fill_randomly(target, 10)
    with pytest.warns(UserWarning, match="keeps its current values"):
        load_replay_buffer_state_dict(target, _old_file_state(source))
    assert_samples_continue(source, target)


def test_old_file_without_grasp_penalties_loads_with_zero_penalties():
    source = BUFFERS["grasp_penalty"]()
    fill_randomly(source, 10)
    target = BUFFERS["grasp_penalty"]()
    with pytest.warns(UserWarning, match="grasp_penalty"):
        load_replay_buffer_state_dict(target, _old_file_state(source))
    assert torch.equal(target.rewards, source.rewards)
    assert not target.grasp_penalty.any()


def test_subclass_columns_named_base_actions_are_saved():
    class ResidualReplayBuffer(ReplayBuffer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.base_actions = torch.zeros_like(self.actions)
            self.next_base_actions = torch.zeros_like(self.actions)

    source, target = ResidualReplayBuffer(*_args(), **CPU), ResidualReplayBuffer(*_args(), **CPU)
    source.base_actions += 1.0
    source.next_base_actions += 2.0
    state = replay_buffer_state_dict(source)
    load_replay_buffer_state_dict(target, state)
    assert torch.equal(target.base_actions, source.base_actions)
    assert torch.equal(target.next_base_actions, source.next_base_actions)
    del state["base_actions"]
    with pytest.raises(ValueError, match="missing 'base_actions'"):
        load_replay_buffer_state_dict(target, state)
