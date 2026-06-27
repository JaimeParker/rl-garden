"""Unit tests for rl_garden replay buffers.

Runs on CPU so it's safe in CI without a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers import DictReplayBuffer, TensorReplayBuffer
from rl_garden.buffers.nstep_buffer import (
    LazyNextNStepDictReplayBuffer,
    NStepDictReplayBuffer,
)


def test_tensor_replay_buffer_add_and_sample():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    num_envs, buffer_size = 4, 32
    device = torch.device("cpu")

    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=num_envs, buffer_size=buffer_size,
        storage_device=device, sample_device=device,
    )
    assert rb.per_env_buffer_size == buffer_size // num_envs

    for _ in range(3):
        rb.add(
            obs=torch.randn(num_envs, 7),
            next_obs=torch.randn(num_envs, 7),
            action=torch.randn(num_envs, 3),
            reward=torch.randn(num_envs),
            done=torch.zeros(num_envs),
        )

    assert rb.pos == 3 and not rb.full
    assert len(rb) == 3 * num_envs

    batch = rb.sample(batch_size=16)
    assert batch.obs.shape == (16, 7)
    assert batch.next_obs.shape == (16, 7)
    assert batch.actions.shape == (16, 3)
    assert batch.rewards.shape == (16,)
    assert batch.dones.shape == (16,)
    assert batch.obs.device == device


def test_tensor_replay_buffer_wraps_when_full():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    num_envs, buffer_size = 2, 8  # per_env = 4
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=num_envs, buffer_size=buffer_size,
        storage_device="cpu", sample_device="cpu",
    )
    for i in range(5):
        rb.add(
            obs=torch.full((num_envs, 2), float(i)),
            next_obs=torch.full((num_envs, 2), float(i + 1)),
            action=torch.full((num_envs, 1), float(i)),
            reward=torch.full((num_envs,), float(i)),
            done=torch.zeros(num_envs),
        )
    assert rb.full and rb.pos == 1


def test_dict_replay_buffer_add_and_sample():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    num_envs, buffer_size = 2, 8
    device = torch.device("cpu")

    rb = DictReplayBuffer(
        obs_space, act_space, num_envs=num_envs, buffer_size=buffer_size,
        storage_device=device, sample_device=device,
    )

    def make_obs():
        return {
            "rgb": torch.randint(0, 256, (num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(num_envs, 5),
        }

    for _ in range(3):
        rb.add(
            obs=make_obs(),
            next_obs=make_obs(),
            action=torch.randn(num_envs, 4),
            reward=torch.randn(num_envs),
            done=torch.zeros(num_envs),
        )

    batch = rb.sample(batch_size=5)
    assert isinstance(batch.obs, dict)
    assert batch.obs["rgb"].shape == (5, 64, 64, 3)
    assert batch.obs["rgb"].dtype == torch.uint8
    assert batch.obs["state"].shape == (5, 5)
    assert batch.next_obs["rgb"].shape == (5, 64, 64, 3)
    assert batch.actions.shape == (5, 4)
    assert batch.rewards.shape == (5,)
    assert batch.dones.shape == (5,)


def _mmap_obs_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "camera": spaces.Dict(
                {
                    "rgb": spaces.Box(
                        low=0, high=255, shape=(2, 2, 3), dtype=np.uint8
                    )
                }
            ),
            "state": spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            ),
        }
    )


def test_dict_replay_buffer_mmap_keeps_obs_and_next_obs_separate(tmp_path):
    rb = DictReplayBuffer(
        _mmap_obs_space(),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=4,
        storage_device="cpu",
        sample_device="cpu",
        mmap_dir=tmp_path,
    )
    rb.add(
        obs={
            "camera": {
                "rgb": torch.full((1, 2, 2, 3), 1, dtype=torch.uint8)
            },
            "state": torch.tensor([[1.0, 2.0]]),
        },
        next_obs={
            "camera": {
                "rgb": torch.full((1, 2, 2, 3), 9, dtype=torch.uint8)
            },
            "state": torch.tensor([[8.0, 9.0]]),
        },
        action=torch.tensor([[0.25]]),
        reward=torch.tensor([3.0]),
        done=torch.tensor([False]),
    )
    rb.flush()

    assert torch.equal(
        rb.obs["camera"]["rgb"][0, 0],
        torch.full((2, 2, 3), 1, dtype=torch.uint8),
    )
    assert torch.equal(
        rb.next_obs["camera"]["rgb"][0, 0],
        torch.full((2, 2, 3), 9, dtype=torch.uint8),
    )
    assert (tmp_path / "obs" / "camera" / "rgb.bin").is_file()
    assert (tmp_path / "next_obs" / "camera" / "rgb.bin").is_file()


def test_dict_replay_buffer_mmap_open_restores_complete_buffer(tmp_path):
    obs_space = _mmap_obs_space()
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    source = DictReplayBuffer(
        obs_space,
        action_space,
        num_envs=1,
        buffer_size=4,
        storage_device="cpu",
        sample_device="cpu",
        mmap_dir=tmp_path,
    )
    source.add(
        obs={
            "camera": {"rgb": torch.zeros((1, 2, 2, 3), dtype=torch.uint8)},
            "state": torch.tensor([[1.0, 2.0]]),
        },
        next_obs={
            "camera": {"rgb": torch.ones((1, 2, 2, 3), dtype=torch.uint8)},
            "state": torch.tensor([[3.0, 4.0]]),
        },
        action=torch.tensor([[0.5]]),
        reward=torch.tensor([2.0]),
        done=torch.tensor([True]),
    )
    source.flush()

    restored = DictReplayBuffer(
        obs_space,
        action_space,
        num_envs=1,
        buffer_size=4,
        storage_device="cpu",
        sample_device="cpu",
        mmap_dir=tmp_path,
        mmap_mode="open",
    )

    assert restored.pos == 1
    assert not restored.full
    sample = restored._index_batch(torch.tensor([0]), torch.tensor([0]))
    assert torch.equal(sample.obs["state"], torch.tensor([[1.0, 2.0]]))
    assert torch.equal(sample.next_obs["state"], torch.tensor([[3.0, 4.0]]))
    assert torch.equal(sample.actions, torch.tensor([[0.5]]))
    assert torch.equal(sample.rewards, torch.tensor([2.0]))
    assert torch.equal(sample.dones, torch.tensor([1.0]))

    restored.add(
        obs={
            "camera": {"rgb": torch.zeros((1, 2, 2, 3), dtype=torch.uint8)},
            "state": torch.tensor([[5.0, 6.0]]),
        },
        next_obs={
            "camera": {"rgb": torch.ones((1, 2, 2, 3), dtype=torch.uint8)},
            "state": torch.tensor([[7.0, 8.0]]),
        },
        action=torch.tensor([[0.75]]),
        reward=torch.tensor([4.0]),
        done=torch.tensor([False]),
    )
    assert restored.pos == 2


def test_dict_replay_buffer_mmap_rejects_overwrite_and_schema_mismatch(tmp_path):
    obs_space = _mmap_obs_space()
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    DictReplayBuffer(
        obs_space,
        action_space,
        num_envs=1,
        buffer_size=4,
        storage_device="cpu",
        sample_device="cpu",
        mmap_dir=tmp_path,
    )

    with pytest.raises(FileExistsError, match="already contains"):
        DictReplayBuffer(
            obs_space,
            action_space,
            num_envs=1,
            buffer_size=4,
            storage_device="cpu",
            sample_device="cpu",
            mmap_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="manifest does not match"):
        DictReplayBuffer(
            obs_space,
            action_space,
            num_envs=1,
            buffer_size=8,
            storage_device="cpu",
            sample_device="cpu",
            mmap_dir=tmp_path,
            mmap_mode="open",
        )


def test_dict_replay_buffer_mmap_requires_cpu_storage(tmp_path):
    with pytest.raises(ValueError, match="CPU storage"):
        DictReplayBuffer(
            _mmap_obs_space(),
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            num_envs=1,
            buffer_size=4,
            storage_device="cuda",
            sample_device="cpu",
            mmap_dir=tmp_path,
        )


def _nstep_state_space() -> spaces.Dict:
    return spaces.Dict(
        {"state": spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)}
    )


def _add_nstep_transition(
    rb: NStepDictReplayBuffer,
    step: int,
    reward: float,
    done: bool = False,
    episode_end: bool | None = None,
) -> None:
    obs = {"state": torch.tensor([[float(step)]])}
    next_obs = {"state": torch.tensor([[float(step + 1)]])}
    rb.add(
        obs=obs,
        next_obs=next_obs,
        action=torch.zeros(1, 1),
        reward=torch.tensor([reward]),
        done=torch.tensor([done]),
        episode_end=torch.tensor(
            [done if episode_end is None else episode_end]
        ),
    )


def _make_nstep_buffer_pair(buffer_size: int = 8):
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    common = dict(
        observation_space=_nstep_state_space(),
        action_space=action_space,
        num_envs=1,
        buffer_size=buffer_size,
        nstep=3,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    return NStepDictReplayBuffer(**common), LazyNextNStepDictReplayBuffer(**common)


def _assert_nstep_transition_equal(
    left: NStepDictReplayBuffer,
    right: NStepDictReplayBuffer,
    t: int,
) -> None:
    assert left._valid_nstep(t, 0)
    assert right._valid_nstep(t, 0)
    left_reward, left_discount, left_next_obs = left._compute_nstep(t, 0)
    right_reward, right_discount, right_next_obs = right._compute_nstep(t, 0)
    assert torch.allclose(left_reward, right_reward)
    assert torch.allclose(left_discount, right_discount)
    assert torch.equal(left_next_obs["state"], right_next_obs["state"])


def test_lazy_nstep_reconstructs_bootstrap_next_obs():
    regular, lazy = _make_nstep_buffer_pair()
    for step in range(4):
        _add_nstep_transition(regular, step, float(step + 1))
        _add_nstep_transition(lazy, step, float(step + 1))

    _assert_nstep_transition_equal(regular, lazy, 0)
    _, _, next_obs = lazy._compute_nstep(0, 0)
    assert torch.equal(next_obs["state"], torch.tensor([3.0]))
    assert lazy.next_obs is None
    assert lazy._final_slot_ids.max().item() == -1


def test_lazy_nstep_uses_sparse_final_obs_at_episode_boundary():
    regular, lazy = _make_nstep_buffer_pair()
    _add_nstep_transition(regular, 0, 1.0)
    _add_nstep_transition(lazy, 0, 1.0)
    _add_nstep_transition(regular, 1, 2.0, episode_end=True)
    _add_nstep_transition(lazy, 1, 2.0, episode_end=True)
    _add_nstep_transition(regular, 2, 100.0)
    _add_nstep_transition(lazy, 2, 100.0)

    _assert_nstep_transition_equal(regular, lazy, 0)
    _, discount, next_obs = lazy._compute_nstep(0, 0)
    assert torch.allclose(discount, torch.tensor(0.9**2))
    assert torch.equal(next_obs["state"], torch.tensor([2.0]))
    assert lazy._final_slot_ids[1, 0].item() >= 0


def test_lazy_nstep_requires_bootstrap_obs_to_be_written():
    _, lazy = _make_nstep_buffer_pair()
    for step in range(3):
        _add_nstep_transition(lazy, step, float(step + 1))

    assert not lazy._valid_nstep(0, 0)


def test_lazy_nstep_sample_matches_regular_vectorized_output():
    regular, lazy = _make_nstep_buffer_pair()
    for step in range(6):
        episode_end = step == 3
        _add_nstep_transition(
            regular, step, float(step + 1), episode_end=episode_end
        )
        _add_nstep_transition(lazy, step, float(step + 1), episode_end=episode_end)

    batch_inds = torch.tensor([0, 1, 3])
    env_inds = torch.zeros(3, dtype=torch.long)
    regular_rewards, regular_discounts, regular_next = regular._compute_nstep_batch(
        batch_inds, env_inds
    )
    lazy_rewards, lazy_discounts, lazy_next = lazy._compute_nstep_batch(
        batch_inds, env_inds
    )
    assert torch.allclose(regular_rewards, lazy_rewards)
    assert torch.allclose(regular_discounts, lazy_discounts)
    assert torch.equal(regular_next["state"], lazy_next["state"])


def test_nstep_buffer_returns_accumulated_reward_and_discount():
    rb = NStepDictReplayBuffer(
        _nstep_state_space(),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=5,
        nstep=3,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    _add_nstep_transition(rb, 0, 1.0)
    _add_nstep_transition(rb, 1, 2.0)
    _add_nstep_transition(rb, 2, 3.0)

    assert rb._valid_nstep(0, 0)
    reward, discount, next_obs = rb._compute_nstep(0, 0)

    assert torch.allclose(reward, torch.tensor(1.0 + 0.9 * 2.0 + 0.9**2 * 3.0))
    assert torch.allclose(discount, torch.tensor(0.9**3))
    assert torch.equal(next_obs["state"], torch.tensor([3.0]))

    batch = rb.sample(1)
    assert hasattr(batch, "discounts")
    assert batch.discounts.shape == (1,)


def test_nstep_buffer_stops_at_terminal_transition():
    rb = NStepDictReplayBuffer(
        _nstep_state_space(),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=5,
        nstep=3,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    _add_nstep_transition(rb, 0, 1.0)
    _add_nstep_transition(rb, 1, 2.0, done=True)
    _add_nstep_transition(rb, 2, 100.0)

    assert rb._valid_nstep(0, 0)
    reward, discount, next_obs = rb._compute_nstep(0, 0)

    assert torch.allclose(reward, torch.tensor(1.0 + 0.9 * 2.0))
    assert torch.allclose(discount, torch.tensor(0.0))
    assert torch.equal(next_obs["state"], torch.tensor([2.0]))


def test_nstep_buffer_truncation_stops_window_but_keeps_bootstrap_discount():
    rb = NStepDictReplayBuffer(
        _nstep_state_space(),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=5,
        nstep=3,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    _add_nstep_transition(rb, 0, 1.0)
    _add_nstep_transition(rb, 1, 2.0, done=False, episode_end=True)
    _add_nstep_transition(rb, 2, 100.0)

    assert rb._valid_nstep(0, 0)
    reward, discount, next_obs = rb._compute_nstep(0, 0)

    assert torch.allclose(reward, torch.tensor(1.0 + 0.9 * 2.0))
    assert torch.allclose(discount, torch.tensor(0.9**2))
    assert torch.equal(next_obs["state"], torch.tensor([2.0]))


def test_nstep_buffer_does_not_cross_episode_when_bootstrap_continues():
    rb = NStepDictReplayBuffer(
        _nstep_state_space(),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=5,
        nstep=3,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    _add_nstep_transition(rb, 0, 1.0, episode_end=True)
    _add_nstep_transition(rb, 1, 100.0)
    _add_nstep_transition(rb, 2, 100.0)

    reward, discount, next_obs = rb._compute_nstep(0, 0)

    assert torch.allclose(reward, torch.tensor(1.0))
    assert torch.allclose(discount, torch.tensor(0.9))
    assert torch.equal(next_obs["state"], torch.tensor([1.0]))


def test_nstep_buffer_rejects_ring_wrap_without_temporal_contiguity():
    rb = NStepDictReplayBuffer(
        _nstep_state_space(),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=5,
        nstep=3,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    for step in range(7):
        _add_nstep_transition(rb, step, float(step))

    assert rb.full
    assert not rb._valid_nstep(1, 0)  # step id 6 followed by overwritten step id 2
    assert rb._valid_nstep(4, 0)  # step ids 4, 5, 6 are contiguous across ring end


def test_nstep_vectorized_helpers_match_scalar_semantics():
    rb = NStepDictReplayBuffer(
        _nstep_state_space(),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=7,
        nstep=3,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    _add_nstep_transition(rb, 0, 1.0)
    _add_nstep_transition(rb, 1, 2.0, done=True)
    _add_nstep_transition(rb, 2, 3.0)
    _add_nstep_transition(rb, 3, 4.0, episode_end=True)
    _add_nstep_transition(rb, 4, 5.0)
    _add_nstep_transition(rb, 5, 6.0)

    batch_inds = torch.arange(6)
    env_inds = torch.zeros(6, dtype=torch.long)
    valid = rb._valid_nstep_batch(batch_inds, env_inds)
    expected_valid = torch.tensor(
        [rb._valid_nstep(t, 0) for t in range(6)]
    )
    assert torch.equal(valid, expected_valid)

    selected = batch_inds[valid]
    selected_envs = env_inds[valid]
    rewards, discounts, next_obs = rb._compute_nstep_batch(
        selected, selected_envs
    )
    for i, t in enumerate(selected.tolist()):
        reward, discount, scalar_next_obs = rb._compute_nstep(t, 0)
        assert torch.allclose(rewards[i], reward)
        assert torch.allclose(discounts[i], discount)
        assert torch.equal(next_obs["state"][i], scalar_next_obs["state"])


def test_nstep_vectorized_helpers_match_scalar_across_ring_wrap():
    rb = NStepDictReplayBuffer(
        _nstep_state_space(),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        num_envs=1,
        buffer_size=5,
        nstep=3,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    for step in range(7):
        _add_nstep_transition(rb, step, float(step))

    batch_inds = torch.arange(rb.per_env_buffer_size)
    env_inds = torch.zeros(rb.per_env_buffer_size, dtype=torch.long)
    valid = rb._valid_nstep_batch(batch_inds, env_inds)
    expected_valid = torch.tensor(
        [
            rb._valid_nstep(t, 0)
            for t in range(rb.per_env_buffer_size)
        ]
    )
    assert torch.equal(valid, expected_valid)

    selected = batch_inds[valid]
    selected_envs = env_inds[valid]
    rewards, discounts, next_obs = rb._compute_nstep_batch(
        selected, selected_envs
    )
    for i, t in enumerate(selected.tolist()):
        reward, discount, scalar_next_obs = rb._compute_nstep(t, 0)
        assert torch.allclose(rewards[i], reward)
        assert torch.allclose(discounts[i], discount)
        assert torch.equal(next_obs["state"][i], scalar_next_obs["state"])


def test_nstep_mmap_open_restores_temporal_tracking(tmp_path):
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    source = NStepDictReplayBuffer(
        _nstep_state_space(),
        action_space,
        num_envs=1,
        buffer_size=5,
        nstep=3,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
        mmap_dir=tmp_path,
    )
    _add_nstep_transition(source, 0, 1.0)
    _add_nstep_transition(source, 1, 2.0, episode_end=True)
    _add_nstep_transition(source, 2, 3.0)
    source.flush()

    restored = NStepDictReplayBuffer(
        _nstep_state_space(),
        action_space,
        num_envs=1,
        buffer_size=5,
        nstep=3,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
        mmap_dir=tmp_path,
        mmap_mode="open",
    )

    assert restored.pos == 3
    assert restored._valid_nstep(0, 0)
    reward, discount, next_obs = restored._compute_nstep(0, 0)
    assert torch.allclose(reward, torch.tensor(1.0 + 0.9 * 2.0))
    assert torch.allclose(discount, torch.tensor(0.9**2))
    assert torch.equal(next_obs["state"], torch.tensor([2.0]))
    assert torch.equal(restored._current_ep_id, torch.tensor([1]))
    assert torch.equal(restored._current_step_id, torch.tensor([3]))


# ----------------------------------------------------------------------------
# sample_without_repeat
# ----------------------------------------------------------------------------


def _fill_tensor_buffer(buf, num_steps):
    obs_dim = buf.obs.shape[-1]
    act_dim = buf.actions.shape[-1]
    n = buf.num_envs
    for step in range(num_steps):
        buf.add(
            obs=torch.full((n, obs_dim), float(step)),
            next_obs=torch.full((n, obs_dim), float(step + 1)),
            action=torch.zeros(n, act_dim),
            reward=torch.full((n,), float(step)),
            done=torch.zeros(n),
        )


def test_sample_without_repeat_no_duplicates_within_epoch():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=4, buffer_size=20,
        storage_device="cpu", sample_device="cpu",
    )
    _fill_tensor_buffer(rb, num_steps=5)  # 5 * 4 = 20 transitions
def test_sample_without_repeat_visits_all_indices_in_epoch():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=4, buffer_size=20,
        storage_device="cpu", sample_device="cpu",
    )
    # Fill with per-env distinct values so we can identify each transition
    for step in range(5):
        rb.add(
            obs=torch.tensor([[step, e] for e in range(4)], dtype=torch.float32),
            next_obs=torch.zeros(4, 2),
            action=torch.zeros(4, 1),
            reward=torch.zeros(4),
            done=torch.zeros(4),
        )
    seen = set()
    epoch = rb.epoch_size
    batch_size = 5
    for _ in range(epoch // batch_size):
        sample = rb.sample_without_repeat(batch_size)
        for o in sample.obs:
            seen.add((int(o[0].item()), int(o[1].item())))
    assert len(seen) == 20  # all distinct (t, env) pairs visited


def test_sample_without_repeat_reshuffles_after_exhaustion():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=2, buffer_size=8,
        storage_device="cpu", sample_device="cpu",
    )
    _fill_tensor_buffer(rb, num_steps=4)  # 8 transitions
    # Exhaust epoch: 8/2 = 4 batches
    for _ in range(4):
        rb.sample_without_repeat(2)
    # Next sample triggers reshuffle (not a new add); should still return valid sample
    sample = rb.sample_without_repeat(2)
    assert sample.obs.shape == (2, 2)


def test_sample_without_repeat_invalidates_after_pos_change():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=2, buffer_size=20,
        storage_device="cpu", sample_device="cpu",
    )
    _fill_tensor_buffer(rb, num_steps=5)  # 10 transitions
    sample1 = rb.sample_without_repeat(2)
    pos_before = rb.pos
    # Add more data → pos changes → permutation should rebuild on next call
    _fill_tensor_buffer(rb, num_steps=2)
    assert rb.pos != pos_before
    sample2 = rb.sample_without_repeat(2)
    assert sample2.obs.shape == (2, 2)


def test_sample_without_repeat_empty_buffer_raises():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    rb = TensorReplayBuffer(
        obs_space, act_space, num_envs=2, buffer_size=10,
        storage_device="cpu", sample_device="cpu",
    )
    import pytest
    with pytest.raises(RuntimeError, match="empty"):
        rb.sample_without_repeat(1)


def test_dict_buffer_sample_without_repeat():
    obs_space = spaces.Dict({
        "state": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
    })
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    rb = DictReplayBuffer(
        obs_space, act_space, num_envs=2, buffer_size=8,
        storage_device="cpu", sample_device="cpu",
    )
    for step in range(4):
        rb.add(
            obs={"state": torch.full((2, 3), float(step))},
            next_obs={"state": torch.zeros(2, 3)},
            action=torch.zeros(2, 2),
            reward=torch.zeros(2),
            done=torch.zeros(2),
        )
    sample = rb.sample_without_repeat(4)
    assert isinstance(sample.obs, dict)
    assert sample.obs["state"].shape == (4, 3)
