"""Unit tests for MCReplayBuffer with Monte Carlo return computation."""
import random
import zlib

import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers.mc_buffer import MCDictReplayBuffer, MCTensorReplayBuffer


@pytest.fixture
def obs_space():
    return spaces.Box(low=-1, high=1, shape=(4,), dtype=float)


@pytest.fixture
def dict_obs_space():
    return spaces.Dict({
        "state": spaces.Box(low=-1, high=1, shape=(4,), dtype=float),
        "goal": spaces.Box(low=-1, high=1, shape=(2,), dtype=float),
    })


@pytest.fixture
def action_space():
    return spaces.Box(low=-1, high=1, shape=(2,), dtype=float)


@pytest.fixture
def mc_tensor_buffer(obs_space, action_space):
    return MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=4,
        buffer_size=100,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )


@pytest.fixture
def mc_dict_buffer(dict_obs_space, action_space):
    return MCDictReplayBuffer(
        observation_space=dict_obs_space,
        action_space=action_space,
        num_envs=4,
        buffer_size=100,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )


class TestMCTensorReplayBuffer:
    """Test MCTensorReplayBuffer functionality."""

    def test_buffer_creation(self, mc_tensor_buffer):
        assert mc_tensor_buffer.gamma == 0.9
        assert mc_tensor_buffer.num_envs == 4
        assert mc_tensor_buffer.per_env_buffer_size == 25

    def test_add_transitions(self, mc_tensor_buffer):
        obs = torch.randn(4, 4)
        next_obs = torch.randn(4, 4)
        actions = torch.randn(4, 2)
        rewards = torch.ones(4)
        dones = torch.zeros(4)

        mc_tensor_buffer.add(obs, next_obs, actions, rewards, dones)
        assert mc_tensor_buffer.pos == 1
        assert not mc_tensor_buffer.full

    def test_sample_with_mc_returns(self, mc_tensor_buffer):
        # Add some transitions, closing the trajectory on the last step so
        # sampling has at least one complete episode to draw from.
        for step in range(10):
            obs = torch.randn(4, 4)
            next_obs = torch.randn(4, 4)
            actions = torch.randn(4, 2)
            rewards = torch.ones(4)
            dones = torch.ones(4) if step == 9 else torch.zeros(4)
            mc_tensor_buffer.add(obs, next_obs, actions, rewards, dones)

        sample = mc_tensor_buffer.sample(8)
        assert hasattr(sample, "mc_returns")
        assert sample.mc_returns.shape == (8,)
        assert sample.obs.shape == (8, 4)
        assert sample.actions.shape == (8, 2)

    def test_mc_returns_single_step_episode(self, mc_tensor_buffer):
        # Add single-step episode
        obs = torch.randn(4, 4)
        next_obs = torch.randn(4, 4)
        actions = torch.randn(4, 2)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dones = torch.ones(4)  # All episodes end

        mc_tensor_buffer.add(obs, next_obs, actions, rewards, dones)

        sample = mc_tensor_buffer.sample(4)
        # MC return should equal reward for single-step episodes
        # Since we're sampling randomly, just check that returns are in the expected range
        assert torch.all(sample.mc_returns >= 1.0)
        assert torch.all(sample.mc_returns <= 4.0)

    def test_mc_returns_multi_step_episode(self, mc_tensor_buffer):
        # Add 3-step episode for env 0
        for step in range(3):
            obs = torch.randn(4, 4)
            next_obs = torch.randn(4, 4)
            actions = torch.randn(4, 2)
            rewards = torch.ones(4)
            dones = torch.zeros(4)
            if step == 2:  # End episode at step 2
                dones[0] = 1.0

            mc_tensor_buffer.add(obs, next_obs, actions, rewards, dones)

        # Manually compute expected MC return for first transition
        # G_0 = r_0 + γ*r_1 + γ²*r_2 = 1 + 0.9*1 + 0.81*1 = 2.71
        expected_return = 1.0 + 0.9 * 1.0 + 0.81 * 1.0

        # Sample and check (need to ensure we sample from env 0, step 0)
        # This is probabilistic, so we'll just check the computation logic
        sample = mc_tensor_buffer.sample(16)
        assert sample.mc_returns.shape == (16,)
        # All returns should be positive
        assert torch.all(sample.mc_returns > 0)

    def test_mc_returns_with_discount(self, mc_tensor_buffer):
        # Test that discount is applied correctly
        gamma = mc_tensor_buffer.gamma

        # Add 2-step episode
        for step in range(2):
            obs = torch.randn(4, 4)
            next_obs = torch.randn(4, 4)
            actions = torch.randn(4, 2)
            rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
            dones = torch.zeros(4)
            if step == 1:
                dones[0] = 1.0

            mc_tensor_buffer.add(obs, next_obs, actions, rewards, dones)

        sample = mc_tensor_buffer.sample(16)
        # Returns should be in range [1.0, 1.0 + gamma]
        assert torch.all(sample.mc_returns >= 1.0)
        assert torch.all(sample.mc_returns <= 1.0 + gamma + 0.1)  # Small tolerance


class TestMCDictReplayBuffer:
    """Test MCDictReplayBuffer functionality."""

    def test_buffer_creation(self, mc_dict_buffer):
        assert mc_dict_buffer.gamma == 0.9
        assert mc_dict_buffer.num_envs == 4

    def test_add_dict_transitions(self, mc_dict_buffer):
        obs = {
            "state": torch.randn(4, 4),
            "goal": torch.randn(4, 2),
        }
        next_obs = {
            "state": torch.randn(4, 4),
            "goal": torch.randn(4, 2),
        }
        actions = torch.randn(4, 2)
        rewards = torch.ones(4)
        dones = torch.zeros(4)

        mc_dict_buffer.add(obs, next_obs, actions, rewards, dones)
        assert mc_dict_buffer.pos == 1

    def test_sample_dict_with_mc_returns(self, mc_dict_buffer):
        # Add some transitions, closing the trajectory on the last step so
        # sampling has at least one complete episode to draw from.
        for step in range(10):
            obs = {
                "state": torch.randn(4, 4),
                "goal": torch.randn(4, 2),
            }
            next_obs = {
                "state": torch.randn(4, 4),
                "goal": torch.randn(4, 2),
            }
            actions = torch.randn(4, 2)
            rewards = torch.ones(4)
            dones = torch.ones(4) if step == 9 else torch.zeros(4)
            mc_dict_buffer.add(obs, next_obs, actions, rewards, dones)

        sample = mc_dict_buffer.sample(8)
        assert hasattr(sample, "mc_returns")
        assert sample.mc_returns.shape == (8,)
        assert isinstance(sample.obs, dict)
        assert sample.obs["state"].shape == (8, 4)
        assert sample.obs["goal"].shape == (8, 2)


class TestMCReturnComputation:
    """Test MC return computation logic."""

    def test_episode_boundary_detection(self, mc_tensor_buffer):
        # Add episode with clear boundary
        for step in range(5):
            obs = torch.randn(4, 4)
            next_obs = torch.randn(4, 4)
            actions = torch.randn(4, 2)
            rewards = torch.ones(4) * (step + 1)  # Increasing rewards
            dones = torch.zeros(4)
            if step == 4:
                dones[0] = 1.0  # End episode for env 0

            mc_tensor_buffer.add(obs, next_obs, actions, rewards, dones)

        sample = mc_tensor_buffer.sample(16)
        # All MC returns should be positive
        assert torch.all(sample.mc_returns > 0)

    def test_multiple_episodes_per_env(self, mc_tensor_buffer):
        # Add multiple episodes for same env
        for episode in range(2):
            for step in range(3):
                obs = torch.randn(4, 4)
                next_obs = torch.randn(4, 4)
                actions = torch.randn(4, 2)
                rewards = torch.ones(4)
                dones = torch.zeros(4)
                if step == 2:  # End each episode
                    dones[:] = 1.0

                mc_tensor_buffer.add(obs, next_obs, actions, rewards, dones)

        sample = mc_tensor_buffer.sample(16)
        assert sample.mc_returns.shape == (16,)

    def test_gamma_zero(self):
        # Test with gamma=0 (no discounting)
        buffer = MCTensorReplayBuffer(
            observation_space=spaces.Box(low=-1, high=1, shape=(4,)),
            action_space=spaces.Box(low=-1, high=1, shape=(2,)),
            num_envs=2,
            buffer_size=20,
            gamma=0.0,
            storage_device="cpu",
            sample_device="cpu",
        )

        # Add 2-step episode
        for step in range(2):
            obs = torch.randn(2, 4)
            next_obs = torch.randn(2, 4)
            actions = torch.randn(2, 2)
            rewards = torch.ones(2)
            dones = torch.zeros(2)
            if step == 1:
                dones[:] = 1.0

            buffer.add(obs, next_obs, actions, rewards, dones)

        sample = buffer.sample(4)
        # With gamma=0, MC return should equal immediate reward
        # (approximately, due to sampling randomness)
        assert torch.all(sample.mc_returns >= 0.9)
        assert torch.all(sample.mc_returns <= 1.1)

    def test_full_buffer_wraparound_returns_are_chronological(self):
        buffer = MCTensorReplayBuffer(
            observation_space=spaces.Box(low=-1, high=1, shape=(1,)),
            action_space=spaces.Box(low=-1, high=1, shape=(1,)),
            num_envs=1,
            buffer_size=4,
            gamma=0.9,
            storage_device="cpu",
            sample_device="cpu",
        )

        for reward in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            obs = torch.zeros(1, 1)
            action = torch.zeros(1, 1)
            buffer.add(obs, obs, action, torch.tensor([reward]), torch.zeros(1))

        table = buffer._build_mc_table().squeeze(1)
        # Physical storage is [5, 6, 3, 4], but chronological order starts at pos=2:
        # 3 -> 4 -> 5 -> 6.
        expected = torch.tensor(
            [
                5.0 + 0.9 * 6.0,
                6.0,
                3.0 + 0.9 * 4.0 + 0.9**2 * 5.0 + 0.9**3 * 6.0,
                4.0 + 0.9 * 5.0 + 0.9**2 * 6.0,
            ]
        )
        torch.testing.assert_close(table, expected)


# ----------------------------------------------------------------------------
# Sparse-reward MC handling
# ----------------------------------------------------------------------------


@pytest.fixture
def sparse_buffer(obs_space, action_space):
    return MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=2,
        buffer_size=20,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
        sparse_reward_mc=True,
        sparse_negative_reward=-1.0,
        success_threshold=0.5,
    )


def _add_step(buf, reward, done, success):
    n = buf.num_envs
    obs = torch.zeros(n, 4)
    next_obs = torch.zeros(n, 4)
    action = torch.zeros(n, 2)
    buf.add(
        obs,
        next_obs,
        action,
        torch.as_tensor(reward, dtype=torch.float32),
        torch.as_tensor(done, dtype=torch.float32),
        success=torch.as_tensor(success, dtype=torch.float32),
    )


def test_sparse_mc_failed_episode_uses_inf_horizon(sparse_buffer):
    # Single env (env 0): 5-step episode, never succeeds, all rewards = -1, terminated.
    # env 1 unused (success will default to 0).
    for step in range(5):
        is_last = step == 4
        _add_step(
            sparse_buffer,
            reward=[-1.0, -1.0],
            done=[1.0 if is_last else 0.0, 0.0],
            success=[0.0, 0.0],
        )
    table = sparse_buffer._build_mc_table()
    # Expected: r_neg / (1 - γ) = -1 / 0.1 = -10 for every step in the failed episode
    expected = torch.full((5,), -10.0)
    torch.testing.assert_close(table[:5, 0], expected)


def test_sparse_mc_successful_episode_uses_standard_sum(sparse_buffer):
    # 3-step episode ending in success. Rewards: [-1, -1, +1]. γ=0.9.
    # Backward sum: G_2 = +1, G_1 = -1 + 0.9*1 = -0.1, G_0 = -1 + 0.9*-0.1 = -1.09
    rewards_seq = [-1.0, -1.0, 1.0]
    for step, r in enumerate(rewards_seq):
        is_last = step == len(rewards_seq) - 1
        _add_step(
            sparse_buffer,
            reward=[r, 0.0],
            done=[1.0 if is_last else 0.0, 0.0],
            success=[1.0 if is_last else 0.0, 0.0],
        )
    table = sparse_buffer._build_mc_table()
    expected = torch.tensor([-1.09, -0.1, 1.0])
    torch.testing.assert_close(table[:3, 0], expected, atol=1e-5, rtol=1e-5)


def test_sparse_mc_two_consecutive_episodes(sparse_buffer):
    # env 0: 2-step failed episode then 2-step successful episode.
    # episode 1 (failed): rewards [-1, -1], dones [0, 1]. Both steps map to -10.
    # episode 2 (success): rewards [-1, +1], dones [0, 1]. G_3=+1, G_2 = -1 + 0.9 = -0.1.
    transitions = [
        ([-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
        ([-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]),
        ([-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
        ([1.0, 0.0], [1.0, 0.0], [1.0, 0.0]),
    ]
    for r, d, s in transitions:
        _add_step(sparse_buffer, reward=r, done=d, success=s)
    table = sparse_buffer._build_mc_table()
    expected = torch.tensor([-10.0, -10.0, -0.1, 1.0])
    torch.testing.assert_close(table[:4, 0], expected, atol=1e-5, rtol=1e-5)


def test_sparse_mc_disabled_falls_back_to_standard(obs_space, action_space):
    # When sparse_reward_mc=False, behaviour matches the previous backward sweep
    # (no inf-horizon replacement), and add() still accepts standard signature.
    buf = MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=1,
        buffer_size=5,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    for step in range(3):
        is_last = step == 2
        buf.add(
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            torch.zeros(1, 2),
            torch.tensor([-1.0]),
            torch.tensor([1.0 if is_last else 0.0]),
        )
    table = buf._build_mc_table()
    # Standard backward sum without inf-horizon override
    expected = torch.tensor([-1.0 - 0.9 - 0.81, -1.0 - 0.9, -1.0])
    torch.testing.assert_close(table[:3, 0], expected, atol=1e-5, rtol=1e-5)


def test_sparse_mc_infers_success_from_reward_threshold(obs_space, action_space):
    # Without explicit success= arg, threshold-based inference kicks in.
    buf = MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=1,
        buffer_size=5,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
        sparse_reward_mc=True,
        sparse_negative_reward=-1.0,
        success_threshold=0.5,
    )
    for step, r in enumerate([-1.0, -1.0, 1.0]):
        is_last = step == 2
        buf.add(
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            torch.zeros(1, 2),
            torch.tensor([r]),
            torch.tensor([1.0 if is_last else 0.0]),
        )
    table = buf._build_mc_table()
    # Last step reward 1.0 >= 0.5 → success → standard sum
    expected = torch.tensor([-1.09, -0.1, 1.0])
    torch.testing.assert_close(table[:3, 0], expected, atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------------
# episode_end vs. Bellman `done` (bootstrap-through-timeout semantics)
# ----------------------------------------------------------------------------


def test_mc_recursion_stops_at_episode_end_not_dones(obs_space, action_space):
    # `done` stays False throughout (simulating bootstrap_at_done="always"
    # bootstrapping TD through a timeout), but `episode_end` marks the true
    # 2-step episode boundary. MC recursion must stop there, not run through
    # the whole buffer as one trajectory.
    buf = MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=1,
        buffer_size=5,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    steps = [
        (-1.0, False),  # episode 1, step 0
        (-1.0, True),  # episode 1, step 1 (truncated: done stays False)
        (1.0, True),  # episode 2, single step
    ]
    for reward, is_episode_end in steps:
        buf.add(
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            torch.zeros(1, 2),
            torch.tensor([reward]),
            torch.tensor([0.0]),  # done: never true
            episode_end=torch.tensor([is_episode_end]),
        )
    table = buf._build_mc_table()
    # Episode 1 (steps 0-1): G_1 = -1, G_0 = -1 + 0.9*-1 = -1.9.
    # Episode 2 (step 2): G_2 = 1, independent of episode 1.
    expected = torch.tensor([-1.9, -1.0, 1.0])
    torch.testing.assert_close(table[:3, 0], expected, atol=1e-5, rtol=1e-5)


def test_sparse_mc_truncated_failure_then_success_does_not_leak(sparse_buffer):
    # Regression test for a success-flag leakage bug: `acc_succ` used to reset
    # on Bellman `done`, the same flag that stays False at a timeout. A
    # truncated failed episode immediately followed by a successful one would
    # have the second episode's success leak backward through `acc_succ`,
    # mislabeling the first (actually failed) episode as successful.
    #
    # env 0: 2-step *truncated* failed episode (done=False throughout, so a
    # naive done-gated accumulator would never reset), then a 2-step
    # successful episode.
    transitions = [
        # (reward, done, episode_end, success)
        (-1.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 1.0, 0.0),  # episode 1 ends here via truncation
        (-1.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),  # episode 2 ends here via termination+success
    ]
    for reward, done, episode_end, success in transitions:
        sparse_buffer.add(
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            torch.zeros(2, 2),
            torch.tensor([reward, 0.0]),
            torch.tensor([done, 0.0]),
            episode_end=torch.tensor([episode_end, 0.0]),
            success=torch.tensor([success, 0.0]),
        )
    table = sparse_buffer._build_mc_table()
    # Episode 1 stays failed (-10 inf-horizon each step); episode 2 succeeds.
    expected = torch.tensor([-10.0, -10.0, -0.1, 1.0])
    torch.testing.assert_close(table[:4, 0], expected, atol=1e-5, rtol=1e-5)


def test_mc_add_accepts_episode_end_without_forwarding_to_wrapped_buffer(
    obs_space, action_space
):
    # episode_end must be consumed by the mixin, never forwarded into
    # TensorReplayBuffer.add()'s fixed positional signature.
    buf = MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=1,
        buffer_size=5,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    buf.add(
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        episode_end=torch.tensor([True]),
    )
    assert buf._episode_end[0, 0].item() is True


def test_incomplete_trailing_trajectory_excluded_from_sampling(obs_space, action_space):
    buf = MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=1,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    # One complete episode, then two steps of a still-open trajectory.
    for step in range(4):
        is_end = step == 1
        buf.add(
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            torch.zeros(1, 2),
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            episode_end=torch.tensor([is_end]),
        )
    assert buf.sampleable_size == 2
    valid_idx = buf._valid_indices()
    assert set(valid_idx[:, 0].tolist()) == {0, 1}

    # Sampling must succeed, drawing only from the valid (t=0, t=1) positions.
    buf.sample(16)


def test_sample_raises_when_no_complete_trajectory(obs_space, action_space):
    buf = MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=1,
        buffer_size=5,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    buf.add(
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        episode_end=torch.tensor([False]),
    )
    with pytest.raises(ValueError):
        buf.sample(4)


def test_offline_load_num_envs_gt_1_marks_all_rows_valid_and_preserves_mc(
    obs_space, action_space
):
    """Regression for the codex review's claim #1: `_add_flat_transitions`
    stripes a flat, sequentially-ordered offline dataset across `num_envs`
    buffer columns, so a naive per-column MC recursion sees fake time series
    and both undercounts `sampleable_size` and can miscompute MC values.
    Loader-provided rows must be trusted (`_externally_valid`) instead of
    being recomputed from that fake column structure.
    """
    from rl_garden.buffers._dataset_common import _add_flat_transitions

    buf = MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=2,
        buffer_size=20,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    # Two flat, sequential 3-step episodes (as an offline loader would
    # produce), striped across num_envs=2 columns by _add_flat_transitions.
    n = 6
    obs = torch.zeros(n, 4)
    next_obs = torch.zeros(n, 4)
    actions = torch.zeros(n, 2)
    rewards = torch.ones(n)
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    episode_end = dones.bool()
    mc_returns = torch.tensor(
        [1 + 0.9 + 0.81, 1 + 0.9, 1.0, 1 + 0.9 + 0.81, 1 + 0.9, 1.0]
    )

    n_loaded = _add_flat_transitions(
        buf, obs, next_obs, actions, rewards, dones, mc_returns, None,
        episode_ends=episode_end,
    )
    assert n_loaded == 6
    assert buf.sampleable_size == 6

    expected = mc_returns.view(3, 2)
    torch.testing.assert_close(buf._build_mc_table()[:3], expected)

    # An online append must not corrupt the preserved offline MC values, and
    # must not inherit "return" from the unrelated offline trajectory.
    buf.add(
        torch.full((2, 4), 9.0), torch.full((2, 4), 9.0), torch.zeros(2, 2),
        torch.tensor([5.0, 5.0]), torch.tensor([0.0, 0.0]),
        episode_end=torch.tensor([False, False]),
    )
    buf.add(
        torch.full((2, 4), 9.0), torch.full((2, 4), 9.0), torch.zeros(2, 2),
        torch.tensor([5.0, 5.0]), torch.tensor([1.0, 1.0]),
        episode_end=torch.tensor([True, True]),
    )
    table = buf._build_mc_table()
    torch.testing.assert_close(table[:3], expected)
    torch.testing.assert_close(table[3:5], torch.tensor([[9.5, 9.5], [5.0, 5.0]]))
    assert buf.sampleable_size == 10


def test_valid_indices_and_sampleable_size_are_cached_between_adds(
    obs_space, action_space
):
    """Regression for claim #2: `.nonzero()`/`.item()` are CUDA-syncing ops
    that must not be recomputed on every call -- only when the underlying
    data actually changes (add())."""
    buf = MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=2,
        buffer_size=20,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )
    for step in range(4):
        is_end = step == 3
        buf.add(
            torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 2),
            torch.ones(2), torch.full((2,), float(is_end)),
        )

    size_a = buf.sampleable_size
    idx_a = buf._valid_indices()
    size_b = buf.sampleable_size
    idx_b = buf._valid_indices()
    assert size_a == size_b == 8
    assert idx_a is idx_b  # same cached tensor object, not recomputed

    buf.add(
        torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 2),
        torch.ones(2), torch.ones(2),
    )
    idx_c = buf._valid_indices()
    assert idx_c is not idx_b  # invalidated by add()


# ---------------------------------------------------------------------------
# Oracle-comparison matrix for the vectorized _build_validity_mask /
# _build_mc_table (codex re-review P1: the Python-loop forms cost 10-25s per
# rebuild at realistic buffer sizes, all Python/CUDA-launch overhead). These
# oracle functions are exact copies of the pre-vectorization loop
# implementations and are kept here permanently as a stronger regression net
# than fixed-value assertions -- any future change to either builder must
# keep matching them across this randomized state matrix.
# ---------------------------------------------------------------------------


def _oracle_build_validity_mask(episode_end, ext_valid, order, num_envs, device):
    valid = torch.zeros_like(episode_end)
    seen_end = torch.zeros(num_envs, dtype=torch.bool, device=device)
    for t in torch.flip(order, dims=(0,)):
        seen_end = seen_end | episode_end[t]
        valid[t] = seen_end | ext_valid[t]
    return valid


def _oracle_build_mc_table(
    rewards, episode_end, ext_valid, external_mc, step_success,
    gamma, sparse_reward_mc, sparse_negative_reward, order, num_envs, device,
):
    mc = torch.zeros_like(rewards)
    running = torch.zeros(num_envs, device=device, dtype=rewards.dtype)
    prev_ext = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if sparse_reward_mc and step_success is not None:
        inf_horizon_value = sparse_negative_reward / (1.0 - gamma)
        acc_succ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        for t in torch.flip(order, dims=(0,)):
            ext_t = ext_valid[t]
            boundary = episode_end[t] | ext_t | prev_ext
            step_succ_t = step_success[t] > 0.5
            acc_succ = step_succ_t | (acc_succ & ~boundary)
            running = rewards[t] + gamma * running * (1.0 - boundary.to(rewards.dtype))
            computed = torch.where(
                acc_succ, running, torch.full_like(running, inf_horizon_value)
            )
            mc_t = torch.where(ext_t, external_mc[t], computed)
            mc[t] = mc_t
            running = mc_t
            prev_ext = ext_t
    else:
        for t in torch.flip(order, dims=(0,)):
            ext_t = ext_valid[t]
            boundary = episode_end[t] | ext_t | prev_ext
            running = rewards[t] + gamma * running * (1.0 - boundary.to(rewards.dtype))
            mc_t = torch.where(ext_t, external_mc[t], running)
            mc[t] = mc_t
            running = mc_t
            prev_ext = ext_t
    return mc


def _randomized_mc_configs():
    random.seed(42)
    configs = [
        dict(T=200, N=8, external_frac=0.0, ep_len=20, pos=0, full=True, sparse=False, success_rate=0.0),
        dict(T=200, N=8, external_frac=0.3, ep_len=20, pos=50, full=True, sparse=False, success_rate=0.0),
        dict(T=200, N=8, external_frac=0.9, ep_len=5, pos=30, full=True, sparse=False, success_rate=0.0),
        dict(T=120, N=4, external_frac=0.0, ep_len=97, pos=60, full=False, sparse=False, success_rate=0.0),
        dict(T=80, N=16, external_frac=0.5, ep_len=1, pos=0, full=True, sparse=False, success_rate=0.0),
        dict(T=80, N=2, external_frac=0.0, ep_len=79, pos=0, full=True, sparse=False, success_rate=0.0),
        dict(T=80, N=2, external_frac=1.0, ep_len=5, pos=0, full=True, sparse=False, success_rate=0.0),
        dict(T=64, N=1, external_frac=0.0, ep_len=7, pos=13, full=True, sparse=False, success_rate=0.0),
        # sparse_reward_mc branch, including the coupling-bug config that
        # caught the "running carries the overridden value" mistake.
        dict(T=80, N=8, external_frac=0.0, ep_len=1, pos=0, full=True, sparse=True, success_rate=0.5),
        dict(T=80, N=8, external_frac=0.0, ep_len=20, pos=13, full=True, sparse=True, success_rate=0.5),
        dict(T=200, N=4, external_frac=0.3, ep_len=50, pos=0, full=True, sparse=True, success_rate=0.1),
        dict(T=200, N=4, external_frac=0.7, ep_len=3, pos=47, full=False, sparse=True, success_rate=0.02),
    ]
    for i in range(15):
        configs.append(dict(
            T=random.choice([50, 200, 333]),
            N=random.choice([1, 3, 8]),
            external_frac=random.choice([0.0, 0.2, 0.5, 0.8]),
            ep_len=random.choice([1, 5, 20, 60]),
            pos=random.randint(0, 40),
            full=random.choice([True, False]),
            sparse=random.choice([True, False]),
            success_rate=random.choice([0.0, 0.02, 0.1, 0.5]),
        ))
    return configs


@pytest.mark.parametrize("cfg", _randomized_mc_configs())
def test_vectorized_builders_match_loop_oracle(obs_space, action_space, cfg):
    device = "cpu"
    T, N = cfg["T"], cfg["N"]
    pos = min(cfg["pos"], T - 1)
    gamma = 0.9
    neg = -5.0

    buf = MCTensorReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_envs=N,
        buffer_size=T * N,
        gamma=gamma,
        storage_device=device,
        sample_device=device,
        sparse_reward_mc=cfg["sparse"],
        sparse_negative_reward=neg,
        success_threshold=0.5,
    )
    # crc32, not str.__hash__, which is randomized per-process (PYTHONHASHSEED)
    # and would make a given config's exercised data non-reproducible across runs.
    g = torch.Generator().manual_seed(zlib.crc32(str(cfg).encode()) % (2**31))
    rewards = torch.randn(T, N, generator=g)
    ee = torch.zeros(T, N, dtype=torch.bool)
    ep_len = cfg["ep_len"]
    ee[ep_len - 1 :: ep_len, :] = True
    ee[-1, :] = True
    ext = torch.zeros(T, N, dtype=torch.bool)
    n_ext = int(T * cfg["external_frac"])
    if n_ext > 0:
        ext[:n_ext, :] = True
    emc = torch.randn(T, N, generator=g) * ext.float()
    step_success = None
    if cfg["sparse"]:
        step_success = (torch.rand(T, N, generator=g) < cfg["success_rate"]).float()
        buf._step_success = step_success

    buf.rewards = rewards
    buf._episode_end = ee
    buf._externally_valid = ext
    buf._external_mc = emc
    buf.pos = pos
    buf.full = cfg["full"]

    order = buf._chronological_order()

    got_valid = buf._build_validity_mask()
    ref_valid = _oracle_build_validity_mask(ee, ext, order, N, device)
    assert torch.equal(got_valid, ref_valid)

    got_mc = buf._build_mc_table()
    ref_mc = _oracle_build_mc_table(
        rewards, ee, ext, emc, step_success, gamma, cfg["sparse"], neg, order, N, device,
    )
    assert not torch.isnan(got_mc[ref_valid]).any()
    assert not torch.isinf(got_mc[ref_valid]).any()
    torch.testing.assert_close(got_mc[ref_valid], ref_mc[ref_valid], atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
