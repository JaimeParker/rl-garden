"""Unit tests for MCReplayBuffer with Monte Carlo return computation."""
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
        # Add some transitions
        for _ in range(10):
            obs = torch.randn(4, 4)
            next_obs = torch.randn(4, 4)
            actions = torch.randn(4, 2)
            rewards = torch.ones(4)
            dones = torch.zeros(4)
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
        # Add some transitions
        for _ in range(10):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
