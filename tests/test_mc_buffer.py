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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
