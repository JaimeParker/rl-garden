"""Unit tests for WSRL with Dict vision observations."""
import numpy as np
import pytest
import torch
from gymnasium import spaces
from unittest.mock import MagicMock

from rl_garden.algorithms import WSRL
from rl_garden.encoders.config import EncoderConfig


@pytest.fixture
def rgbd_env():
    """Create a mock environment with Dict observation space."""
    env = MagicMock()
    env.num_envs = 2
    env.single_observation_space = spaces.Dict({
        "rgb_cam": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype="uint8"),  # HWC format
        "state": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
    })
    env.single_action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    return env


@pytest.fixture
def wsrlrgbd_agent(rgbd_env):
    """Create WSRL agent with small vision networks for testing."""
    return WSRL(
        env=rgbd_env,
        buffer_size=100,
        buffer_device="cpu",
        learning_starts=10,
        batch_size=4,
        gamma=0.99,
        # Small networks for fast testing
        net_arch={"pi": [32, 32], "qf": [32, 32]},
        n_critics=4,
        critic_subsample_size=2,
        # CQL parameters
        use_cql_loss=True,
        cql_n_actions=4,
        cql_alpha=1.0,
        # Vision parameters
        encoder_config=EncoderConfig(proprio_latent_dim=16),
        # General
        device="cpu",
        seed=42,
    )


class TestWSRLDictCreation:
    """Test WSRL Dict observation agent creation and initialization."""

    def test_agent_creation(self, wsrlrgbd_agent):
        assert wsrlrgbd_agent.n_critics == 4
        assert wsrlrgbd_agent.critic_subsample_size == 2
        assert wsrlrgbd_agent.use_cql_loss
        assert wsrlrgbd_agent.use_calql

    def test_dict_observation_space(self, wsrlrgbd_agent):
        obs_space = wsrlrgbd_agent.env.single_observation_space
        assert isinstance(obs_space, spaces.Dict)
        assert "rgb_cam" in obs_space.spaces
        assert "state" in obs_space.spaces

    def test_features_extractor_is_combined(self, wsrlrgbd_agent):
        from rl_garden.encoders.combined import CombinedExtractor
        assert isinstance(wsrlrgbd_agent.policy.features_extractor, CombinedExtractor)

    def test_replay_buffer_is_mc_buffer(self, wsrlrgbd_agent):
        from rl_garden.buffers.mc_buffer import MCReplayBuffer
        assert isinstance(wsrlrgbd_agent.replay_buffer, MCReplayBuffer)

    def test_actor_image_stop_gradient_optional(self, rgbd_env):
        # encoder_sharing="shared" is now a first-class supported mode (not
        # an error): both losses train the encoder, no actor-path detach.
        agent = WSRL(
            env=rgbd_env,
            buffer_size=100,
            buffer_device="cpu",
            batch_size=4,
            encoder_sharing="shared",
            device="cpu",
        )
        assert agent._actor_stop_gradient() is False

    def test_without_proprio(self, rgbd_env):
        """``ObsGroups`` excluding ``state`` from both actor and critic drops
        proprio entirely -- the encoder sees only the image key. Symmetric
        (actor == critic) so no ``encoder_sharing='separate'`` is required."""
        from rl_garden.observations import ObsGroups

        agent = WSRL(
            env=rgbd_env,
            buffer_size=100,
            buffer_device="cpu",
            learning_starts=10,
            batch_size=4,
            gamma=0.99,
            net_arch={"pi": [32, 32], "qf": [32, 32]},
            n_critics=4,
            critic_subsample_size=2,
            use_cql_loss=True,
            cql_n_actions=4,
            cql_alpha=1.0,
            encoder_config=EncoderConfig(proprio_latent_dim=16),
            obs_groups=ObsGroups(actor=("rgb_cam",), critic=("rgb_cam",)),
            device="cpu",
            seed=42,
        )
        assert not agent.observation_encoders.actor.has_state
        assert not agent.observation_encoders.schema.subset(("rgb_cam",)).has_state


class TestWSRLDictObservations:
    """Test handling of dict observations."""

    def test_dict_observation_forward(self, wsrlrgbd_agent):
        obs = {
            "rgb_cam": torch.randint(0, 255, (4, 128, 128, 3), dtype=torch.uint8),  # HWC
            "state": torch.randn(4, 4),
        }

        # Extract features
        features = wsrlrgbd_agent.policy.extract_features(obs)
        assert features.shape[0] == 4  # Batch size
        assert features.ndim == 2  # (batch, features_dim)

    def test_actor_action_with_dict_obs(self, wsrlrgbd_agent):
        obs = {
            "rgb_cam": torch.randint(0, 255, (4, 128, 128, 3), dtype=torch.uint8),  # HWC
            "state": torch.randn(4, 4),
        }

        action, log_prob, features = wsrlrgbd_agent.policy.actor_action_log_prob(obs)
        assert action.shape == (4, 2)
        assert log_prob.shape == (4, 1)

    def test_actor_stop_gradient(self, wsrlrgbd_agent):
        obs = {
            "rgb_cam": torch.randint(0, 255, (4, 128, 128, 3), dtype=torch.uint8),  # HWC
            "state": torch.randn(4, 4),
        }

        _, _, features_stopped = wsrlrgbd_agent.policy.actor_action_log_prob(
            obs, stop_gradient=True
        )
        assert features_stopped.shape[0] == 4
        # RGBD stop-gradient follows hil-serl: image encodings are detached, while
        # proprio features may still require grad before optimizer filtering.
        image_features = wsrlrgbd_agent.policy.features_extractor._encode_images(
            obs, stop_gradient=True
        )[0]
        assert not image_features.requires_grad

        _, _, features_attached = wsrlrgbd_agent.policy.actor_action_log_prob(
            obs, stop_gradient=False
        )
        assert features_attached.shape[0] == 4
        # Features should require grad when stop-gradient is disabled (if encoder has grad)
        # Note: This depends on whether encoder parameters require grad

    def test_detach_encoder_alias_still_works(self, wsrlrgbd_agent):
        obs = {
            "rgb_cam": torch.randint(0, 255, (4, 128, 128, 3), dtype=torch.uint8),
            "state": torch.randn(4, 4),
        }
        _, _, features = wsrlrgbd_agent.policy.actor_action_log_prob(
            obs, detach_encoder=True
        )
        assert features.shape[0] == 4


class TestWSRLDictTraining:
    """Test WSRL training with dict observations."""

    def test_add_dict_transitions(self, wsrlrgbd_agent):
        obs = {
            "rgb_cam": torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8),  # HWC
            "state": torch.randn(2, 4),
        }
        next_obs = {
            "rgb_cam": torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8),  # HWC
            "state": torch.randn(2, 4),
        }
        actions = torch.randn(2, 2)
        rewards = torch.ones(2)
        dones = torch.zeros(2)

        wsrlrgbd_agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)
        assert wsrlrgbd_agent.replay_buffer.pos == 1

    def test_sample_dict_batch(self, wsrlrgbd_agent):
        # Add some transitions, closing the trajectory on the last step -- the
        # MC buffer only samples complete trajectories.
        for step in range(10):
            obs = {
                "rgb_cam": torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8),  # HWC
                "state": torch.randn(2, 4),
            }
            next_obs = {
                "rgb_cam": torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8),  # HWC
                "state": torch.randn(2, 4),
            }
            actions = torch.randn(2, 2)
            rewards = torch.ones(2)
            dones = torch.ones(2) if step == 9 else torch.zeros(2)
            wsrlrgbd_agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)

        # Sample batch
        batch = wsrlrgbd_agent.replay_buffer.sample(4)
        assert isinstance(batch.obs, dict)
        assert "rgb_cam" in batch.obs
        assert "state" in batch.obs
        assert batch.obs["rgb_cam"].shape == (4, 128, 128, 3)  # HWC
        assert batch.obs["state"].shape == (4, 4)
        assert hasattr(batch, "mc_returns")

    def test_actor_loss_with_dict_obs(self, wsrlrgbd_agent):
        obs = {
            "rgb_cam": torch.randint(0, 255, (4, 128, 128, 3), dtype=torch.uint8),  # HWC
            "state": torch.randn(4, 4),
        }

        actor_loss, log_prob = wsrlrgbd_agent._actor_loss(obs)
        assert actor_loss.shape == ()
        assert log_prob.shape == (4, 1)

    def test_train_step_with_dict_obs(self, wsrlrgbd_agent):
        # Add enough data to replay buffer, closing the trajectory on the last
        # step -- the MC buffer only samples complete trajectories.
        for step in range(20):
            obs = {
                "rgb_cam": torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8),  # HWC
                "state": torch.randn(2, 4),
            }
            next_obs = {
                "rgb_cam": torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8),  # HWC
                "state": torch.randn(2, 4),
            }
            actions = torch.randn(2, 2)
            rewards = torch.ones(2)
            dones = torch.ones(2) if step == 19 else torch.zeros(2)
            wsrlrgbd_agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)

        # Run training step
        info = wsrlrgbd_agent.train(gradient_steps=2, compute_info=True)

        assert "critic_loss" in info
        assert "actor_loss" in info
        assert "alpha" in info


class TestWSRLDictConfiguration:
    """Test WSRL Dict observation configuration options."""

    def test_custom_image_keys(self, rgbd_env):
        # Modify env to have depth with same size as rgb (HWC format). Both
        # image keys are already the only ones present, so the schema-driven
        # default extractor includes them with no obs_groups override needed
        # (see the observation-redesign phase-2 recipe's test-migration
        # pattern: "just delete the kwarg" when it only selected a subset
        # that's already the whole space).
        rgbd_env.single_observation_space = spaces.Dict({
            "rgb_cam": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype="uint8"),  # HWC
            "depth_cam": spaces.Box(low=0, high=10, shape=(128, 128, 1), dtype=np.float32),  # HWC
            "state": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        })

        agent = WSRL(
            env=rgbd_env,
            buffer_size=100,
            buffer_device="cpu",
            batch_size=4,
            device="cpu",
        )

        assert agent.policy.features_extractor.features_dim > 0
        from rl_garden.buffers.mc_buffer import MCReplayBuffer

        assert isinstance(agent.replay_buffer, MCReplayBuffer)

    def test_custom_proprio_latent_dim(self, rgbd_env):
        agent = WSRL(
            env=rgbd_env,
            buffer_size=100,
            buffer_device="cpu",
            batch_size=4,
            encoder_config=EncoderConfig(proprio_latent_dim=32),
            device="cpu",
        )

        assert agent.encoder_config.proprio_latent_dim == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
