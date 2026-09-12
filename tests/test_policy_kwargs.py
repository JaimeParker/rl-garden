from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC, WSRL, OfflineEnvSpec
from rl_garden.algorithms.bc import BC
from rl_garden.buffers import ReplayBuffer
from rl_garden.buffers.mc_buffer import MCReplayBuffer
from rl_garden.encoders import BaseFeaturesExtractor, CombinedExtractor, FlattenExtractor
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObservationContractError, ObsGroups


class DummyVecEnv:
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box) -> None:
        self.num_envs = 1
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = action_space


class RecordingExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 13,
        marker: str = "default",
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.marker = marker

    def forward(self, obs):
        batch = obs.shape[0] if isinstance(obs, torch.Tensor) else next(iter(obs.values())).shape[0]
        return torch.zeros(batch, self.features_dim)


class RecordingImageExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 17,
        marker: str = "image",
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.marker = marker

    def forward(self, obs):
        return torch.zeros(obs.shape[0], self.features_dim)


def _state_env() -> DummyVecEnv:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return DummyVecEnv(obs_space, act_space)


def _rgbd_env() -> DummyVecEnv:
    obs_space = spaces.Dict(
        {
            "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "depth_cam": spaces.Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return DummyVecEnv(obs_space, act_space)


def _image_only_env() -> DummyVecEnv:
    obs_space = spaces.Dict(
        {
            "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return DummyVecEnv(obs_space, act_space)


def _dict_vector_env() -> DummyVecEnv:
    obs_space = spaces.Dict(
        {
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "extra": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return DummyVecEnv(obs_space, act_space)


def _agent_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 8,
        "batch_size": 2,
        "eval_freq": 0,
    }


def test_sac_uses_flatten_extractor_by_default():
    agent = SAC(env=_state_env(), **_agent_kwargs())
    assert isinstance(agent.policy.features_extractor, FlattenExtractor)
    assert isinstance(agent.replay_buffer, ReplayBuffer)


def test_rollout_obs_on_policy_device_is_noop_for_tensor_obs():
    agent = SAC(env=_state_env(), **_agent_kwargs())
    obs = torch.randn(2, 5)
    moved = agent._obs_to_policy_device(obs)
    assert moved is obs
    assert moved.device == agent.device


def test_rollout_obs_on_policy_device_is_noop_for_dict_obs():
    agent = SAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
    )
    obs = {
        "rgb_cam": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "state": torch.randn(2, 5),
    }
    moved = agent._obs_to_policy_device(obs)
    assert moved["rgb_cam"] is obs["rgb_cam"]
    assert moved["state"] is obs["state"]
    assert all(v.device == agent.device for v in moved.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rollout_cpu_obs_moves_to_cuda_policy_device_when_needed():
    agent = SAC(env=_state_env(), **{**_agent_kwargs(), "device": "cuda"})
    obs = torch.randn(2, 5, device="cpu")
    moved = agent._obs_to_policy_device(obs)
    assert moved.device.type == "cuda"
    assert obs.device.type == "cpu"


def test_sac_features_extractor_kwargs_without_class_raises():
    with pytest.raises(ValueError, match="features_extractor_class"):
        SAC(
            env=_state_env(),
            **_agent_kwargs(),
            policy_kwargs={"features_extractor_kwargs": {"features_dim": 23}},
        )


def test_sac_critic_features_extractor_kwargs_without_class_raises():
    with pytest.raises(ValueError, match="critic_features_extractor_class"):
        SAC(
            env=_rgbd_env(),
            **_agent_kwargs(),
            policy_kwargs={"critic_features_extractor_kwargs": {"features_dim": 23}},
        )


def test_bc_features_extractor_kwargs_without_class_raises():
    """Non-SAC-family regression: the guard hoisted into
    rl_garden.algorithms._policy_kwargs.normalize_policy_kwargs applies to
    every algorithm that delegates to it, not just SAC."""
    env = OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=1,
    )
    with pytest.raises(ValueError, match="features_extractor_class"):
        BC(
            env=env,
            buffer_size=8,
            buffer_device="cpu",
            batch_size=2,
            device="cpu",
            policy_kwargs={"features_extractor_kwargs": {"features_dim": 23}},
        )


def test_sac_policy_kwargs_can_build_custom_extractor():
    agent = SAC(
        env=_state_env(),
        **_agent_kwargs(),
        policy_kwargs={
            "features_extractor_class": RecordingExtractor,
            "features_extractor_kwargs": {"features_dim": 23, "marker": "state-custom"},
        },
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, RecordingExtractor)
    assert extractor.features_dim == 23
    assert extractor.marker == "state-custom"


def test_sac_dict_obs_uses_combined_extractor_by_default():
    agent = SAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
    )
    assert isinstance(agent.policy.features_extractor, CombinedExtractor)
    assert isinstance(agent.replay_buffer, ReplayBuffer)


def test_sac_image_only_dict_obs_uses_combined_extractor():
    agent = SAC(
        env=_image_only_env(),
        **_agent_kwargs(),
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, CombinedExtractor)
    assert extractor.image_keys == ("rgb_cam",)
    assert extractor.has_state is False


def test_sac_rejects_unknown_obs_key_in_obs_groups():
    with pytest.raises(ObservationContractError):
        SAC(env=_dict_vector_env(), **_agent_kwargs(), obs_groups=ObsGroups(actor=("extra",)))


def test_sac_dict_policy_kwargs_can_override_with_custom_extractor():
    agent = SAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
        policy_kwargs={
            "features_extractor_class": RecordingExtractor,
            "features_extractor_kwargs": {"features_dim": 29, "marker": "rgbd-custom"},
        },
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, RecordingExtractor)
    assert extractor.features_dim == 29
    assert extractor.marker == "rgbd-custom"


def test_sac_encoder_config_controls_default_extractor():
    # Schema-driven default: encoder_config (backbone/fusion knobs) plus the
    # observation space (which keys exist) fully determine the extractor --
    # no more (features_extractor_class, features_extractor_kwargs) tuple to
    # override piecemeal without a class override (see
    # test_sac_dict_policy_kwargs_can_override_with_custom_extractor for the
    # full-override escape hatch, which still works unchanged).
    agent = SAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
        encoder_config=EncoderConfig(image_fusion_mode="per_key"),
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, CombinedExtractor)
    assert set(extractor.image_keys) == {"rgb_cam", "depth_cam"}
    assert extractor.fusion_mode == "per_key"


def test_unknown_policy_kwargs_raise_clear_error():
    with pytest.raises(ValueError, match="Unsupported policy_kwargs keys"):
        SAC(
            env=_state_env(),
            **_agent_kwargs(),
            policy_kwargs={"unknown_key": 1},
        )


def test_sac_net_arch_dict_splits_actor_and_critic():
    agent = SAC(
        env=_state_env(),
        **_agent_kwargs(),
        net_arch={"pi": [7], "qf": [9, 8]},
    )
    actor_first_linear = next(m for m in agent.policy.actor.trunk.modules() if isinstance(m, torch.nn.Linear))
    assert actor_first_linear.out_features == 7
    # vmap-fused EnsembleQCritic stores stacked params, not a ModuleList of
    # critics. The first trunk-Linear weight has shape
    # ``(n_critics, qf_hidden_0, features_dim + act_dim)``, so we read the
    # qf_hidden_0 (= 9) off axis 1.
    first_critic_weight = agent.policy.critic.ens_p_trunk__0__weight
    assert first_critic_weight.shape[1] == 9


def test_sac_deprecated_hidden_dims_still_work_with_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        agent = SAC(
            env=_state_env(),
            **_agent_kwargs(),
            actor_hidden_dims=(13, 11),
            critic_hidden_dims=(17, 15),
        )
    assert agent.net_arch == {"pi": [13, 11], "qf": [17, 15]}


def test_sac_net_arch_missing_keys_raises():
    with pytest.raises(ValueError, match="both 'pi' and 'qf'"):
        SAC(
            env=_state_env(),
            **_agent_kwargs(),
            net_arch={"pi": [32, 32]},
        )


def test_sac_encoder_sharing_shared_disables_actor_stop_gradient():
    # "shared" (unlike the "shared_critic_grad" default) trains the encoder
    # from both actor and critic losses -- replaces the old hardcoded
    # detach_encoder_on_actor=False rejection: the redesign makes this a
    # first-class supported encoder_sharing value, not an error.
    agent = SAC(env=_rgbd_env(), **_agent_kwargs(), encoder_sharing="shared")
    assert agent.policy.critic_features_extractor is agent.policy.features_extractor
    assert agent._actor_stop_gradient() is False


def test_sac_encoder_sharing_shared_critic_grad_stop_gradients_actor():
    agent = SAC(env=_rgbd_env(), **_agent_kwargs())  # default: shared_critic_grad
    assert agent._actor_stop_gradient() is True


def test_sac_box_obs_rejects_unknown_obs_group_key():
    # A Box space normalizes to {"state": Box}; requesting "rgb_cam" in
    # obs_groups for it is a contract violation (unknown key), replacing the
    # old hardcoded "Box rejects image kwargs" check.
    with pytest.raises(ObservationContractError):
        SAC(env=_state_env(), **_agent_kwargs(), obs_groups=ObsGroups(actor=("rgb_cam",)))


def test_sac_checkpoint_metadata_always_includes_encoder_fields():
    agent = SAC(env=_state_env(), **_agent_kwargs())
    meta = agent._checkpoint_metadata()
    assert meta["encoder_sharing"] == "shared_critic_grad"
    assert meta["encoder_config"] is None
    assert meta["obs_groups"] is None
    assert meta["critic_encoder_config"] is None


def test_sac_dict_checkpoint_metadata_serializes_encoder_config():
    agent = SAC(
        env=_rgbd_env(), **_agent_kwargs(), encoder_config=EncoderConfig(features_dim=17)
    )
    meta = agent._checkpoint_metadata()
    assert meta["encoder_config"]["features_dim"] == 17
    assert meta["encoder_config"]["backbone"] == "plain_conv"


def test_sac_passes_image_augmentation_to_combined_extractor():
    agent = SAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
        encoder_config=EncoderConfig(image_augmentation="random_shift", image_random_shift_pad=2),
        image_augmentation_seed=123,
    )
    ext = agent.policy.features_extractor

    assert isinstance(ext, CombinedExtractor)
    assert ext.image_augmentation == "random_shift"
    assert ext.random_shift_pad == 2
    assert ext._augmentation_seed == 123


def test_wsrl_uses_flatten_extractor_by_default():
    agent = WSRL(env=_state_env(), **_agent_kwargs())
    assert isinstance(agent.policy.features_extractor, FlattenExtractor)
    assert isinstance(agent.replay_buffer, MCReplayBuffer)


def test_wsrl_dict_obs_uses_combined_extractor_by_default():
    agent = WSRL(
        env=_rgbd_env(),
        **_agent_kwargs(),
        obs_groups=ObsGroups(actor=("rgb_cam", "state"), critic=("rgb_cam", "state")),
    )
    assert isinstance(agent.policy.features_extractor, CombinedExtractor)
    assert isinstance(agent.replay_buffer, MCReplayBuffer)
    assert agent._actor_stop_gradient() is True


def test_wsrl_dict_obs_encoder_sharing_shared_disables_actor_stop_gradient():
    # encoder_sharing="shared" is a first-class supported mode now (not an
    # error), replacing the old hardcoded detach_encoder_on_actor=False
    # rejection ("trained only by critic loss").
    agent = WSRL(env=_rgbd_env(), **_agent_kwargs(), encoder_sharing="shared")
    assert agent._actor_stop_gradient() is False


def test_wsrl_box_obs_rejects_unknown_obs_group_key():
    # A Box space normalizes to {"state": Box}; requesting "rgb_cam" in
    # obs_groups for it is a contract violation (unknown key), replacing the
    # old hardcoded "Box rejects image kwargs" check.
    from rl_garden.observations import ObservationContractError

    with pytest.raises(ObservationContractError):
        WSRL(env=_state_env(), **_agent_kwargs(), obs_groups=ObsGroups(actor=("rgb_cam",)))


def test_wsrl_dict_checkpoint_metadata_includes_encoder_fields():
    agent = WSRL(
        env=_rgbd_env(),
        **_agent_kwargs(),
        encoder_config=EncoderConfig(features_dim=17),
    )
    meta = agent._checkpoint_metadata()
    assert meta["encoder_config"]["features_dim"] == 17
    assert meta["encoder_sharing"] == "shared_critic_grad"


def _multi_camera_env() -> DummyVecEnv:
    obs_space = spaces.Dict(
        {
            "rgb_base_camera": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "rgb_hand_camera": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return DummyVecEnv(obs_space, act_space)


def test_sac_per_camera_keys_build_separate_encoders():
    agent = SAC(
        env=_multi_camera_env(),
        **_agent_kwargs(),
        encoder_config=EncoderConfig(image_fusion_mode="per_key"),
    )
    ext = agent.policy.features_extractor
    assert isinstance(ext, CombinedExtractor)
    assert set(ext.image_encoders.keys()) == {"rgb_base_camera", "rgb_hand_camera"}
    for enc in ext.image_encoders.values():
        # Each per-camera encoder is built from a 3-channel (C, H, W) Box.
        assert enc._observation_space.shape[0] == 3


def test_schema_image_keys_orders_rgb_before_depth():
    from rl_garden.observations import ObservationSchema

    # gymnasium.spaces.Dict sorts its keys alphabetically regardless of
    # insertion order, so ObservationSchema (built by iterating
    # space.spaces.items()) sees them alphabetically too.
    obs_space = spaces.Dict(
        {
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "rgb_hand_camera": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "rgb_base_camera": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "depth_base_camera": spaces.Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
        }
    )
    keys = ObservationSchema.from_space(obs_space).image_keys
    assert keys == ("rgb_base_camera", "rgb_hand_camera", "depth_base_camera")
