from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.buffers import DictReplayBuffer, TensorReplayBuffer
from rl_garden.encoders import BaseFeaturesExtractor, CombinedExtractor, FlattenExtractor


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
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return DummyVecEnv(obs_space, act_space)


def _image_only_env() -> DummyVecEnv:
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
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
    assert isinstance(agent.replay_buffer, TensorReplayBuffer)


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
        image_keys=("rgb",),
    )
    obs = {
        "rgb": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "state": torch.randn(2, 5),
    }
    moved = agent._obs_to_policy_device(obs)
    assert moved["rgb"] is obs["rgb"]
    assert moved["state"] is obs["state"]
    assert all(v.device == agent.device for v in moved.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rollout_cpu_obs_moves_to_cuda_policy_device_when_needed():
    agent = SAC(env=_state_env(), **{**_agent_kwargs(), "device": "cuda"})
    obs = torch.randn(2, 5, device="cpu")
    moved = agent._obs_to_policy_device(obs)
    assert moved.device.type == "cuda"
    assert obs.device.type == "cpu"


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
        image_keys=("rgb", "depth"),
    )
    assert isinstance(agent.policy.features_extractor, CombinedExtractor)
    assert isinstance(agent.replay_buffer, DictReplayBuffer)


def test_sac_image_only_dict_obs_uses_combined_extractor():
    agent = SAC(
        env=_image_only_env(),
        **_agent_kwargs(),
        image_keys=("rgb",),
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, CombinedExtractor)
    assert extractor.image_keys == ("rgb",)
    assert extractor.has_state is False


def test_sac_dict_vector_keys_are_flattened():
    agent = SAC(env=_dict_vector_env(), **_agent_kwargs())
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, CombinedExtractor)
    assert "extra" in extractor.vector_extractors
    assert extractor.features_dim == 64 + 3


def test_sac_image_encoder_factory_still_works():
    def factory(img_space):
        return RecordingImageExtractor(img_space, features_dim=19, marker="legacy")

    agent = SAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
        image_keys=("rgb",),
        image_encoder_factory=factory,
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, CombinedExtractor)
    assert isinstance(extractor.image_encoder, RecordingImageExtractor)
    assert extractor.image_encoder.marker == "legacy"
    assert extractor.image_keys == ("rgb",)


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


def test_sac_dict_policy_kwargs_win_over_extractor_args():
    def legacy_factory(img_space):
        return RecordingImageExtractor(img_space, features_dim=19, marker="legacy")

    def new_factory(img_space):
        return RecordingImageExtractor(img_space, features_dim=31, marker="policy")

    agent = SAC(
        env=_rgbd_env(),
        **_agent_kwargs(),
        image_keys=("rgb", "depth"),
        image_encoder_factory=legacy_factory,
        policy_kwargs={
            "features_extractor_kwargs": {
                "image_keys": ("rgb",),
                "image_encoder_factory": new_factory,
            }
        },
    )
    extractor = agent.policy.features_extractor
    assert isinstance(extractor, CombinedExtractor)
    assert extractor.image_keys == ("rgb",)
    assert isinstance(extractor.image_encoder, RecordingImageExtractor)
    assert extractor.image_encoder.marker == "policy"


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


def test_sac_dict_obs_rejects_actor_encoder_updates():
    with pytest.raises(ValueError, match="trained only by critic loss"):
        SAC(
            env=_rgbd_env(),
            **_agent_kwargs(),
            detach_encoder_on_actor=False,
        )


def test_sac_box_obs_rejects_image_kwargs():
    with pytest.raises(ValueError, match="image-related kwargs"):
        SAC(env=_state_env(), **_agent_kwargs(), image_keys=("rgb",))

    with pytest.raises(ValueError, match="image-related kwargs"):
        SAC(env=_state_env(), **_agent_kwargs(), use_proprio=False)


def test_sac_box_checkpoint_metadata_omits_image_fields():
    agent = SAC(env=_state_env(), **_agent_kwargs())
    meta = agent._checkpoint_metadata()
    for key in (
        "image_keys",
        "state_key",
        "use_proprio",
        "proprio_latent_dim",
        "image_fusion_mode",
        "enable_stacking",
    ):
        assert key not in meta


def test_sac_dict_checkpoint_metadata_includes_image_fields():
    agent = SAC(env=_rgbd_env(), **_agent_kwargs(), image_keys=("rgb",))
    meta = agent._checkpoint_metadata()
    assert meta["image_keys"] == ("rgb",)
    assert meta["use_proprio"] is True
