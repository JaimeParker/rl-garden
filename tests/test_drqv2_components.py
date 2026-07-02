from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import DDPG
from rl_garden.buffers.nstep_buffer import LazyNextNStepDictReplayBuffer
from rl_garden.encoders.drqv2_conv import DrQv2Encoder
from rl_garden.networks.ddpg_critic import DrQv2Critic


class DummyDictVecEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.single_observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(low=0, high=255, shape=(32, 40, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


class DummyPerCameraDictVecEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.single_observation_space = spaces.Dict(
            {
                "rgb_base_camera": spaces.Box(
                    low=0, high=255, shape=(32, 32, 3), dtype=np.uint8
                ),
                "rgb_hand_camera": spaces.Box(
                    low=0, high=255, shape=(32, 32, 3), dtype=np.uint8
                ),
                "state": spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                ),
            }
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


def test_drqv2_encoder_features_dim_matches_non_square_forward():
    obs_space = spaces.Box(low=0, high=255, shape=(3, 32, 40), dtype=np.uint8)
    encoder = DrQv2Encoder(obs_space)

    out = encoder(torch.randint(0, 256, (2, 3, 32, 40), dtype=torch.uint8))

    assert out.shape == (2, encoder.features_dim)


def test_ddpg_exported_from_algorithms_package():
    from rl_garden.algorithms import DDPG as ExportedDDPG

    assert ExportedDDPG is DDPG


def test_drqv2_critic_matches_reference_architecture():
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    critic = DrQv2Critic(
        features_dim=20,
        action_space=action_space,
        feature_dim=7,
        hidden_dim=11,
    )

    assert isinstance(critic.trunk[0], torch.nn.Linear)
    assert critic.trunk[0].in_features == 20
    assert critic.trunk[0].out_features == 7
    assert isinstance(critic.trunk[1], torch.nn.LayerNorm)
    assert isinstance(critic.trunk[2], torch.nn.Tanh)
    assert critic.q1[0].in_features == 10
    assert critic.q1[0].out_features == 11
    assert critic.q2[0].in_features == 10
    assert critic.q2[0].out_features == 11

    q_all = critic.forward_all(torch.randn(5, 20), torch.randn(5, 3))
    assert q_all.shape == (2, 5, 1)


def test_ddpg_rejects_dict_observations_without_selected_images():
    with pytest.raises(ValueError, match="at least one image observation key"):
        DDPG(
            env=DummyPerCameraDictVecEnv(),
            image_keys=("rgb",),
            device="cpu",
            buffer_device="cpu",
            buffer_size=16,
            batch_size=2,
            eval_freq=0,
            hidden_dim=16,
            feature_dim=8,
            image_augmentation="none",
        )


def test_ddpg_uses_explicit_per_camera_image_keys():
    agent = DDPG(
        env=DummyPerCameraDictVecEnv(),
        image_keys=("rgb_base_camera", "rgb_hand_camera"),
        image_fusion_mode="per_key",
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        eval_freq=0,
        hidden_dim=16,
        feature_dim=8,
        image_augmentation="none",
    )

    assert agent.policy.features_extractor.image_keys == (
        "rgb_base_camera",
        "rgb_hand_camera",
    )
    assert set(agent.policy.features_extractor.image_encoders) == {
        "rgb_base_camera",
        "rgb_hand_camera",
    }


def test_ddpg_builds_mmap_nstep_buffer(tmp_path):
    agent = DDPG(
        env=DummyDictVecEnv(),
        image_keys=("rgb",),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        eval_freq=0,
        hidden_dim=16,
        feature_dim=8,
        image_augmentation="none",
        mmap_dir=tmp_path,
    )

    assert agent.replay_buffer._mmap_store is not None
    assert (tmp_path / "manifest.json").is_file()


def test_ddpg_builds_lazy_next_nstep_buffer():
    agent = DDPG(
        env=DummyDictVecEnv(),
        image_keys=("rgb",),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        eval_freq=0,
        hidden_dim=16,
        feature_dim=8,
        image_augmentation="none",
        replay_lazy_next_obs=True,
        replay_pin_sampled_batch=True,
    )

    assert isinstance(agent.replay_buffer, LazyNextNStepDictReplayBuffer)
    assert agent.replay_buffer.next_obs is None
    assert agent.replay_buffer.pin_sampled_batch is True


def test_ddpg_rejects_lazy_next_with_mmap(tmp_path):
    with pytest.raises(ValueError, match="lazy next_obs"):
        DDPG(
            env=DummyDictVecEnv(),
            image_keys=("rgb",),
            device="cpu",
            buffer_device="cpu",
            buffer_size=16,
            batch_size=2,
            eval_freq=0,
            hidden_dim=16,
            feature_dim=8,
            image_augmentation="none",
            mmap_dir=tmp_path,
            replay_lazy_next_obs=True,
        )


def test_ddpg_rejects_pinned_sampling_without_lazy_next():
    with pytest.raises(ValueError, match="requires replay_lazy_next_obs"):
        DDPG(
            env=DummyDictVecEnv(),
            image_keys=("rgb",),
            device="cpu",
            buffer_device="cpu",
            buffer_size=16,
            batch_size=2,
            eval_freq=0,
            hidden_dim=16,
            feature_dim=8,
            image_augmentation="none",
            replay_pin_sampled_batch=True,
        )


def test_ddpg_rejects_mmap_replay_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="cannot be embedded"):
        DDPG(
            env=DummyDictVecEnv(),
            image_keys=("rgb",),
            device="cpu",
            buffer_device="cpu",
            buffer_size=16,
            batch_size=2,
            eval_freq=0,
            hidden_dim=16,
            feature_dim=8,
            image_augmentation="none",
            mmap_dir=tmp_path,
            save_replay_buffer=True,
        )


def test_ddpg_critic_loss_sums_both_q_heads():
    q_all = torch.tensor([[[1.0]], [[3.0]]])
    target_q = torch.tensor([[0.0]])

    loss = DDPG._critic_loss(q_all, target_q)

    assert torch.equal(loss, torch.tensor(10.0))


def test_ddpg_policy_actor_action_applies_requested_noise_clip(monkeypatch):
    agent = DDPG(
        env=DummyDictVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        eval_freq=0,
        hidden_dim=16,
        feature_dim=8,
        image_keys=("rgb",),
        proprio_latent_dim=4,
        image_augmentation="none",
    )
    observed: dict[str, float | None] = {}
    original_forward = agent.policy.actor.forward

    def wrapped_forward(features, std):
        dist = original_forward(features, std)
        original_sample = dist.sample

        def wrapped_sample(clip=None, sample_shape=torch.Size()):
            observed["clip"] = clip
            return original_sample(clip=clip, sample_shape=sample_shape)

        dist.sample = wrapped_sample
        return dist

    monkeypatch.setattr(agent.policy.actor, "forward", wrapped_forward)
    obs = {
        "rgb": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
        "state": torch.randn(1, 4),
    }

    agent.policy.actor_action(obs, std=0.2, noise_clip=0.3)

    assert observed["clip"] == 0.3


def test_ddpg_rollout_does_not_clip_exploration_noise(monkeypatch):
    agent = DDPG(
        env=DummyDictVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        learning_starts=0,
        eval_freq=0,
        hidden_dim=16,
        feature_dim=8,
        image_keys=("rgb",),
        proprio_latent_dim=4,
        image_augmentation="none",
    )
    observed: dict[str, float | None] = {}
    original_forward = agent.policy.actor.forward

    def wrapped_forward(features, std):
        dist = original_forward(features, std)
        original_sample = dist.sample

        def wrapped_sample(clip=None, sample_shape=torch.Size()):
            observed["clip"] = clip
            return original_sample(clip=clip, sample_shape=sample_shape)

        dist.sample = wrapped_sample
        return dist

    monkeypatch.setattr(agent.policy.actor, "forward", wrapped_forward)
    agent._global_step = agent.num_expl_steps
    obs = {
        "rgb": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
        "state": torch.randn(1, 4),
    }

    agent._rollout_action(obs, learning_has_started=True)

    assert observed["clip"] is None


def test_ddpg_update_encodes_each_observation_once_and_clips_training_noise(
    monkeypatch,
):
    agent = DDPG(
        env=DummyDictVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        nstep=3,
        gamma=0.9,
        hidden_dim=16,
        feature_dim=8,
        image_keys=("rgb",),
        proprio_latent_dim=4,
        image_augmentation="none",
    )
    for step in range(5):
        obs = {
            "rgb": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
        }
        next_obs = {
            "rgb": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
        }
        agent.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=torch.randn(1, 2).clamp(-1, 1),
            reward=torch.full((1,), float(step)),
            done=torch.zeros(1, dtype=torch.bool),
            episode_end=torch.zeros(1, dtype=torch.bool),
        )

    extract_count = 0
    original_extract = agent.policy.extract_features

    def wrapped_extract(obs, stop_gradient=False):
        nonlocal extract_count
        extract_count += 1
        return original_extract(obs, stop_gradient=stop_gradient)

    clips: list[float | None] = []
    original_forward = agent.policy.actor.forward

    def wrapped_forward(features, std):
        dist = original_forward(features, std)
        original_sample = dist.sample

        def wrapped_sample(clip=None, sample_shape=torch.Size()):
            clips.append(clip)
            return original_sample(clip=clip, sample_shape=sample_shape)

        dist.sample = wrapped_sample
        return dist

    monkeypatch.setattr(agent.policy, "extract_features", wrapped_extract)
    monkeypatch.setattr(agent.policy.actor, "forward", wrapped_forward)

    agent.train(1)

    assert extract_count == 2
    assert clips == [agent.stddev_clip, agent.stddev_clip]


def test_ddpg_one_update_uses_nstep_discount_path():
    env = DummyDictVecEnv()
    agent = DDPG(
        env=env,
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        nstep=3,
        gamma=0.9,
        hidden_dim=16,
        feature_dim=8,
        image_keys=("rgb",),
        proprio_latent_dim=4,
        image_augmentation="none",
    )

    for step in range(5):
        obs = {
            "rgb": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
        }
        next_obs = {
            "rgb": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
        }
        agent.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=torch.randn(1, 2).clamp(-1, 1),
            reward=torch.full((1,), float(step)),
            done=torch.zeros(1, dtype=torch.bool),
            episode_end=torch.zeros(1, dtype=torch.bool),
        )

    metrics = agent.train(1, compute_info=True)

    assert set(metrics) >= {"critic_loss", "actor_loss", "target_q", "predicted_q"}
    assert all(np.isfinite(v) for v in metrics.values())


def _drqv2_build_args(encoder: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        encoder=encoder,
        encoder_features_dim=256,
        pretrained_weights=None,
        freeze_resnet_encoder=False,
        freeze_resnet_backbone=False,
        buffer_size=1000,
        buffer_device="cpu",
        mmap_dir=None,
        mmap_mode="create",
        replay_lazy_next_obs=False,
        replay_pin_sampled_batch=False,
        learning_starts=10,
        batch_size=8,
        gamma=0.99,
        tau=0.01,
        training_freq=1,
        utd=0.5,
        policy_lr=1e-4,
        q_lr=1e-4,
        feature_dim=50,
        hidden_dim=64,
        nstep=1,
        stddev_schedule="linear(1.0,0.1,100)",
        stddev_clip=0.3,
        num_expl_steps=100,
        weight_decay=0.0,
        use_adamw=False,
        grad_clip_norm=None,
        image_fusion_mode="stack_channels",
        image_augmentation="none",
        image_random_shift_pad=4,
        frame_stack=3,
        seed=1,
        std_log=False,
        log_freq=100,
        eval_freq=0,
        num_eval_steps=10,
        checkpoint_freq=0,
        save_replay_buffer=False,
        save_final_checkpoint=False,
        load_checkpoint=None,
        load_replay_buffer=False,
    )


def test_build_drqv2_warns_when_encoder_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_garden.algorithms.ddpg as ddpg_module
    from rl_garden.training.online.drqv2 import build_drqv2

    captured_kwargs: dict = {}

    class _FakeDDPG:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(ddpg_module, "DDPG", _FakeDDPG)
    args = _drqv2_build_args(encoder="cnn3d")

    with pytest.warns(UserWarning, match="drqv2_conv"):
        build_drqv2(args, DummyDictVecEnv(), None, None, None)

    assert captured_kwargs["image_encoder_factory"] is not None


def test_build_drqv2_default_encoder_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    import rl_garden.algorithms.ddpg as ddpg_module
    from rl_garden.training.online.drqv2 import build_drqv2

    class _FakeDDPG:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(ddpg_module, "DDPG", _FakeDDPG)
    args = _drqv2_build_args(encoder="drqv2_conv")

    build_drqv2(args, DummyDictVecEnv(), None, None, None)

    assert len(recwarn) == 0
