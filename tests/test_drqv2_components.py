from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import DDPG
from rl_garden.buffers.nstep_buffer import LazyNextNStepReplayBuffer
from rl_garden.encoders.config import EncoderConfig
from rl_garden.encoders.drqv2_conv import DrQv2Encoder
from rl_garden.networks.ddpg_critic import DrQv2Critic
from rl_garden.observations import ObsGroups


def _no_aug_encoder_config(**overrides) -> EncoderConfig:
    """DDPG's own default backbone (drqv2_conv), augmentation disabled so
    tests get deterministic, unpadded feature shapes."""
    return EncoderConfig(backbone="drqv2_conv", image_augmentation="none", **overrides)


class DummyDictVecEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.single_observation_space = spaces.Dict(
            {
                "rgb_cam": spaces.Box(low=0, high=255, shape=(32, 40, 3), dtype=np.uint8),
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


class DummyStateExtraVecEnv:
    """A ``rgb_cam`` + ``state`` + ``state_object_pose`` env: ``rgb_cam``/
    ``state`` are shared by actor and critic (both get a real, trainable
    extractor); ``state_object_pose`` (Section A's ``state_<name>`` family)
    is critic-only."""

    def __init__(self) -> None:
        self.num_envs = 1
        self.single_observation_space = spaces.Dict(
            {
                "rgb_cam": spaces.Box(low=0, high=255, shape=(32, 40, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "state_object_pose": spaces.Box(
                    low=-1.0, high=1.0, shape=(3,), dtype=np.float32
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


def test_ddpg_rejects_dict_observations_with_unknown_obs_group_key():
    with pytest.raises(ValueError, match="unknown observation key"):
        DDPG(
            env=DummyPerCameraDictVecEnv(),
            obs_groups=ObsGroups(actor=("rgb_cam",), critic=("rgb_cam",)),
            device="cpu",
            buffer_device="cpu",
            buffer_size=16,
            batch_size=2,
            eval_freq=0,
            hidden_dim=16,
            feature_dim=8,
            encoder_config=_no_aug_encoder_config(),
        )


def test_ddpg_uses_explicit_per_camera_image_keys():
    agent = DDPG(
        env=DummyPerCameraDictVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        eval_freq=0,
        hidden_dim=16,
        feature_dim=8,
        encoder_config=_no_aug_encoder_config(image_fusion_mode="per_key"),
    )

    assert agent.policy.actor_extractor.image_keys == (
        "rgb_base_camera",
        "rgb_hand_camera",
    )
    assert set(agent.policy.actor_extractor.image_encoders) == {
        "rgb_base_camera",
        "rgb_hand_camera",
    }


def test_ddpg_builds_mmap_nstep_buffer(tmp_path):
    agent = DDPG(
        env=DummyDictVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        eval_freq=0,
        hidden_dim=16,
        feature_dim=8,
        encoder_config=_no_aug_encoder_config(),
        mmap_dir=tmp_path,
    )

    assert agent.replay_buffer._mmap_store is not None
    assert (tmp_path / "manifest.json").is_file()


def test_ddpg_builds_lazy_next_nstep_buffer():
    agent = DDPG(
        env=DummyDictVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        eval_freq=0,
        hidden_dim=16,
        feature_dim=8,
        encoder_config=_no_aug_encoder_config(),
        replay_lazy_next_obs=True,
        replay_pin_sampled_batch=True,
    )

    assert isinstance(agent.replay_buffer, LazyNextNStepReplayBuffer)
    assert agent.replay_buffer.next_obs is None
    assert agent.replay_buffer.pin_sampled_batch is True


def test_ddpg_rejects_lazy_next_with_mmap(tmp_path):
    with pytest.raises(ValueError, match="lazy next_obs"):
        DDPG(
            env=DummyDictVecEnv(),
            device="cpu",
            buffer_device="cpu",
            buffer_size=16,
            batch_size=2,
            eval_freq=0,
            hidden_dim=16,
            feature_dim=8,
            encoder_config=_no_aug_encoder_config(),
            mmap_dir=tmp_path,
            replay_lazy_next_obs=True,
        )


def test_ddpg_rejects_pinned_sampling_without_lazy_next():
    with pytest.raises(ValueError, match="requires replay_lazy_next_obs"):
        DDPG(
            env=DummyDictVecEnv(),
            device="cpu",
            buffer_device="cpu",
            buffer_size=16,
            batch_size=2,
            eval_freq=0,
            hidden_dim=16,
            feature_dim=8,
            encoder_config=_no_aug_encoder_config(),
            replay_pin_sampled_batch=True,
        )


def test_ddpg_rejects_mmap_replay_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="cannot be embedded"):
        DDPG(
            env=DummyDictVecEnv(),
            device="cpu",
            buffer_device="cpu",
            buffer_size=16,
            batch_size=2,
            eval_freq=0,
            hidden_dim=16,
            feature_dim=8,
            encoder_config=_no_aug_encoder_config(),
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
        encoder_config=_no_aug_encoder_config(proprio_latent_dim=4),
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
        "rgb_cam": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
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
        encoder_config=_no_aug_encoder_config(proprio_latent_dim=4),
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
        "rgb_cam": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
        "state": torch.randn(1, 4),
    }

    agent._rollout_action(obs, learning_has_started=True)

    assert observed["clip"] is None


def test_ddpg_update_role_based_extraction_and_clips_training_noise(
    monkeypatch,
):
    """Under the default shared encoder (encoder_sharing="shared_critic_grad",
    no critic_extractor), DDPG.train() restores DrQ-v2's single augmented
    view (see BasePolicy.actor_features_from_critic): the critic path
    (``extract_critic_features``) extracts ``data.obs`` once and
    ``data.next_obs`` once (under no_grad) -- two extractor calls total --
    and the actor path REUSES the critic's own ``data.obs`` features
    (detached) instead of a third, redundant ``extract_actor_features``
    call. See the "separate" sibling test below for the case where the
    actor genuinely has its own encoder and a third call is unavoidable."""
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
        encoder_config=_no_aug_encoder_config(proprio_latent_dim=4),
    )
    for step in range(5):
        obs = {
            "rgb_cam": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
        }
        next_obs = {
            "rgb_cam": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
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
    original_extract_critic = agent.policy.extract_critic_features
    original_extract_actor = agent.policy.extract_actor_features

    def wrapped_extract_critic(obs, stop_gradient=False):
        nonlocal extract_count
        extract_count += 1
        return original_extract_critic(obs, stop_gradient=stop_gradient)

    def wrapped_extract_actor(obs):
        nonlocal extract_count
        extract_count += 1
        return original_extract_actor(obs)

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

    monkeypatch.setattr(agent.policy, "extract_critic_features", wrapped_extract_critic)
    monkeypatch.setattr(agent.policy, "extract_actor_features", wrapped_extract_actor)
    monkeypatch.setattr(agent.policy.actor, "forward", wrapped_forward)

    agent.train(1)

    assert extract_count == 2
    assert clips == [agent.stddev_clip, agent.stddev_clip]


def test_ddpg_update_separate_encoders_extracts_three_times(monkeypatch):
    """Sibling of the shared-encoder test above: under
    encoder_sharing="separate" the actor has its own, genuinely different
    extractor, so BasePolicy.actor_features_from_critic returns None and
    DDPG.train() falls back to a real extract_actor_features(data.obs)
    call -- three extractor calls total (critic obs, critic next_obs,
    actor obs), not two."""
    agent = DDPG(
        env=DummyStateExtraVecEnv(),
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
        encoder_config=_no_aug_encoder_config(proprio_latent_dim=4),
        obs_groups=ObsGroups(
            actor=("rgb_cam", "state"), critic=("rgb_cam", "state", "state_object_pose")
        ),
        encoder_sharing="separate",
    )
    for step in range(5):
        obs = {
            "rgb_cam": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
            "state_object_pose": torch.randn(1, 3),
        }
        next_obs = {
            "rgb_cam": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
            "state_object_pose": torch.randn(1, 3),
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
    original_extract_critic = agent.policy.extract_critic_features
    original_extract_actor = agent.policy.extract_actor_features

    def wrapped_extract_critic(obs, stop_gradient=False):
        nonlocal extract_count
        extract_count += 1
        return original_extract_critic(obs, stop_gradient=stop_gradient)

    def wrapped_extract_actor(obs):
        nonlocal extract_count
        extract_count += 1
        return original_extract_actor(obs)

    monkeypatch.setattr(agent.policy, "extract_critic_features", wrapped_extract_critic)
    monkeypatch.setattr(agent.policy, "extract_actor_features", wrapped_extract_actor)

    agent.train(1)

    assert extract_count == 3


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
        encoder_config=_no_aug_encoder_config(proprio_latent_dim=4),
    )

    for step in range(5):
        obs = {
            "rgb_cam": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
        }
        next_obs = {
            "rgb_cam": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
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


def test_ddpg_asymmetric_obs_groups_critic_sees_extra_state_actor_does_not():
    """End-to-end asymmetric actor/critic encoders via state_<name>: critic
    sees state_object_pose, actor does not, encoder_sharing="separate".
    Asserts the actor extractor's schema lacks the key and, over one real
    train() step, gradients reach only the right extractor."""
    agent = DDPG(
        env=DummyStateExtraVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=4,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        nstep=1,
        hidden_dim=16,
        feature_dim=8,
        encoder_config=_no_aug_encoder_config(proprio_latent_dim=4),
        obs_groups=ObsGroups(
            actor=("rgb_cam", "state"), critic=("rgb_cam", "state", "state_object_pose")
        ),
        encoder_sharing="separate",
    )

    actor_extractor = agent.policy.actor_extractor
    critic_extractor = agent.policy.critic_extractor
    assert critic_extractor is not None and critic_extractor is not actor_extractor
    assert "state_object_pose" not in actor_extractor.state_keys
    assert "state_object_pose" in critic_extractor.state_keys

    for step in range(8):
        obs = {
            "rgb_cam": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
            "state_object_pose": torch.randn(1, 3),
        }
        next_obs = {
            "rgb_cam": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
            "state_object_pose": torch.randn(1, 3),
        }
        agent.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=torch.randn(1, 2).clamp(-1, 1),
            reward=torch.full((1,), float(step)),
            done=torch.zeros(1, dtype=torch.bool),
            episode_end=torch.zeros(1, dtype=torch.bool),
        )

    data = agent.replay_buffer.sample(4)

    # --- Critic loss: mirrors DDPG.train()'s own computation, without
    # stepping the optimizer, so the loss tensor is available for isolation
    # checks below. ---
    critic_features = agent.policy.extract_critic_features(data.obs)
    with torch.no_grad():
        next_features = agent.policy.extract_critic_features(data.next_obs)
        target_std, target_clip = agent._target_action_noise()
        dist = agent.policy.actor(next_features, target_std)
        next_action = dist.sample(clip=target_clip)
        target_q_all = agent.policy.q_values_all(next_features, next_action, target=True)
        target_q = (
            data.rewards.reshape(-1, 1)
            + data.discounts.reshape(-1, 1) * target_q_all.min(dim=0).values
        )
    q_all = agent.policy.q_values_all(critic_features, data.actions, target=False)
    critic_loss = agent._critic_loss(q_all, target_q)

    critic_grad_on_critic = torch.autograd.grad(
        critic_loss, list(critic_extractor.parameters()), retain_graph=True, allow_unused=True
    )
    assert any(g is not None and torch.any(g != 0) for g in critic_grad_on_critic)
    critic_grad_on_actor = torch.autograd.grad(
        critic_loss, list(actor_extractor.parameters()), allow_unused=True
    )
    assert all(g is None for g in critic_grad_on_actor)

    # --- Actor loss: mirrors DDPG.train()'s actor-update block. ---
    actor_features = agent.policy.extract_actor_features(data.obs)
    action = agent.policy.actor_action_from_features(
        actor_features, agent._current_stddev(), noise_clip=agent.stddev_clip
    )
    q_actor_features = agent.policy.critic_features_for(data.obs, actor_features, stop_gradient=True)
    q_actor_all = agent.policy.q_values_all(q_actor_features, action, target=False)
    actor_loss = -agent._actor_q_value(q_actor_all).mean()

    actor_grad_on_actor = torch.autograd.grad(
        actor_loss, list(actor_extractor.parameters()), retain_graph=True, allow_unused=True
    )
    assert any(g is not None and torch.any(g != 0) for g in actor_grad_on_actor)
    # The actor loss's Q(s, pi(s)) term re-extracts critic-role features with
    # stop_gradient=True (DDPGPolicy.critic_features_for); CombinedExtractor's
    # own stop_gradient convention detaches only the image branch, so this
    # checks image-branch isolation specifically (its proprio branch
    # legitimately still requires_grad here -- unrelated to obs_groups).
    actor_grad_on_critic_image = torch.autograd.grad(
        actor_loss, list(critic_extractor.image_encoder.parameters()), allow_unused=True
    )
    assert all(g is None for g in actor_grad_on_critic_image)

    # A real end-to-end rollout+update step also runs cleanly.
    info = agent.train(gradient_steps=1, compute_info=True)
    assert all(np.isfinite(v) for v in info.values())


def _drqv2_build_args(encoder: str):
    from types import SimpleNamespace

    from rl_garden.encoders.config import EncoderConfig
    from rl_garden.observations import ObsGroups, ObservationConfig

    return SimpleNamespace(
        obs=ObservationConfig(rgb=("base_camera",), depth=("base_camera",)),
        obs_groups=ObsGroups(),
        encoder=EncoderConfig(backbone=encoder),
        critic_encoder=EncoderConfig(),
        encoder_sharing=None,
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
        bootstrap_at_done="truncated",
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

    assert captured_kwargs["encoder_config"].backbone == "cnn3d"


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
