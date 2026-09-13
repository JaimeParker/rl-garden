from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms.gail import GAIL
from rl_garden.encoders.config import EncoderConfig
from rl_garden.envs.wrappers.gail_reward import GAILRewardWrapper
from rl_garden.networks.discriminator import GAILDiscriminator
from rl_garden.observations import ObsGroups


class _FakeBoxEnv(gym.vector.VectorEnv):
    """Mirrors tests/test_ppo_normalize_obs.py's _FakeBoxEnv: constant
    reward=1 so substitution by the discriminator reward is easy to detect."""

    def __init__(self, num_envs: int = 3, episode_len: int = 5, obs_dim: int = 5) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self.obs_dim = obs_dim
        self._t = torch.zeros(num_envs, dtype=torch.long)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def _obs(self):
        return torch.randn(self.num_envs, self.obs_dim)

    def reset(self, seed=None):
        del seed
        self._t.zero_()
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.episode_len
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        reward = torch.ones(self.num_envs)
        self._t[terminated] = 0
        return self._obs(), reward, terminated, truncated, {}


def _fake_demo_loader(buffer, env_id):
    """Stand-in for load_d4rl_legacy_dataset_to_replay_buffer: fills the
    buffer with random transitions matching its own spaces, avoiding a real
    D4RL download in tests."""
    del env_id
    n = 64
    obs_shape = buffer.obs["state"].shape[2:]
    act_shape = buffer.actions.shape[2:]
    for _ in range(n):
        state = torch.randn(1, *obs_shape)
        next_state = torch.randn(1, *obs_shape)
        action = torch.rand(1, *act_shape) * 2 - 1
        reward = torch.zeros(1)
        done = torch.zeros(1)
        buffer.add({"state": state}, {"state": next_state}, action, reward, done)


def _gail_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "num_steps": 8,
        "num_minibatches": 1,
        "update_epochs": 1,
        "eval_freq": 0,
        "log_freq": 0,
        "target_kl": None,
        "net_arch": [16],
        "demo_env_id": "halfcheetah-expert-v2",
        "demo_buffer_size": 64,
        "demo_batch_size": 8,
        "n_disc_updates_per_round": 2,
        "disc_net_arch": (16,),
    }


def _build_gail(monkeypatch, env=None) -> GAIL:
    monkeypatch.setattr(
        "rl_garden.buffers.d4rl_legacy_dataset.load_d4rl_legacy_dataset_to_replay_buffer",
        _fake_demo_loader,
    )
    env = env or _FakeBoxEnv()
    return GAIL(env, **_gail_kwargs())


def test_gail_discriminator_forward_shape():
    action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
    disc = GAILDiscriminator(5, action_space, net_arch=(16,))
    features = torch.randn(7, 5)
    action = torch.randn(7, 2)
    logits = disc(features, action)
    assert logits.shape == (7,)


def test_gail_reward_wrapper_substitutes_reward():
    env = _FakeBoxEnv(num_envs=2)
    calls = []

    def reward_fn(obs, action):
        calls.append((obs.clone(), action.clone()))
        return torch.full((env.num_envs,), 42.0)

    wrapped = GAILRewardWrapper(env, reward_fn=reward_fn)
    obs0, _ = wrapped.reset()
    action = torch.zeros(env.num_envs, 2)
    obs1, reward, terminated, truncated, info = wrapped.step(action)

    assert torch.equal(reward, torch.full((env.num_envs,), 42.0))
    # reward_fn must have been called with the PRE-step obs, not obs1.
    assert torch.equal(calls[0][0], obs0)
    assert torch.equal(calls[0][1], action)
    # transparent attribute passthrough (num_envs etc.)
    assert wrapped.num_envs == env.num_envs


def test_gail_reward_wrapper_last_obs_is_a_clone_not_alias():
    env = _FakeBoxEnv(num_envs=1)
    wrapped = GAILRewardWrapper(env, reward_fn=lambda obs, action: torch.zeros(1))
    obs0, _ = wrapped.reset()
    obs0 += 100.0  # mutate the caller's copy
    assert not torch.equal(wrapped._last_obs, obs0)


class _IdentityCriticFeaturesPolicy:
    """Stand-in for a real BasePolicy: ``extract_critic_features`` is an
    identity passthrough of the ``"state"`` key, matching FlattenExtractor's
    actual behavior for a state-only Box observation with no normalization
    (see rl_garden/encoders/flatten.py) -- exactly what GAIL's default
    (shared, unconfigured) extractor produces."""

    def extract_critic_features(self, obs):
        return obs["state"]


def test_discriminator_reward_matches_log_sigmoid_formula():
    env = _FakeBoxEnv()

    class _DummyGAIL:
        device = torch.device("cpu")

        def __init__(self):
            self.policy = _IdentityCriticFeaturesPolicy()
            action_space = env.single_action_space
            self.discriminator = GAILDiscriminator(env.obs_dim, action_space, net_arch=(16,))

        _discriminator_reward = GAIL._discriminator_reward
        _obs_to_policy_device = GAIL._obs_to_policy_device

    dummy = _DummyGAIL()
    state = torch.randn(4, env.obs_dim)
    obs = {"state": state}
    action = torch.randn(4, 2)
    reward = dummy._discriminator_reward(obs, action)
    with torch.no_grad():
        logits = dummy.discriminator(state, action)
    expected = -F.logsigmoid(-logits)
    assert torch.allclose(reward, expected)


def test_discriminator_reward_handles_cpu_env_with_gpu_discriminator():
    """Regression: CPU-backed env backends (e.g. d4rl_legacy's mujoco_py)
    hand obs/action to the wrapper on CPU while the discriminator/policy
    live on self.device -- caught for real on a CUDA host running GAIL
    against d4rl_legacy/AntMaze (obs.device='cpu', discriminator on cuda)."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("requires CUDA to exercise the cross-device path")

    env = _FakeBoxEnv()

    class _DummyGAIL:
        device = torch.device("cuda")

        def __init__(self):
            self.policy = _IdentityCriticFeaturesPolicy()
            self.discriminator = GAILDiscriminator(
                env.obs_dim, env.single_action_space, net_arch=(16,)
            ).to(self.device)

        _discriminator_reward = GAIL._discriminator_reward
        _obs_to_policy_device = GAIL._obs_to_policy_device

    dummy = _DummyGAIL()
    obs = {"state": torch.randn(4, env.obs_dim)}  # CPU
    action = torch.randn(4, 2)  # CPU
    reward = dummy._discriminator_reward(obs, action)
    assert torch.isfinite(reward).all()


def test_train_discriminator_step_labels_and_loss_decreases(monkeypatch):
    agent = _build_gail(monkeypatch)
    obs_dim = agent.env.single_observation_space["state"].shape[0]
    gen_obs = {"state": torch.randn(8, obs_dim)}
    gen_actions = torch.rand(8, 2) * 2 - 1
    expert_obs = {"state": torch.randn(8, obs_dim) + 5.0}
    expert_actions = torch.rand(8, 2) * 2 - 1

    losses = []
    for _ in range(20):
        stats = agent._train_discriminator_step(gen_obs, gen_actions, expert_obs, expert_actions)
        losses.append(stats["disc_loss"])
    assert losses[-1] < losses[0]


def test_gail_learn_smoke_and_reward_substitution(monkeypatch):
    agent = _build_gail(monkeypatch)
    agent.learn(total_timesteps=agent.env.num_envs * 8 * 2)

    # rollout_buffer stores the substituted (discriminator) reward, not the
    # scripted env's raw reward of 1.0 everywhere.
    assert not torch.all(agent.rollout_buffer.rewards == 1.0)

    losses = agent.train()
    assert "disc_loss" in losses and "disc_acc" in losses
    assert torch.isfinite(torch.tensor(losses["disc_loss"]))
    assert torch.isfinite(torch.tensor(losses["disc_acc"]))


def test_gail_checkpoint_round_trip(tmp_path, monkeypatch):
    agent = _build_gail(monkeypatch)
    agent.learn(total_timesteps=agent.env.num_envs * 8 * 2)
    disc_state_before = {
        k: v.clone() for k, v in agent.discriminator.state_dict().items()
    }

    path = agent.save(tmp_path / "ckpt.pt", include_replay_buffer=False)

    resumed = _build_gail(monkeypatch)
    resumed.load(path, load_replay_buffer=False)
    disc_state_after = resumed.discriminator.state_dict()
    for key, value in disc_state_before.items():
        assert torch.equal(value, disc_state_after[key])

    assert resumed._checkpoint_metadata()["demo_env_id"] == agent.demo_env_id
    assert resumed._checkpoint_metadata()["disc_net_arch"] == list(agent.disc_net_arch)


def test_gail_registered_in_online_registry():
    from rl_garden.training.online._registry import registry

    registry.discover()
    assert "gail" in registry._entries


# --- asymmetric actor/critic encoders (policy-extractor-contract) ---


class _AsymmetricVecEnv:
    """A ``rgb_cam`` + ``state`` + ``state_object_pose`` env: ``rgb_cam``/
    ``state`` are shared by actor and critic (both get a real, trainable
    ``CombinedExtractor``); ``state_object_pose`` (Section A's ``state_<name>``
    family) is critic-only -- and, via GAIL's discriminator being a
    critic-role consumer, discriminator-only."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Dict(
            {
                "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "state_object_pose": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self._t = torch.zeros(num_envs, dtype=torch.long)

    def _obs(self):
        return {
            "rgb_cam": torch.randint(0, 256, (self.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(self.num_envs, 4),
            "state_object_pose": torch.randn(self.num_envs, 3),
        }

    def reset(self, seed=None):
        del seed
        self._t.zero_()
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        reward = torch.ones(self.num_envs)
        return self._obs(), reward, terminated, truncated, {}

    def close(self) -> None:
        return None


def _fake_asymmetric_demo_loader(buffer, env_id):
    """Like ``_fake_demo_loader``, but fills every key (``rgb_cam``/``state``/
    ``state_object_pose``) instead of just ``state`` -- this test's own
    stand-in for a demo pipeline that can supply privileged/image keys (see
    ``rl_garden/algorithms/gail.py``'s module docstring open question: the
    real ``load_d4rl_legacy_dataset_to_replay_buffer`` cannot do this)."""
    del env_id
    n = 64
    act_shape = buffer.actions.shape[2:]
    for _ in range(n):
        obs = {
            "rgb_cam": torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
            "state_object_pose": torch.randn(1, 3),
        }
        next_obs = {
            "rgb_cam": torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(1, 4),
            "state_object_pose": torch.randn(1, 3),
        }
        action = torch.rand(1, *act_shape) * 2 - 1
        reward = torch.zeros(1)
        done = torch.zeros(1)
        buffer.add(obs, next_obs, action, reward, done)


def test_gail_demo_buffer_raises_for_critic_keys_the_demo_dataset_cannot_provide():
    """``demo_dataset_backend="d4rl_legacy"`` only ever supplies a flat
    ``"state"`` key (``rl_garden.buffers._dataset_common._match_obs_to_buffer``
    wraps every flat loader output as ``{"state": ...}``); an
    ``obs_groups.critic`` asking for anything else must raise
    ``ObservationContractError`` naming it, not silently zero-fill it (see
    ``GAIL._build_demo_buffer``). Raises before ever calling the real loader,
    so no monkeypatch is needed here."""
    from rl_garden.observations import ObservationContractError

    with pytest.raises(ObservationContractError, match="state_object_pose"):
        GAIL(
            _AsymmetricVecEnv(),
            device="cpu",
            num_steps=8,
            num_minibatches=1,
            update_epochs=1,
            eval_freq=0,
            log_freq=0,
            target_kl=None,
            net_arch=[16],
            demo_env_id="halfcheetah-expert-v2",
            demo_buffer_size=64,
            demo_batch_size=8,
            n_disc_updates_per_round=1,
            disc_net_arch=(16,),
            encoder_config=EncoderConfig(proprio_latent_dim=4),
            obs_groups=ObsGroups(
                actor=("rgb_cam", "state"), critic=("rgb_cam", "state", "state_object_pose")
            ),
            encoder_sharing="separate",
        )


def test_gail_asymmetric_obs_groups_critic_sees_extra_state_actor_does_not(monkeypatch):
    """End-to-end asymmetric actor/critic encoders via state_<name>: the
    discriminator (critic-side -- see GAILDiscriminator's docstring) sees
    state_object_pose, the actor does not, encoder_sharing="separate".
    Asserts the actor extractor's schema lacks the key and, over one real
    rollout+update step (PPO update + discriminator update), the
    discriminator loss's gradient reaches only critic_extractor, never
    actor_extractor.

    Patches ``_build_demo_buffer`` itself (not just the loader) because a
    real ``demo_dataset_backend="d4rl_legacy"`` can never supply
    ``state_object_pose`` (see the raise test above) -- this test's own
    concern is the encoder-sharing/gradient-isolation wiring downstream of
    an already-built demo buffer, not the demo-loading contract."""

    def _fake_build_demo_buffer(self) -> "ReplayBuffer":
        from rl_garden.buffers.replay_buffer import ReplayBuffer

        buffer = ReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=1,
            buffer_size=self.demo_buffer_size,
            storage_device=self.device,
            sample_device=self.device,
        )
        _fake_asymmetric_demo_loader(buffer, self.demo_env_id)
        return buffer

    monkeypatch.setattr(GAIL, "_build_demo_buffer", _fake_build_demo_buffer)
    agent = GAIL(
        _AsymmetricVecEnv(),
        device="cpu",
        num_steps=8,
        num_minibatches=1,
        update_epochs=1,
        eval_freq=0,
        log_freq=0,
        target_kl=None,
        net_arch=[16],
        demo_env_id="halfcheetah-expert-v2",
        demo_buffer_size=64,
        demo_batch_size=8,
        n_disc_updates_per_round=1,
        disc_net_arch=(16,),
        encoder_config=EncoderConfig(proprio_latent_dim=4),
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

    # A real end-to-end rollout+update step (PPO policy/value update, then
    # the discriminator update) runs cleanly with the asymmetric schema.
    agent.learn(total_timesteps=agent.env.num_envs * agent.num_steps * 2)
    losses = agent.train()
    assert "disc_loss" in losses and "disc_acc" in losses

    # Discriminator-loss gradient isolation.
    gen_sample = next(agent.rollout_buffer.get(agent.demo_batch_size))
    expert_sample = agent._demo_buffer.sample(agent.demo_batch_size)
    gen_features = agent.policy.extract_critic_features(gen_sample.obs)
    expert_features = agent.policy.extract_critic_features(expert_sample.obs)
    features = torch.cat([expert_features, gen_features], dim=0)
    actions = torch.cat([expert_sample.actions, gen_sample.actions], dim=0)
    disc_loss = agent.discriminator(features, actions).sum()

    grad_on_critic = torch.autograd.grad(
        disc_loss, list(critic_extractor.parameters()), retain_graph=True, allow_unused=True
    )
    assert any(g is not None and torch.any(g != 0) for g in grad_on_critic)
    grad_on_actor = torch.autograd.grad(
        disc_loss, list(actor_extractor.parameters()), allow_unused=True
    )
    assert all(g is None for g in grad_on_actor)
