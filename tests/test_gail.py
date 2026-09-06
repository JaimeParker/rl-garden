from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms.gail import GAIL
from rl_garden.envs.wrappers.gail_reward import GAILRewardWrapper
from rl_garden.networks.discriminator import GAILDiscriminator


class _FakeBoxEnv(gym.vector.VectorEnv):
    """Mirrors tests/test_ppo_normalize_obs.py's _FakeBoxEnv: constant
    reward=1 so substitution by the discriminator reward is easy to detect.
    Subclasses gym.vector.VectorEnv (not plain duck-typing) because
    GAILRewardWrapper is a gym.vector.VectorWrapper, which asserts this."""

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
    obs_shape = buffer.obs.shape[2:]
    act_shape = buffer.actions.shape[2:]
    for _ in range(n):
        obs = torch.randn(1, *obs_shape)
        next_obs = torch.randn(1, *obs_shape)
        action = torch.rand(1, *act_shape) * 2 - 1
        reward = torch.zeros(1)
        done = torch.zeros(1)
        buffer.add(obs, next_obs, action, reward, done)


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
    obs_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
    action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
    disc = GAILDiscriminator(obs_space, action_space, net_arch=(16,))
    obs = torch.randn(7, 5)
    action = torch.randn(7, 2)
    logits = disc(obs, action)
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


def test_discriminator_reward_matches_log_sigmoid_formula():
    env = _FakeBoxEnv()

    class _DummyGAIL:
        device = torch.device("cpu")

        def __init__(self):
            obs_space = env.single_observation_space
            action_space = env.single_action_space
            self.discriminator = GAILDiscriminator(obs_space, action_space, net_arch=(16,))

        _discriminator_reward = GAIL._discriminator_reward
        _obs_to_policy_device = GAIL._obs_to_policy_device

    dummy = _DummyGAIL()
    obs = torch.randn(4, env.obs_dim)
    action = torch.randn(4, 2)
    reward = dummy._discriminator_reward(obs, action)
    with torch.no_grad():
        logits = dummy.discriminator(obs, action)
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
            self.discriminator = GAILDiscriminator(
                env.single_observation_space, env.single_action_space, net_arch=(16,)
            ).to(self.device)

        _discriminator_reward = GAIL._discriminator_reward
        _obs_to_policy_device = GAIL._obs_to_policy_device

    dummy = _DummyGAIL()
    obs = torch.randn(4, env.obs_dim)  # CPU
    action = torch.randn(4, 2)  # CPU
    reward = dummy._discriminator_reward(obs, action)
    assert torch.isfinite(reward).all()


def test_train_discriminator_step_labels_and_loss_decreases(monkeypatch):
    agent = _build_gail(monkeypatch)
    obs_dim = agent.env.single_observation_space.shape[0]
    gen_obs = torch.randn(8, obs_dim)
    gen_actions = torch.rand(8, 2) * 2 - 1
    expert_obs = torch.randn(8, obs_dim) + 5.0
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
