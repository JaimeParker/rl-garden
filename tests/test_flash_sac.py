"""Unit tests for FlashSAC components (CPU, no ManiSkill dependency)."""
from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from unittest.mock import MagicMock, patch

from rl_garden.algorithms.flash_sac import (
    FlashSAC,
    _build_truncated_zeta_cdf,
    _compute_categorical_td_target,
    _sample_integer_from_cdf,
    _select_min_q_log_probs,
)
from rl_garden.buffers.nstep_tensor_buffer import NStepTensorReplayBuffer
from rl_garden.common.distributions import safe_tanh_log_det_jacobian
from rl_garden.common.types import NStepReplayBufferSample
from rl_garden.networks.flash_sac_layers import UnitLinear
from rl_garden.policies.flash_sac_policy import FlashSACPolicy


OBS_DIM = 20
ACT_DIM = 7
NUM_ENVS = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def obs_space():
    return spaces.Box(-1, 1, (OBS_DIM,), np.float32)


@pytest.fixture
def act_space():
    return spaces.Box(-1, 1, (ACT_DIM,), np.float32)


@pytest.fixture
def policy(obs_space, act_space):
    return FlashSACPolicy(
        observation_space=obs_space,
        action_space=act_space,
        actor_hidden_dim=32,
        actor_num_blocks=1,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        num_bins=11,
        min_v=-2.0,
        max_v=2.0,
    )


class _FakeEnv:
    def __init__(self):
        self.num_envs = NUM_ENVS
        self.single_observation_space = spaces.Box(-1, 1, (OBS_DIM,), np.float32)
        self.single_action_space = spaces.Box(-1, 1, (ACT_DIM,), np.float32)
        self.observation_space = spaces.Box(-1, 1, (NUM_ENVS, OBS_DIM), np.float32)
        self.action_space = spaces.Box(-1, 1, (NUM_ENVS, ACT_DIM), np.float32)

    def reset(self, seed=None):
        return torch.randn(NUM_ENVS, OBS_DIM), {}

    def step(self, actions):
        return (
            torch.randn(NUM_ENVS, OBS_DIM),
            torch.randn(NUM_ENVS),
            torch.zeros(NUM_ENVS, dtype=torch.bool),
            torch.zeros(NUM_ENVS, dtype=torch.bool),
            {},
        )


@pytest.fixture
def agent():
    return FlashSAC(
        env=_FakeEnv(),
        eval_env=None,
        buffer_size=400,
        buffer_device="cpu",
        learning_starts=0,
        batch_size=8,
        gamma=0.99,
        tau=0.005,
        training_freq=10,
        utd=1.0,
        n_step=3,
        actor_hidden_dim=32,
        actor_num_blocks=1,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        num_bins=11,
        min_v=-2.0,
        max_v=2.0,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        seed=1,
        device="cpu",
        std_log=False,
        log_freq=0,
        eval_freq=0,
        save_final_checkpoint=False,
    )


def _fill_buffer(agent: FlashSAC, n_transitions: int = 50) -> None:
    obs = torch.randn(NUM_ENVS, OBS_DIM)
    for i in range(n_transitions):
        next_obs = torch.randn(NUM_ENVS, OBS_DIM)
        actions = torch.randn(NUM_ENVS, ACT_DIM)
        rewards = torch.randn(NUM_ENVS)
        done = torch.zeros(NUM_ENVS, dtype=torch.bool)
        ep_end = torch.zeros(NUM_ENVS, dtype=torch.bool)
        if i % 10 == 9:
            ep_end[:] = True
        agent.replay_buffer.add(obs, next_obs, actions, rewards, done, ep_end)
        obs = next_obs


# ---------------------------------------------------------------------------
# safe_tanh_log_det_jacobian
# ---------------------------------------------------------------------------


def test_safe_tanh_log_det_jacobian_shape():
    x = torch.randn(8, ACT_DIM)
    j = safe_tanh_log_det_jacobian(x)
    assert j.shape == (8, ACT_DIM)


def test_safe_tanh_log_det_jacobian_matches_formula():
    x = torch.randn(4, 3)
    expected = torch.log(1.0 - torch.tanh(x) ** 2 + 1e-6)
    result = safe_tanh_log_det_jacobian(x)
    assert torch.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# FlashSACPolicy
# ---------------------------------------------------------------------------


def test_policy_predict_stochastic(policy):
    obs = torch.randn(8, OBS_DIM)
    actions = policy.predict(obs, deterministic=False)
    assert actions.shape == (8, ACT_DIM)
    assert actions.abs().max() <= 1.0 + 1e-5  # tanh bound


def test_policy_predict_deterministic(policy):
    obs = torch.randn(8, OBS_DIM)
    a1 = policy.predict(obs, deterministic=True)
    a2 = policy.predict(obs, deterministic=True)
    assert torch.allclose(a1, a2)


def test_policy_critic_output_shapes(policy):
    obs = torch.randn(8, OBS_DIM)
    actions = torch.randn(8, ACT_DIM)
    qs, infos = policy.critic(obs, actions, training=True)
    assert qs.shape == (2, 8)
    assert infos["log_prob"].shape == (2, 8, 11)


def test_policy_min_q_value_shape(policy):
    obs = torch.randn(8, OBS_DIM)
    actions = torch.randn(8, ACT_DIM)
    min_q = policy.min_q_value(obs, actions, target=False)
    assert min_q.shape == (8, 1)


def test_policy_normalize_weights_called(policy):
    called = []
    # Monkey-patch one layer's normalize_parameters to track calls
    orig = policy.actor.embedder.norm.normalize_parameters
    policy.actor.embedder.norm.normalize_parameters = lambda: (called.append(1), orig())[1]
    policy.normalize_weights()
    assert len(called) == 1, "normalize_parameters was not called"


# ---------------------------------------------------------------------------
# NStepTensorReplayBuffer
# ---------------------------------------------------------------------------


def test_nstep_tensor_buffer_sample(obs_space, act_space):
    buf = NStepTensorReplayBuffer(
        observation_space=obs_space,
        action_space=act_space,
        num_envs=NUM_ENVS,
        buffer_size=200,
        nstep=3,
        gamma=0.99,
        storage_device="cpu",
        sample_device="cpu",
    )
    # Fill
    for ep in range(5):
        for step in range(20):
            obs = torch.randn(NUM_ENVS, OBS_DIM)
            next_obs = torch.randn(NUM_ENVS, OBS_DIM)
            action = torch.randn(NUM_ENVS, ACT_DIM)
            reward = torch.randn(NUM_ENVS)
            done = torch.zeros(NUM_ENVS, dtype=torch.bool)
            ep_end = torch.zeros(NUM_ENVS, dtype=torch.bool)
            if step == 19:
                ep_end[:] = True
            buf.add(obs, next_obs, action, reward, done, ep_end)

    sample = buf.sample(16)
    assert isinstance(sample, NStepReplayBufferSample)
    assert sample.obs.shape == (16, OBS_DIM)
    assert sample.discounts.shape == (16,)
    # Discounts should be gamma^n or gamma^k for k < n
    assert sample.discounts.min() >= 0.0
    assert sample.discounts.max() <= 0.99 + 1e-5


def test_nstep_discount_is_zero_when_terminal(obs_space, act_space):
    buf = NStepTensorReplayBuffer(
        observation_space=obs_space,
        action_space=act_space,
        num_envs=1,
        buffer_size=200,
        nstep=3,
        gamma=0.99,
        storage_device="cpu",
        sample_device="cpu",
    )
    # Insert: terminal at step 0
    obs = torch.randn(1, OBS_DIM)
    next_obs = torch.randn(1, OBS_DIM)
    buf.add(obs, next_obs, torch.randn(1, ACT_DIM), torch.ones(1), torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    # More non-terminal steps
    for _ in range(20):
        buf.add(obs, next_obs, torch.randn(1, ACT_DIM), torch.ones(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
    # Sample until we get the terminal — discount must be 0
    found_zero = False
    for _ in range(200):
        s = buf.sample(1)
        if s.dones.item():
            assert s.discounts.item() == 0.0
            found_zero = True
            break
    assert found_zero, "Did not sample a terminal transition after 200 tries"


# ---------------------------------------------------------------------------
# Categorical TD helpers
# ---------------------------------------------------------------------------


def test_categorical_td_target_sum_to_one():
    B, bins = 16, 11
    log_probs = torch.randn(B, bins).log_softmax(-1)
    reward = torch.randn(B)
    discounts = torch.ones(B) * 0.97
    actor_entropy = torch.zeros(B)
    probs = _compute_categorical_td_target(log_probs, reward, discounts, actor_entropy, bins, -2.0, 2.0)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)


def test_select_min_q_log_probs_picks_min():
    B, bins = 8, 11
    qs = torch.tensor([[1.0] * B, [2.0] * B])  # critic 0 always smaller
    log_probs = torch.stack([torch.ones(B, bins), torch.zeros(B, bins)])
    selected = _select_min_q_log_probs(qs, log_probs)
    assert selected.shape == (B, bins)
    assert torch.allclose(selected, torch.ones(B, bins))  # should pick critic 0


# ---------------------------------------------------------------------------
# Zeta noise
# ---------------------------------------------------------------------------


def test_zeta_cdf_ends_at_one():
    cdf = _build_truncated_zeta_cdf(2.0, 16)
    assert abs(float(cdf[-1]) - 1.0) < 1e-5


def test_zeta_samples_in_range():
    cdf = _build_truncated_zeta_cdf(2.0, 16)
    samples = [int(_sample_integer_from_cdf(cdf).item()) for _ in range(200)]
    assert min(samples) >= 1
    assert max(samples) <= 16


# ---------------------------------------------------------------------------
# FlashSAC.train()
# ---------------------------------------------------------------------------


def test_train_produces_finite_losses(agent):
    _fill_buffer(agent)
    info = agent.train(gradient_steps=2, compute_info=True)
    assert "critic/loss" in info
    assert torch.isfinite(torch.tensor(info["critic/loss"]))
    assert "actor/loss" in info
    assert torch.isfinite(torch.tensor(info["actor/loss"]))
    assert "temperature/value" in info
    assert info["temperature/value"] > 0


def test_temperature_gradient_direction():
    """alpha * (H - H_target) has the correct gradient sign.

    H > H_target → gradient positive → alpha decreases (policy too entropic).
    H < H_target → gradient negative → alpha increases (policy too deterministic).
    """
    import torch.nn as nn

    target_entropy = -float(ACT_DIM)

    # H > H_target: gradient should be positive → descent decreases alpha
    log_temp = nn.Parameter(torch.tensor(0.0))
    loss = log_temp.exp() * (torch.tensor(0.5) - target_entropy)  # 0.5 > -7
    loss.backward()
    assert log_temp.grad.item() > 0, "Gradient should be positive when entropy > target"

    # H < H_target: gradient should be negative → descent increases alpha
    log_temp2 = nn.Parameter(torch.tensor(0.0))
    loss2 = log_temp2.exp() * (torch.tensor(-10.0) - target_entropy)  # -10 < -7
    loss2.backward()
    assert log_temp2.grad.item() < 0, "Gradient should be negative when entropy < target"


def test_normalize_weights_called_after_train(agent):
    _fill_buffer(agent)
    called = []
    orig = agent.policy.normalize_weights
    agent.policy.normalize_weights = lambda: (called.append(1), orig())
    agent.train(gradient_steps=1, compute_info=False)
    # Called at least once for critic, once for actor
    assert len(called) >= 2


def test_target_network_updated_after_train(agent):
    _fill_buffer(agent)
    # Copy target params before update
    before = [p.clone() for p in agent.policy.critic_target.parameters()]
    agent.train(gradient_steps=1, compute_info=False)
    after = list(agent.policy.critic_target.parameters())
    # Some params should change (polyak update moves them toward critic)
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed, "Target network was not updated by polyak"


def test_zeta_noise_repeats_for_same_env(agent):
    """Same environment should produce identical actions within a noise repeat window."""
    obs = torch.randn(NUM_ENVS, OBS_DIM)

    # First call: force reinit by setting count=0
    agent._zeta_count[:] = 0
    actions1, _, _ = agent._rollout_action(obs, learning_has_started=True)
    assert agent._zeta_count.min().item() == 1

    # Force _zeta_n to be large so the second call definitely does NOT reinit
    agent._zeta_n[:] = 100
    actions2, _, _ = agent._rollout_action(obs, learning_has_started=True)
    assert torch.allclose(actions1, actions2, atol=1e-5), "Noise was not repeated"
