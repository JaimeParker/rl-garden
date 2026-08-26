"""Tests for DAgger: composes BC (unmodified) with DemoInterventionMixin's
growing buffer (unmodified) for interactive imitation learning."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms.dagger import DAgger

OBS_DIM = 4
ACTION_DIM = 2


class _FakeEnv:
    """Fixed-episode-length auto-reset fake vector env, same shape as
    test_acfql_smoke.py's fixture."""

    def __init__(self, num_envs: int = 3, episode_len: int = 5) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self._step_count = torch.zeros(num_envs, dtype=torch.long)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def reset(self, seed=None):
        del seed
        self._step_count.zero_()
        return torch.randn(self.num_envs, OBS_DIM), {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= self.episode_len
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        reward = torch.ones(self.num_envs)
        self._step_count[terminated] = 0
        obs = torch.randn(self.num_envs, OBS_DIM)
        return obs, reward, terminated, truncated, {}


class _ConstantExpert:
    """Deterministic mock expert: always returns a fixed, easily-identified
    action so tests can distinguish expert-labeled data from policy actions."""

    def __init__(self, value: float = 0.75, action_dim: int = ACTION_DIM) -> None:
        self.value = value
        self.action_dim = action_dim
        self.call_count = 0

    def __call__(self, obs) -> torch.Tensor:
        self.call_count += 1
        batch = obs.shape[0]
        return torch.full((batch, self.action_dim), self.value)


def _make_agent(**kwargs) -> DAgger:
    defaults = dict(
        env=_FakeEnv(num_envs=3),
        expert=_ConstantExpert(),
        demo_buffer_size=1000,
        rollout_steps_per_round=10,
        gradient_steps_per_round=2,
        batch_size=8,
        buffer_device="cpu",
        device="cpu",
        net_arch=[16, 16],
    )
    defaults.update(kwargs)
    return DAgger(**defaults)


def test_mro_puts_mixin_before_bc():
    mro_names = [c.__name__ for c in DAgger.__mro__]
    assert mro_names.index("DemoInterventionMixin") < mro_names.index("BC")


def test_sample_train_batch_resolves_to_mixing_version():
    agent = _make_agent()
    # BC's own _sample_train_batch takes no args; the mixin's takes
    # batch_size explicitly. If MRO were wrong this call would raise.
    agent.collect_round(num_steps=5, beta=1.0)
    data = agent._sample_train_batch(agent.batch_size)
    assert data.actions.shape[0] == agent.batch_size


def test_online_replay_buffer_stays_empty_offline_buffer_grows():
    agent = _make_agent()
    assert len(agent.replay_buffer) == 0
    agent.collect_round(num_steps=10, beta=1.0)
    assert len(agent.replay_buffer) == 0  # DAgger never writes here
    assert len(agent.offline_replay_buffer) == 10 * agent.num_envs


def test_collect_round_always_labels_with_expert_action_even_when_beta_low():
    expert = _ConstantExpert(value=0.75)
    agent = _make_agent(expert=expert, beta_rounds=1)
    beta = 0.0  # policy drives execution entirely, expert only labels
    agent.collect_round(num_steps=20, beta=beta)
    assert expert.call_count == 20  # expert queried every step regardless of beta

    n = len(agent.offline_replay_buffer)
    sample = agent.offline_replay_buffer.sample(n)
    assert torch.allclose(sample.actions, torch.full_like(sample.actions, 0.75))


def test_beta_schedule_decays_linearly_to_zero():
    agent = _make_agent(beta_rounds=10)
    assert agent.beta_schedule(0) == 1.0
    assert agent.beta_schedule(5) == 0.5
    assert agent.beta_schedule(10) == 0.0
    assert agent.beta_schedule(20) == 0.0  # clamped, not negative


def test_learn_runs_rounds_and_grows_demo_buffer():
    agent = _make_agent(rollout_steps_per_round=5, gradient_steps_per_round=2, beta_rounds=3)
    total_timesteps = 5 * agent.num_envs * 4  # 4 rounds worth
    agent.learn(total_timesteps)
    assert agent._round_num == 4
    assert len(agent.offline_replay_buffer) == 5 * agent.num_envs * 4
    assert len(agent.replay_buffer) == 0


def test_dagger_policy_is_plain_bc_policy_class():
    # Confirms DAgger reuses BC's policy machinery unmodified rather than
    # introducing a new policy type.
    from rl_garden.policies.bc_policy import BCPolicy

    agent = _make_agent()
    assert isinstance(agent.policy, BCPolicy)
