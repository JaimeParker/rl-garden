"""Tests for PolicyDistillation: on-policy teacher-student action regression.

Uses a plain BCPolicy as the mock teacher (not PPOPolicy) to exercise the
property that matters -- any BasePolicy (any algorithm) can be the teacher,
not just PPO (see rl_garden/algorithms/policy_distillation.py's module
docstring, "any algorithm can be the teacher").
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms.policy_distillation import PolicyDistillation
from rl_garden.buffers.distillation_rollout_buffer import DistillationRolloutBuffer
from rl_garden.encoders.combined import CombinedExtractor
from rl_garden.policies.base import BasePolicy
from rl_garden.policies.bc_policy import BCPolicy

PRIVILEGED_DIM = 6
PROPRIO_DIM = 4
ACTION_DIM = 2


class _FakeDualObsEnv:
    """Fixed-episode-length auto-reset fake vector env with two obs groups
    (privileged/proprio), same shape convention as test_dagger.py's
    _FakeEnv fixture."""

    def __init__(self, num_envs: int = 3, episode_len: int = 5) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self._step_count = torch.zeros(num_envs, dtype=torch.long)
        self.single_observation_space = spaces.Dict(
            {
                "privileged": spaces.Box(-np.inf, np.inf, (PRIVILEGED_DIM,), np.float32),
                "proprio": spaces.Box(-np.inf, np.inf, (PROPRIO_DIM,), np.float32),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def _obs(self):
        return {
            "privileged": torch.randn(self.num_envs, PRIVILEGED_DIM),
            "proprio": torch.randn(self.num_envs, PROPRIO_DIM),
        }

    def reset(self, seed=None):
        del seed
        self._step_count.zero_()
        return self._obs(), {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= self.episode_len
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        reward = torch.ones(self.num_envs)
        self._step_count[terminated] = 0
        return self._obs(), reward, terminated, truncated, {}


class _ConstantTeacherPolicy(BasePolicy):
    """Ignores its input, always predicts the same fixed action -- makes the
    student's regression target a learnable constant (deterministic w.r.t.
    the student's own obs) rather than noise uncorrelated with student obs,
    so "loss decreases" is a robust, non-flaky assertion."""

    def __init__(self, action_dim: int, value: float = 0.5) -> None:
        super().__init__()
        self._dummy_param = nn.Parameter(torch.zeros(1))
        self.action_dim = action_dim
        self.value = value

    def predict(self, obs, deterministic: bool = False) -> torch.Tensor:
        batch = next(iter(obs.values())).shape[0]
        return torch.full(
            (batch, self.action_dim), self.value, device=self._dummy_param.device
        )


def _make_bc_teacher() -> BCPolicy:
    obs_space = spaces.Dict(
        {"privileged": spaces.Box(-np.inf, np.inf, (PRIVILEGED_DIM,), np.float32)}
    )
    features_extractor = CombinedExtractor(
        observation_space=obs_space, image_keys=(), use_proprio=False
    )
    return BCPolicy(
        observation_space=obs_space,
        action_space=spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32),
        features_extractor=features_extractor,
        net_arch=[16, 16],
    )


def _make_agent(**kwargs) -> PolicyDistillation:
    defaults = dict(
        env=_FakeDualObsEnv(num_envs=3),
        teacher_policy=_ConstantTeacherPolicy(ACTION_DIM),
        teacher_obs_keys=["privileged"],
        student_obs_keys=["proprio"],
        num_steps=8,
        num_learning_epochs=2,
        num_minibatches=2,
        actor_lr=1e-2,
        net_arch=[16, 16],
        device="cpu",
        eval_freq=0,
        log_freq=0,
    )
    defaults.update(kwargs)
    return PolicyDistillation(**defaults)


def test_teacher_from_any_algorithm_bc_policy():
    """Confirms PolicyDistillation accepts a non-PPO BasePolicy as teacher."""
    agent = _make_agent(teacher_policy=_make_bc_teacher())
    assert isinstance(agent.teacher_policy, BCPolicy)


def test_teacher_is_frozen_and_unchanged_after_training():
    agent = _make_agent()
    for param in agent.teacher_policy.parameters():
        assert not param.requires_grad
    before = [p.clone() for p in agent.teacher_policy.parameters()]

    agent.learn(total_timesteps=agent.num_steps * agent.num_envs * 3)

    for param in agent.teacher_policy.parameters():
        assert not param.requires_grad
    after = list(agent.teacher_policy.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_student_loss_decreases_toward_constant_teacher():
    agent = _make_agent(teacher_policy=_ConstantTeacherPolicy(ACTION_DIM, value=0.7))
    agent.learn(total_timesteps=agent.num_steps * agent.num_envs * 20)
    obs, _ = agent.env.reset(seed=agent.seed)
    student_obs = {"proprio": obs["proprio"]}
    with torch.no_grad():
        pred = agent.policy.predict(student_obs, deterministic=True)
    assert torch.allclose(pred, torch.full_like(pred, 0.7), atol=0.1)


def test_distillation_buffer_is_fixed_length_not_growing():
    obs_space = spaces.Dict(
        {
            "privileged": spaces.Box(-np.inf, np.inf, (PRIVILEGED_DIM,), np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, (PROPRIO_DIM,), np.float32),
        }
    )
    action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
    buffer = DistillationRolloutBuffer(
        observation_space=obs_space,
        action_space=action_space,
        num_steps=4,
        num_envs=2,
        device="cpu",
    )
    assert buffer.buffer_size == 8
    for _ in range(4):
        obs = {
            "privileged": torch.randn(2, PRIVILEGED_DIM),
            "proprio": torch.randn(2, PROPRIO_DIM),
        }
        buffer.add(obs, torch.zeros(2, ACTION_DIM), torch.zeros(2))
    assert buffer.full
    with pytest.raises(RuntimeError, match="is full"):
        buffer.add(obs, torch.zeros(2, ACTION_DIM), torch.zeros(2))

    buffer.reset()
    assert not buffer.full
    assert buffer.pos == 0


def test_checkpoint_round_trip_saves_student_not_teacher(tmp_path):
    agent = _make_agent()
    agent.learn(total_timesteps=agent.num_steps * agent.num_envs)
    ckpt_path = agent.save(tmp_path / "checkpoint.pt", include_replay_buffer=False)

    raw = torch.load(ckpt_path, map_location="cpu")
    assert set(raw["state"]["policy"]) == set(agent.policy.state_dict())
    assert "teacher_policy" not in raw["state"]

    resumed = _make_agent(env=agent.env)
    resumed.load(ckpt_path, load_replay_buffer=False)
    for original, loaded in zip(
        agent.policy.state_dict().values(), resumed.policy.state_dict().values()
    ):
        assert torch.equal(original, loaded)
