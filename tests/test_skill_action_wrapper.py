from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.envs.wrappers import SkillActionWrapper
from rl_garden.networks.actor_critic import UnsquashedGaussianActor


class _FakeStaggeredEnv(gym.Env):
    """SAME_STEP-autoreset fake vector env: each env's ``state`` counts up by
    1 per step and resets to 0 the instant it hits its configured
    termination/truncation limit (``None`` = never)."""

    def __init__(self, term_limit: list[int | None], trunc_limit: list[int | None]) -> None:
        assert len(term_limit) == len(trunc_limit)
        self.num_envs = len(term_limit)
        self.term_limit = term_limit
        self.trunc_limit = trunc_limit
        self._value = torch.zeros(self.num_envs, 1)
        self.single_action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (1,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def reset(self, *, seed=None, options=None):
        self._value.zero_()
        return self._value.clone(), {}

    def step(self, action):
        del action
        self._value += 1
        terminated = torch.tensor(
            [lim is not None and self._value[i, 0].item() >= lim for i, lim in enumerate(self.term_limit)]
        )
        truncated = torch.tensor(
            [lim is not None and self._value[i, 0].item() >= lim for i, lim in enumerate(self.trunc_limit)]
        ) & ~terminated
        done = terminated | truncated
        reward = torch.ones(self.num_envs)
        final_value = self._value.clone()
        info: dict = {}
        if done.any():
            self._value[done] = 0.0
            info = {
                "final_observation": final_value.clone(),
                "_final_observation": done.clone(),
            }
        return self._value.clone(), reward, terminated, truncated, info


class _RecordingDecoder:
    """Minimal decoder stand-in recording ``features`` at each call."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.calls: list[torch.Tensor] = []

    def action_log_prob(self, features):
        self.calls.append(features.clone())
        return torch.zeros(features.shape[0], self.action_dim), None

    def deterministic_action(self, features):
        self.calls.append(features.clone())
        return torch.zeros(features.shape[0], self.action_dim)


def test_action_space_is_box_skill_dim():
    env = SkillActionWrapper(
        _FakeStaggeredEnv([None], [None]), _RecordingDecoder(1), horizon=4, skill_dim=3,
    )
    assert env.single_action_space.shape == (3,)
    assert env.action_space.shape == (1, 3)


def test_decoder_receives_current_obs_each_substep():
    decoder = _RecordingDecoder(1)
    env = SkillActionWrapper(
        _FakeStaggeredEnv([None], [None]), decoder, horizon=3, skill_dim=2,
    )
    env.reset()
    skill = torch.tensor([[0.5, -0.5]])
    env.step(skill)

    assert len(decoder.calls) == 3
    # Each call's obs portion (all but the last skill_dim columns) should be
    # the *current* obs (incrementing 1, 2, 3), not the macro-step's
    # original obs (which would be 0 every call).
    obs_values = [call[0, 0].item() for call in decoder.calls]
    assert obs_values == [0.0, 1.0, 2.0]
    for call in decoder.calls:
        assert torch.equal(call[:, 1:], skill)


def test_staggered_termination_freezes_final_obs_and_masks_reward():
    decoder = _RecordingDecoder(1)
    env = SkillActionWrapper(
        _FakeStaggeredEnv(term_limit=[1, 3, None], trunc_limit=[None, None, None]),
        decoder, horizon=4, skill_dim=1,
    )
    env.reset()
    skill = torch.zeros(3, 1)
    obs, reward, terminated, truncated, infos = env.step(skill)

    # env0 terminates at sub-step 1 (reward=1), env1 at sub-step 3
    # (reward=3), env2 never (reward=4).
    assert torch.equal(reward, torch.tensor([1.0, 3.0, 4.0]))
    assert torch.equal(terminated, torch.tensor([True, True, False]))
    assert not truncated.any()

    done_mask = infos["_final_observation"]
    assert torch.equal(done_mask, torch.tensor([True, True, False]))
    final_obs = infos["final_observation"][done_mask, 0]
    assert torch.allclose(final_obs, torch.tensor([1.0, 3.0]))


def test_deterministic_flag_controls_repeatability():
    action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
    decoder = UnsquashedGaussianActor(2, action_space, [8], std_parameterization="exp")

    features = torch.randn(4, 2)
    torch.manual_seed(0)
    det1 = decoder.deterministic_action(features)
    torch.manual_seed(1)
    det2 = decoder.deterministic_action(features)
    assert torch.equal(det1, det2)

    torch.manual_seed(0)
    stoch1, _ = decoder.action_log_prob(features)
    torch.manual_seed(1)
    stoch2, _ = decoder.action_log_prob(features)
    assert not torch.equal(stoch1, stoch2)
