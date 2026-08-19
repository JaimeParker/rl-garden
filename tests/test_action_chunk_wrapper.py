from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.envs.wrappers import ActionChunkWrapper


class _FakeStaggeredEnv(gym.Env):
    """SAME_STEP-autoreset fake vector env: each env's ``state`` counts up by
    1 per step and resets to 0 the instant it hits its configured
    termination/truncation limit (``None`` = never)."""

    def __init__(self, term_limit: list[int | None], trunc_limit: list[int | None]) -> None:
        assert len(term_limit) == len(trunc_limit)
        self.num_envs = len(term_limit)
        self.term_limit = term_limit
        self.trunc_limit = trunc_limit
        self._value = torch.zeros(self.num_envs)
        self.single_action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (1,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def reset(self, *, seed=None, options=None):
        self._value.zero_()
        return {"state": self._value[:, None].clone()}, {}

    def step(self, action):
        del action
        self._value += 1
        terminated = torch.tensor(
            [lim is not None and self._value[i].item() >= lim for i, lim in enumerate(self.term_limit)]
        )
        truncated = torch.tensor(
            [lim is not None and self._value[i].item() >= lim for i, lim in enumerate(self.trunc_limit)]
        ) & ~terminated
        done = terminated | truncated
        reward = torch.ones(self.num_envs)
        final_value = self._value.clone()
        info: dict = {}
        if done.any():
            self._value[done] = 0.0
            info = {
                "final_observation": {"state": final_value[:, None].clone()},
                "_final_observation": done.clone(),
                "final_info": {"raw_value": final_value.clone()},
                "_final_info": done.clone(),
            }
        obs = {"state": self._value[:, None].clone()}
        return obs, reward, terminated, truncated, info


def test_action_space_gains_chunk_dimension():
    env = ActionChunkWrapper(_FakeStaggeredEnv([None], [None]), act_steps=4)
    assert env.single_action_space.shape == (4, 1)
    assert env.action_space.shape == (1, 4, 1)


def test_staggered_termination_and_truncation_within_one_chunk():
    # env0 terminates on sub-step 0, env1 terminates on sub-step 2, env2
    # never finishes, env3 truncates on sub-step 1.
    env = ActionChunkWrapper(
        _FakeStaggeredEnv(
            term_limit=[1, 3, None, None],
            trunc_limit=[None, None, None, 2],
        ),
        act_steps=4,
    )
    action = torch.zeros(4, 4, 1)
    obs, reward, terminated, truncated, infos = env.step(action)

    assert torch.equal(reward, torch.tensor([1.0, 3.0, 4.0, 2.0]))
    assert torch.equal(terminated, torch.tensor([True, True, False, False]))
    assert torch.equal(truncated, torch.tensor([False, False, False, True]))

    # Post-chunk "current" obs reflects however many steps happened after
    # each env's SAME_STEP autoreset (env0's term_limit=1 and env3's
    # trunc_limit=2 both keep re-triggering on later sub-steps too, so both
    # land back at a freshly-reset 0; env1/env2 don't re-trigger).
    assert torch.allclose(obs["state"].squeeze(-1), torch.tensor([0.0, 1.0, 4.0, 0.0]))

    done_mask = infos["_final_observation"]
    assert torch.equal(done_mask, torch.tensor([True, True, False, True]))
    final_state = infos["final_observation"]["state"].squeeze(-1)
    assert torch.allclose(final_state[done_mask], torch.tensor([1.0, 3.0, 2.0]))

    assert torch.equal(infos["_final_info"], done_mask)
    assert torch.allclose(infos["final_info"]["raw_value"][done_mask], torch.tensor([1.0, 3.0, 2.0]))


def test_act_steps_one_matches_single_inner_step():
    env = ActionChunkWrapper(_FakeStaggeredEnv([None, None], [None, None]), act_steps=1)
    action = torch.zeros(2, 1, 1)
    obs, reward, terminated, truncated, infos = env.step(action)
    assert torch.equal(reward, torch.tensor([1.0, 1.0]))
    assert not terminated.any() and not truncated.any()
    assert "final_observation" not in infos
