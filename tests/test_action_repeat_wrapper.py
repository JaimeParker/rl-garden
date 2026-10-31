from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.envs.wrappers import ActionRepeatWrapper


class _FakeStaggeredEnv(gym.Env):
    """SAME_STEP-autoreset fake vector env without host syncs: each env's
    ``state`` counts up by 1 per step and resets to 0 the instant it hits its
    termination/truncation limit (``None`` = never). Reward is 1 per step.

    By default ``final_info``/``final_observation`` are reported on every
    step like the Isaac Lab adapter (valid on done rows only);
    ``final_info_every_step=False`` reports them only on steps with a done
    env. ``reuse_buffers=True`` writes them into one persistent tensor in
    place, like Isaac Lab's buffers. ``logs`` maps a step count (1-based) to
    the ``infos["log"]`` dict reported on that step."""

    def __init__(
        self,
        term_limit: list[int | None],
        trunc_limit: list[int | None],
        *,
        device: str = "cpu",
        final_observation: bool = True,
        final_info_every_step: bool = True,
        reuse_buffers: bool = False,
        logs: dict[int, dict] | None = None,
    ) -> None:
        assert len(term_limit) == len(trunc_limit)
        self.num_envs = len(term_limit)
        self.device = torch.device(device)
        never = float("inf")
        self._term = torch.tensor([never if lim is None else lim for lim in term_limit], device=device)
        self._trunc = torch.tensor([never if lim is None else lim for lim in trunc_limit], device=device)
        self._value = torch.zeros(self.num_envs, device=device)
        self._final_value = torch.zeros(self.num_envs, device=device)
        self._final_step = torch.zeros(self.num_envs, device=device)
        self._final_observation = final_observation
        self._final_info_every_step = final_info_every_step
        self._reuse_buffers = reuse_buffers
        self._logs = logs or {}
        self.num_steps = 0
        self.post_update_sync_calls = 0
        self.single_action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (1,), np.float32)})
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def post_update_sync(self) -> None:
        self.post_update_sync_calls += 1

    def reset(self, *, seed=None, options=None):
        self._value = torch.zeros_like(self._value)
        return {"state": self._value[:, None].clone()}, {}

    def step(self, action):
        del action
        self.num_steps += 1
        self._value = self._value + 1
        terminated = self._value >= self._term
        truncated = (self._value >= self._trunc) & ~terminated
        done = terminated | truncated
        if self._reuse_buffers:
            final_value = self._final_value
            final_value.copy_(self._value)
            final_step = self._final_step
            final_step.fill_(float(self.num_steps))
        else:
            final_value = self._value.clone()
            final_step = torch.full((self.num_envs,), float(self.num_steps), device=self.device)
        self._value = torch.where(done, torch.zeros_like(self._value), self._value)
        info: dict = {"step_count": torch.full((self.num_envs,), float(self.num_steps), device=self.device)}
        if self._final_info_every_step or bool(done.any()):
            info["final_info"] = {"raw_value": final_value, "step": final_step}
            info["_final_info"] = done
            if self._final_observation:
                info["final_observation"] = {"state": final_value[:, None]}
        if self.num_steps in self._logs:
            info["log"] = self._logs[self.num_steps]
        obs = {"state": self._value[:, None].clone()}
        reward = torch.ones(self.num_envs, device=self.device)
        return obs, reward, terminated, truncated, info


def _staggered_env(**kwargs) -> _FakeStaggeredEnv:
    # env0 terminates on sub-step 0 and again on 1, 2, 3; env1 terminates on
    # sub-step 2; env2 never ends; env3 truncates on sub-step 1 and again on 3.
    return _FakeStaggeredEnv(term_limit=[1, 3, None, None], trunc_limit=[None, None, None, 2], **kwargs)


def _action(env: gym.Env) -> torch.Tensor:
    return torch.zeros(env.num_envs, 1, device=env.device)


def test_repeat_below_one_raises() -> None:
    with pytest.raises(ValueError, match="repeat must be >= 1"):
        ActionRepeatWrapper(_staggered_env(), repeat=0)


@pytest.mark.parametrize("reuse_buffers", [False, True], ids=["fresh-tensors", "reused-buffers"])
def test_staggered_episode_ends_freeze_at_the_ending_sub_step(reuse_buffers: bool) -> None:
    env = ActionRepeatWrapper(_staggered_env(reuse_buffers=reuse_buffers), repeat=4)

    obs, reward, terminated, truncated, infos = env.step(_action(env))

    assert torch.equal(reward, torch.tensor([1.0, 3.0, 4.0, 2.0]))
    assert torch.equal(terminated, torch.tensor([True, True, False, False]))
    assert torch.equal(truncated, torch.tensor([False, False, False, True]))
    assert torch.equal(obs["state"].squeeze(-1), torch.tensor([0.0, 1.0, 4.0, 0.0]))
    ended = torch.tensor([True, True, False, True])
    assert "_final_observation" not in infos
    assert torch.equal(infos["_final_info"], ended)
    final_state = infos["final_observation"]["state"].squeeze(-1)
    assert torch.equal(final_state[ended], torch.tensor([1.0, 3.0, 2.0]))
    # The row that did not end holds the returned observation.
    assert final_state[~ended].tolist() == obs["state"].squeeze(-1)[~ended].tolist()
    assert torch.equal(infos["final_info"]["raw_value"][ended], torch.tensor([1.0, 3.0, 2.0]))
    assert torch.equal(infos["final_info"]["step"][ended], torch.tensor([1.0, 3.0, 2.0]))
    # Non-final keys come from the last sub-step.
    assert torch.equal(infos["step_count"], torch.full((4,), 4.0))
    assert "log" not in infos


def test_second_episode_end_in_a_chunk_is_counted_and_not_cleared_by_reset() -> None:
    env = ActionRepeatWrapper(_staggered_env(), repeat=4)

    env.step(_action(env))
    assert env.dropped_episode_ends.dtype == torch.int64
    assert env.dropped_episode_ends.tolist() == [3, 0, 0, 1]

    env.reset()
    env.step(_action(env))
    assert env.dropped_episode_ends.tolist() == [6, 0, 0, 2]


def test_no_final_observation_when_the_inner_env_reports_none() -> None:
    env = ActionRepeatWrapper(_staggered_env(final_observation=False), repeat=4)

    infos = env.step(_action(env))[4]

    assert "final_observation" not in infos and "_final_observation" not in infos
    assert torch.equal(infos["_final_info"], torch.tensor([True, True, False, True]))


def test_log_from_a_middle_sub_step_is_forwarded() -> None:
    log = {"Metrics/success": torch.tensor(0.5)}
    env = ActionRepeatWrapper(_staggered_env(logs={2: log}), repeat=4)

    first = env.step(_action(env))[4]
    second = env.step(_action(env))[4]

    assert first["log"] is log
    assert "log" not in second


def test_latest_log_dict_in_a_chunk_is_forwarded_whole() -> None:
    earlier = {"Metrics/success": torch.tensor(0.5), "Metrics/only_earlier": torch.tensor(1.0)}
    later = {"Metrics/success": torch.tensor(0.75)}
    env = ActionRepeatWrapper(_staggered_env(logs={2: earlier, 3: later}), repeat=4)

    infos = env.step(_action(env))[4]

    assert infos["log"] is later


def _assert_tree_equal(actual, expected) -> None:
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    else:
        assert actual is expected


@pytest.mark.parametrize("final_info_every_step", [True, False], ids=["every-step", "on-done-only"])
def test_repeat_one_matches_the_inner_step(final_info_every_step: bool) -> None:
    log = {"Metrics/success": torch.tensor(1.0)}

    def make() -> _FakeStaggeredEnv:
        return _FakeStaggeredEnv(
            term_limit=[2, None], trunc_limit=[None, 3], final_info_every_step=final_info_every_step, logs={3: log}
        )

    inner, wrapped = make(), ActionRepeatWrapper(make(), repeat=1)
    inner.reset()
    wrapped.reset()

    for _ in range(4):
        expected = inner.step(_action(inner))
        actual = wrapped.step(_action(wrapped))
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_tree_equal(actual_item, expected_item)


def test_changing_final_info_keys_between_sub_steps_raises() -> None:
    class _ChangingKeysEnv(_FakeStaggeredEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            if self.num_steps == 2:
                info["final_info"] = {"other": info["final_info"]["raw_value"]}
            return obs, reward, terminated, truncated, info

    env = ActionRepeatWrapper(_ChangingKeysEnv([None], [None]), repeat=4)

    with pytest.raises(ValueError, match="final_info"):
        env.step(_action(env))


def test_attributes_are_forwarded_to_the_inner_env() -> None:
    inner = _staggered_env()
    env = ActionRepeatWrapper(inner, repeat=4)

    assert env.num_envs == 4
    assert env.device == inner.device
    assert env.single_action_space is inner.single_action_space
    assert env.action_space is inner.action_space
    env.post_update_sync()
    assert inner.post_update_sync_calls == 1

    # Read on every access, not snapshotted: a frame stack above rewrites it.
    rewritten = spaces.Dict({"state": spaces.Box(-1.0, 1.0, (1,), np.float32)})
    inner.single_observation_space = rewritten
    assert env.single_observation_space is rewritten


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_step_does_not_sync_the_host_on_cuda() -> None:
    env = ActionRepeatWrapper(_staggered_env(device="cuda", logs={2: {"Metrics/success": 0.5}}), repeat=4)
    env.reset()
    action = _action(env)
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            _obs, reward, _terminated, _truncated, infos = env.step(action)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert reward.device.type == "cuda"
    assert infos["_final_info"].device.type == "cuda"
    # env1 (term_limit 3) ends twice only in the third chunk.
    assert env.dropped_episode_ends.tolist() == [9, 1, 0, 3]
