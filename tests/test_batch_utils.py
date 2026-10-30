from __future__ import annotations

import pytest
import torch

from rl_garden.envs.wrappers._batch_utils import ChunkAccumulator


class _StaggeredSubSteps:
    """Scripted SAME_STEP-autoreset sub-steps, no env and no host syncs: each
    env's value counts up by 1 per sub-step and resets to 0 the instant it
    hits its termination/truncation limit (``None`` = never). Reward is 1 per
    sub-step.

    By default ``final_info``/``final_observation`` are reported on every
    sub-step (valid on done rows only); ``final_info_every_step=False``
    reports them only on sub-steps with a done env. ``reuse_buffers=True``
    writes them into one persistent tensor in place. ``logs`` maps a sub-step
    count (1-based) to the ``infos["log"]`` dict reported on it."""

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

    def step(self):
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
                info["_final_observation"] = done
        if self.num_steps in self._logs:
            info["log"] = self._logs[self.num_steps]
        reward = torch.ones(self.num_envs, device=self.device)
        return reward, terminated, truncated, info


def _staggered(**kwargs) -> _StaggeredSubSteps:
    # env0 terminates on sub-step 0 and again on 1, 2, 3; env1 terminates on
    # sub-step 2; env2 never ends; env3 truncates on sub-step 1 and again on 3.
    return _StaggeredSubSteps(term_limit=[1, 3, None, None], trunc_limit=[None, None, None, 2], **kwargs)


def _run_chunk(chunk: ChunkAccumulator, source: _StaggeredSubSteps, num_sub_steps: int):
    chunk.begin()
    for _ in range(num_sub_steps):
        reward, terminated, truncated, infos = source.step()
        chunk.add(reward, terminated, truncated, infos)
    return chunk.finish(infos)


@pytest.mark.parametrize("reuse_buffers", [False, True], ids=["fresh-tensors", "reused-buffers"])
def test_staggered_episode_ends_freeze_at_the_ending_sub_step(reuse_buffers: bool) -> None:
    reward, terminated, truncated, infos = _run_chunk(
        ChunkAccumulator(), _staggered(reuse_buffers=reuse_buffers), num_sub_steps=4
    )

    assert torch.equal(reward, torch.tensor([1.0, 3.0, 4.0, 2.0]))
    assert torch.equal(terminated, torch.tensor([True, True, False, False]))
    assert torch.equal(truncated, torch.tensor([False, False, False, True]))
    ended = torch.tensor([True, True, False, True])
    assert torch.equal(infos["_final_observation"], ended)
    assert torch.equal(infos["_final_info"], ended)
    assert torch.equal(infos["final_observation"]["state"].squeeze(-1)[ended], torch.tensor([1.0, 3.0, 2.0]))
    assert torch.equal(infos["final_info"]["raw_value"][ended], torch.tensor([1.0, 3.0, 2.0]))
    # With reused buffers this only holds because the first report is cloned.
    assert torch.equal(infos["final_info"]["step"][ended], torch.tensor([1.0, 3.0, 2.0]))
    # Non-final keys come from the last sub-step.
    assert torch.equal(infos["step_count"], torch.full((4,), 4.0))
    assert "log" not in infos


def test_second_episode_end_in_a_chunk_is_counted_and_not_cleared_by_begin() -> None:
    chunk, source = ChunkAccumulator(), _staggered()
    assert chunk.dropped_episode_ends is None

    _run_chunk(chunk, source, num_sub_steps=4)
    assert chunk.dropped_episode_ends.dtype == torch.int64
    assert chunk.dropped_episode_ends.tolist() == [3, 0, 0, 1]

    _run_chunk(chunk, source, num_sub_steps=4)
    assert chunk.dropped_episode_ends.tolist() == [6, 0, 0, 2]


def test_no_final_observation_when_the_inner_env_reports_none() -> None:
    infos = _run_chunk(ChunkAccumulator(), _staggered(final_observation=False), num_sub_steps=4)[3]

    assert "final_observation" not in infos and "_final_observation" not in infos
    assert torch.equal(infos["_final_info"], torch.tensor([True, True, False, True]))


def test_final_info_reported_only_on_done_sub_steps_is_forwarded_and_frozen() -> None:
    infos = _run_chunk(ChunkAccumulator(), _staggered(final_info_every_step=False), num_sub_steps=4)[3]

    ended = torch.tensor([True, True, False, True])
    assert torch.equal(infos["_final_info"], ended)
    assert torch.equal(infos["final_info"]["raw_value"][ended], torch.tensor([1.0, 3.0, 2.0]))
    assert torch.equal(infos["final_info"]["step"][ended], torch.tensor([1.0, 3.0, 2.0]))


def test_no_final_info_when_no_sub_step_reports_one() -> None:
    source = _StaggeredSubSteps(term_limit=[None, None], trunc_limit=[None, None], final_info_every_step=False)

    infos = _run_chunk(ChunkAccumulator(), source, num_sub_steps=4)[3]

    assert not any(key.startswith(("final_", "_final_")) for key in infos)


def test_log_from_a_middle_sub_step_is_forwarded() -> None:
    log = {"Metrics/success": torch.tensor(0.5)}
    chunk, source = ChunkAccumulator(), _staggered(logs={2: log})

    first = _run_chunk(chunk, source, num_sub_steps=4)[3]
    second = _run_chunk(chunk, source, num_sub_steps=4)[3]

    assert first["log"] is log
    assert "log" not in second


def test_latest_log_dict_in_a_chunk_is_forwarded_whole() -> None:
    earlier = {"Metrics/success": torch.tensor(0.5), "Metrics/only_earlier": torch.tensor(1.0)}
    later = {"Metrics/success": torch.tensor(0.75)}

    infos = _run_chunk(ChunkAccumulator(), _staggered(logs={2: earlier, 3: later}), num_sub_steps=4)[3]

    assert infos["log"] is later


@pytest.mark.parametrize("name", ["final_info", "final_observation"])
def test_changing_final_keys_between_sub_steps_raises(name: str) -> None:
    class _ChangingKeys(_StaggeredSubSteps):
        def step(self):
            reward, terminated, truncated, info = super().step()
            if self.num_steps == 2:
                info[name] = {"other": terminated.float()}
            return reward, terminated, truncated, info

    with pytest.raises(ValueError, match=name):
        _run_chunk(ChunkAccumulator(), _ChangingKeys([None], [None]), num_sub_steps=4)


def test_finish_without_add_raises() -> None:
    chunk = ChunkAccumulator()
    chunk.begin()

    with pytest.raises(RuntimeError, match="before any add"):
        chunk.finish({})


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
def test_one_sub_step_matches_the_inner_step(final_info_every_step: bool) -> None:
    log = {"Metrics/success": torch.tensor(1.0)}

    def make() -> _StaggeredSubSteps:
        return _StaggeredSubSteps(
            term_limit=[2, None], trunc_limit=[None, 3], final_info_every_step=final_info_every_step, logs={3: log}
        )

    inner, accumulated = make(), make()
    chunk = ChunkAccumulator()

    for _ in range(4):
        expected = inner.step()
        actual = _run_chunk(chunk, accumulated, num_sub_steps=1)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_tree_equal(actual_item, expected_item)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_begin_add_finish_do_not_sync_the_host_on_cuda() -> None:
    chunk, source = ChunkAccumulator(), _staggered(device="cuda", logs={2: {"Metrics/success": 0.5}})
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            reward, _terminated, _truncated, infos = _run_chunk(chunk, source, num_sub_steps=4)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert reward.device.type == "cuda"
    assert infos["_final_info"].device.type == "cuda"
    assert chunk.dropped_episode_ends.device.type == "cuda"
    # env1 (term_limit 3) ends twice only in the third chunk.
    assert chunk.dropped_episode_ends.tolist() == [9, 1, 0, 3]
