"""_IsaacLabVecEnvAdapter against a duck-typed fake env (no Isaac Lab needed)."""
from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_garden.envs.isaaclab.env import _box_from_tensor, _IsaacLabVecEnvAdapter
from rl_garden.envs.wrappers import ActionRepeatWrapper, ImageFrameStackWrapper
from rl_garden.observations.schema import ObservationContractError

NUM_ENVS = 2


class _FakeIsaacLabEnv:
    def __init__(self, obs: dict, declared_policy_space: gym.spaces.Space | None = None) -> None:
        self._obs = obs
        self.unwrapped = SimpleNamespace(
            num_envs=NUM_ENVS,
            device="cpu",
            action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(NUM_ENVS, 1), dtype=np.float32),
        )
        if declared_policy_space is not None:
            # DirectRLEnv publishes cfg.observation_space under the "policy" group.
            self.unwrapped.single_observation_space = gym.spaces.Dict({"policy": declared_policy_space})

    def reset(self, seed=None, options=None):
        return self._obs, {}

    def step(self, actions):
        zeros = torch.zeros(NUM_ENVS, dtype=torch.bool)
        return self._obs, torch.zeros(NUM_ENVS), zeros, zeros, {}


def _state() -> torch.Tensor:
    return torch.randn(NUM_ENVS, 4)


def test_nonzero_rgb_and_state_build_dict_space() -> None:
    rgb = torch.zeros(NUM_ENVS, 64, 64, 3, dtype=torch.uint8)
    rgb[0, 10, 10] = 200
    adapter = _IsaacLabVecEnvAdapter(
        _FakeIsaacLabEnv({"rgb_front": rgb, "state": _state()}), seed=0, is_visual=True
    )

    space = adapter.single_observation_space
    assert set(space.spaces) == {"rgb_front", "state"}
    assert space["rgb_front"].shape == (64, 64, 3) and space["rgb_front"].dtype == np.uint8
    assert space["state"].shape == (4,) and space["state"].dtype == np.float32
    assert adapter.single_action_space.shape == (1,)


def test_all_zero_rgb_raises() -> None:
    rgb = torch.zeros(NUM_ENVS, 64, 64, 3, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="zero frames"):
        _IsaacLabVecEnvAdapter(
            _FakeIsaacLabEnv({"rgb_front": rgb, "state": _state()}), seed=0, is_visual=True
        )


def test_state_only_obs_unaffected() -> None:
    adapter = _IsaacLabVecEnvAdapter(_FakeIsaacLabEnv({"policy": torch.zeros(NUM_ENVS, 4)}), seed=0)

    assert set(adapter.single_observation_space.spaces) == {"state"}


def _depth() -> torch.Tensor:
    return torch.rand(NUM_ENVS, 8, 8, 1)


def _declared_depth_and_state() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "depth_front": gym.spaces.Box(0.0, 1.0, (8, 8, 1), np.float32),
            "state": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
        }
    )


def _declared_depth_adapter() -> _IsaacLabVecEnvAdapter:
    env = _FakeIsaacLabEnv({"depth_front": _depth(), "state": _state()}, _declared_depth_and_state())
    return _IsaacLabVecEnvAdapter(env, seed=0, is_visual=True)


def test_declared_space_bounds_are_used_in_visual_mode() -> None:
    space = _declared_depth_adapter().single_observation_space
    declared = _declared_depth_and_state()

    assert set(space.spaces) == {"depth_front", "state"}
    assert space["depth_front"].shape == (8, 8, 1) and space["depth_front"].dtype == np.float32
    assert np.all(space["depth_front"].low == 0.0) and np.all(space["depth_front"].high == 1.0)
    assert space["state"] == declared["state"]


def test_update_obs_space_broadcasts_declared_bounds_over_frame_dim() -> None:
    adapter = _declared_depth_adapter()

    adapter.update_obs_space({"depth_front": torch.rand(NUM_ENVS, 3, 8, 8, 1), "state": _state()})

    depth_space = adapter.single_observation_space["depth_front"]
    assert depth_space.shape == (3, 8, 8, 1) and depth_space.dtype == np.float32
    assert np.all(depth_space.low == 0.0) and np.all(depth_space.high == 1.0)
    assert adapter.single_observation_space["state"].shape == (4,)


def test_update_obs_space_rejects_shape_matching_neither_declared_nor_stacked() -> None:
    adapter = _declared_depth_adapter()

    with pytest.raises(ObservationContractError, match="depth_front"):
        adapter.update_obs_space({"depth_front": torch.rand(NUM_ENVS, 8, 8, 2), "state": _state()})


def test_frame_stack_wrapper_keeps_declared_bounds() -> None:
    wrapped = ImageFrameStackWrapper(_declared_depth_adapter(), frame_stack=3)

    depth_space = wrapped.single_observation_space["depth_front"]
    assert depth_space.shape == (3, 8, 8, 1)
    assert np.all(depth_space.low == 0.0) and np.all(depth_space.high == 1.0)


def test_batched_action_space_has_leading_num_envs_dim_and_env_bounds() -> None:
    env = _FakeIsaacLabEnv({"state": _state()})
    low = np.tile(np.array([-2.0, 0.0], dtype=np.float32), (NUM_ENVS, 1))
    high = np.tile(np.array([2.0, 0.5], dtype=np.float32), (NUM_ENVS, 1))
    env.unwrapped.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    adapter = _IsaacLabVecEnvAdapter(env, seed=0)

    single = adapter.single_action_space
    assert single.shape == (2,)
    assert adapter.action_space.shape == (NUM_ENVS, *single.shape)
    assert adapter.action_space.dtype == single.dtype
    np.testing.assert_array_equal(adapter.action_space.low, low)
    np.testing.assert_array_equal(adapter.action_space.high, high)


def test_frame_stack_wrapper_resolves_batched_action_space() -> None:
    adapter = _declared_depth_adapter()

    wrapped = ImageFrameStackWrapper(adapter, frame_stack=3)

    assert wrapped.action_space == adapter.action_space
    assert wrapped.action_space.shape == (NUM_ENVS, 1)


@pytest.mark.parametrize(
    ("obs", "declared", "match"),
    [
        pytest.param(
            {"depth_front": _depth().half(), "state": _state()},
            _declared_depth_and_state(),
            "depth_front",
            id="wrong-dtype",
        ),
        pytest.param(
            {"depth_front": torch.rand(NUM_ENVS, 8, 8, 2), "state": _state()},
            _declared_depth_and_state(),
            "depth_front",
            id="wrong-shape",
        ),
        pytest.param(
            {"depth_front": _depth(), "state": _state()},
            gym.spaces.Dict({"depth_front": gym.spaces.Box(0.0, 1.0, (8, 8, 1), np.float32)}),
            "state",
            id="emitted-key-not-declared",
        ),
        pytest.param(
            {"depth_front": _depth(), "state": _state()},
            gym.spaces.Dict(
                {
                    **_declared_depth_and_state().spaces,
                    "rgb_side": gym.spaces.Box(0, 255, (8, 8, 3), np.uint8),
                }
            ),
            "rgb_side",
            id="declared-key-not-emitted",
        ),
        pytest.param(
            {"depth_front": _depth(), "state": _state()},
            gym.spaces.Dict(
                {
                    "depth_front": gym.spaces.Box(0.0, 1.0, (8, 8, 1), np.float32),
                    "state": gym.spaces.Discrete(4),
                }
            ),
            "state",
            id="declared-entry-not-box",
        ),
    ],
)
def test_declared_space_mismatch_raises(obs: dict, declared: gym.spaces.Dict, match: str) -> None:
    with pytest.raises(ObservationContractError, match=match):
        _IsaacLabVecEnvAdapter(_FakeIsaacLabEnv(obs, declared), seed=0, is_visual=True)


def test_dropped_state_is_still_validated_against_declaration() -> None:
    obs = {"depth_front": _depth(), "state": _state()}
    declared = _declared_depth_and_state()

    adapter = _IsaacLabVecEnvAdapter(_FakeIsaacLabEnv(obs, declared), seed=0, is_visual=True, state=False)
    assert set(adapter.single_observation_space.spaces) == {"depth_front"}

    without_state = gym.spaces.Dict({"depth_front": declared["depth_front"]})
    with pytest.raises(ObservationContractError, match="state"):
        _IsaacLabVecEnvAdapter(_FakeIsaacLabEnv(obs, without_state), seed=0, is_visual=True, state=False)


@pytest.mark.parametrize(
    "declared_policy_space",
    [
        pytest.param(None, id="no-declared-space"),
        pytest.param(gym.spaces.Box(-np.inf, np.inf, (4,), np.float32), id="box-under-policy"),
    ],
)
def test_without_declared_dict_spaces_are_inferred(declared_policy_space) -> None:
    obs = {"depth_front": _depth(), "state": _state()}

    adapter = _IsaacLabVecEnvAdapter(_FakeIsaacLabEnv(obs, declared_policy_space), seed=0, is_visual=True)

    expected = gym.spaces.Dict({key: _box_from_tensor(value) for key, value in obs.items()})
    assert adapter.single_observation_space == expected


def test_state_only_mode_uses_declared_state_bounds() -> None:
    declared = gym.spaces.Dict({"state": gym.spaces.Box(-1.0, 1.0, (4,), np.float32)})

    adapter = _IsaacLabVecEnvAdapter(_FakeIsaacLabEnv({"state": _state()}, declared), seed=0)

    state_space = adapter.single_observation_space["state"]
    assert np.all(state_space.low == -1.0) and np.all(state_space.high == 1.0)


class _ScriptedExtrasEnv(_FakeIsaacLabEnv):
    """Steps through scripted ``(reward, terminated, truncated)`` rows and
    returns one persistent ``extras`` dict, like Isaac Lab's envs."""

    def __init__(
        self, script: list[tuple[list[float], list[bool], list[bool]]], obs: dict | None = None
    ) -> None:
        super().__init__(obs if obs is not None else {"policy": torch.zeros(NUM_ENVS, 4)})
        self._script = iter(script)
        self.extras: dict = {}

    def reset(self, seed=None, options=None):
        return self._obs, self.extras

    def step(self, actions):
        reward, terminated, truncated = next(self._script)
        return (
            self._obs,
            torch.tensor(reward),
            torch.tensor(terminated),
            torch.tensor(truncated),
            self.extras,
        )


_NO_DONE = ([0.0, 0.0], [False, False], [False, False])


def _scripted_adapter(script) -> tuple[_IsaacLabVecEnvAdapter, _ScriptedExtrasEnv]:
    env = _ScriptedExtrasEnv(script)
    adapter = _IsaacLabVecEnvAdapter(env, seed=0)
    adapter.reset()
    return adapter, env


def _step(adapter: _IsaacLabVecEnvAdapter) -> dict:
    return adapter.step(torch.zeros(NUM_ENVS, 1))[4]


def test_final_info_carries_return_and_length_on_every_step_across_a_reset() -> None:
    adapter, _ = _scripted_adapter(
        [
            ([1.0, 2.0], [False, False], [False, False]),
            ([3.0, 4.0], [True, False], [False, False]),
            ([5.0, 6.0], [False, False], [False, True]),
        ]
    )

    infos = [_step(adapter) for _ in range(3)]

    for info in infos:
        assert set(info["final_info"]["episode"]) == {"return", "episode_len"}
    assert infos[0]["_final_info"].tolist() == [False, False]
    assert infos[1]["_final_info"].tolist() == [True, False]
    assert infos[1]["final_info"]["episode"]["return"][0].item() == 4.0
    assert infos[1]["final_info"]["episode"]["episode_len"][0].item() == 2
    assert infos[2]["_final_info"].tolist() == [False, True]
    assert infos[2]["final_info"]["episode"]["return"][1].item() == 12.0
    assert infos[2]["final_info"]["episode"]["episode_len"][1].item() == 3


def test_env_extras_dict_is_not_mutated() -> None:
    adapter, env = _scripted_adapter([_NO_DONE])
    env.extras["episode_metrics"] = {"success": torch.ones(NUM_ENVS, dtype=torch.bool)}

    _step(adapter)

    assert set(env.extras) == {"episode_metrics"}


def test_episode_metrics_are_merged_into_final_info_and_removed() -> None:
    adapter, env = _scripted_adapter([([1.0, 1.0], [True, False], [False, False])])
    success = torch.tensor([True, False])
    env.extras["episode_metrics"] = {"success": success}

    info = _step(adapter)

    assert "episode_metrics" not in info
    assert set(info["final_info"]["episode"]) == {"return", "episode_len", "success"}
    assert info["final_info"]["episode"]["success"] is success


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(torch.ones(NUM_ENVS, 1), id="extra-dim"),
        pytest.param(torch.ones(NUM_ENVS + 1), id="wrong-width"),
        pytest.param(1.0, id="not-a-tensor"),
    ],
)
def test_episode_metric_with_bad_shape_raises(value) -> None:
    adapter, env = _scripted_adapter([_NO_DONE])
    env.extras["episode_metrics"] = {"success": value}

    with pytest.raises(ValueError, match="'success'"):
        _step(adapter)


@pytest.mark.parametrize("key", ["return", "episode_len"])
def test_episode_metric_colliding_with_adapter_statistic_raises(key) -> None:
    adapter, env = _scripted_adapter([_NO_DONE])
    env.extras["episode_metrics"] = {key: torch.zeros(NUM_ENVS)}

    with pytest.raises(ValueError, match="collides"):
        _step(adapter)


def test_log_is_forwarded_only_when_a_new_dict_object_is_assigned() -> None:
    adapter, env = _scripted_adapter([_NO_DONE] * 4)
    first_log = {"Episode_Reward/alive": torch.tensor(1.0)}
    env.extras["log"] = first_log

    forwarded = [_step(adapter).get("log")]
    forwarded.append(_step(adapter).get("log"))
    second_log = {"Episode_Reward/alive": torch.tensor(2.0)}
    env.extras["log"] = second_log
    forwarded.append(_step(adapter).get("log"))
    forwarded.append(_step(adapter).get("log"))

    assert forwarded[0] is first_log and forwarded[1] is None
    assert forwarded[2] is second_log and forwarded[3] is None


def test_reset_time_log_is_not_forwarded() -> None:
    env = _ScriptedExtrasEnv([_NO_DONE])
    adapter = _IsaacLabVecEnvAdapter(env, seed=0)
    env.extras["log"] = {"Episode_Reward/alive": 0.0}
    adapter.reset()

    assert "log" not in _step(adapter)


def test_reset_restarts_episode_statistics() -> None:
    adapter, _ = _scripted_adapter(
        [([1.0, 1.0], [False, False], [False, False]), ([2.0, 2.0], [True, True], [False, False])]
    )
    _step(adapter)

    adapter.reset()
    episode = _step(adapter)["final_info"]["episode"]

    assert episode["return"].tolist() == [2.0, 2.0]
    assert episode["episode_len"].tolist() == [1, 1]


_RESET_ENV0 = ([1.0, 1.0], [True, False], [False, False])


def _visual_obs() -> dict:
    return {"rgb_front": torch.randint(1, 255, (NUM_ENVS, 8, 8, 3), dtype=torch.uint8), "state": _state()}


def test_final_obs_is_reported_as_state_final_observation_with_the_step_obs_on_non_done_rows() -> None:
    adapter, env = _scripted_adapter([_RESET_ENV0])
    terminal = torch.randn(NUM_ENVS, 4)
    env.extras["final_obs"] = {"policy": terminal}

    obs, _, _, _, info = adapter.step(torch.zeros(NUM_ENVS, 1))

    assert "final_obs" not in info and "_final_observation" not in info
    assert set(info["final_observation"]) == {"state"}
    final_state = info["final_observation"]["state"]
    assert torch.equal(final_state[0], terminal[0])
    assert torch.equal(final_state[1], obs["state"][1])
    assert set(env.extras) == {"final_obs"}


def test_stale_final_obs_is_replaced_by_the_step_obs_on_a_step_without_resets() -> None:
    adapter, env = _scripted_adapter([_NO_DONE])
    env.extras["final_obs"] = {"policy": torch.randn(NUM_ENVS, 4)}

    obs, _, _, _, info = adapter.step(torch.zeros(NUM_ENVS, 1))

    assert torch.equal(info["final_observation"]["state"], obs["state"])


def test_without_final_obs_no_final_observation_is_reported() -> None:
    adapter, _ = _scripted_adapter([_RESET_ENV0])

    info = _step(adapter)

    assert "final_observation" not in info and "_final_observation" not in info


def test_visual_final_obs_keeps_image_and_state_keys() -> None:
    env = _ScriptedExtrasEnv([_RESET_ENV0], obs=_visual_obs())
    adapter = _IsaacLabVecEnvAdapter(env, seed=0, is_visual=True)
    terminal = _visual_obs()
    env.extras["final_obs"] = terminal

    obs, _, _, _, info = adapter.step(torch.zeros(NUM_ENVS, 1))

    assert set(info["final_observation"]) == {"rgb_front", "state"}
    for key in ("rgb_front", "state"):
        assert torch.equal(info["final_observation"][key][0], terminal[key][0])
        assert torch.equal(info["final_observation"][key][1], obs[key][1])


def test_frame_stack_wrapper_stacks_final_observation_over_the_previous_history() -> None:
    obs = _visual_obs()
    env = _ScriptedExtrasEnv([_RESET_ENV0], obs=obs)
    adapter = _IsaacLabVecEnvAdapter(env, seed=0, is_visual=True)
    wrapped = ImageFrameStackWrapper(adapter, frame_stack=3, autoreset_in_step=True)
    wrapped.reset()
    terminal = _visual_obs()
    env.extras["final_obs"] = terminal

    info = wrapped.step(torch.zeros(NUM_ENVS, 1))[4]

    stacked = info["final_observation"]["rgb_front"]
    assert stacked.shape == (NUM_ENVS, 3, 8, 8, 3)
    assert torch.equal(stacked[0, -1], terminal["rgb_front"][0])
    assert torch.equal(stacked[1, -1], obs["rgb_front"][1])
    assert torch.equal(stacked[:, 0], obs["rgb_front"]) and torch.equal(stacked[:, 1], obs["rgb_front"])
    assert torch.equal(info["final_observation"]["state"][0], terminal["state"][0])
    assert torch.equal(info["final_observation"]["state"][1], obs["state"][1])


def _repeat_step(env) -> tuple:
    return env.step(torch.zeros(NUM_ENVS, 1))


def test_action_repeat_freezes_episode_statistics_at_the_ending_sub_step_in_env_steps() -> None:
    adapter, _ = _scripted_adapter(
        [([1.0, 1.0], [False, False], [False, False])] * 4
        + [
            ([1.0, 1.0], [False, False], [False, False]),
            ([2.0, 1.0], [True, False], [False, False]),
            ([5.0, 1.0], [False, False], [False, False]),
            ([5.0, 1.0], [False, False], [False, False]),
        ]
        + [
            ([1.0, 1.0], [False, False], [True, False]),
            ([1.0, 1.0], [False, False], [False, False]),
            ([1.0, 1.0], [False, False], [False, False]),
            ([1.0, 1.0], [False, False], [False, True]),
        ]
    )
    env = ActionRepeatWrapper(adapter, repeat=4)

    _repeat_step(env)
    _obs, reward, terminated, _truncated, info = _repeat_step(env)

    assert reward.tolist() == [3.0, 4.0]
    assert terminated.tolist() == [True, False]
    assert info["_final_info"].tolist() == [True, False]
    assert info["final_info"]["episode"]["return"][0].item() == 7.0
    assert info["final_info"]["episode"]["episode_len"][0].item() == 6

    # env0's new episode includes the two uncounted sub-steps of the previous decision.
    _obs, reward, terminated, truncated, info = _repeat_step(env)

    assert reward.tolist() == [1.0, 4.0]
    assert terminated.tolist() == [False, False] and truncated.tolist() == [True, True]
    assert info["final_info"]["episode"]["return"].tolist() == [11.0, 12.0]
    assert info["final_info"]["episode"]["episode_len"].tolist() == [3, 12]
    assert env.dropped_episode_ends.tolist() == [0, 0]


class _LogAtStepEnv(_ScriptedExtrasEnv):
    """Assigns a new ``extras["log"]`` dict on the scripted (1-based) steps."""

    def __init__(self, script, logs: dict[int, dict]) -> None:
        super().__init__(script)
        self._logs = logs
        self._num_steps = 0

    def step(self, actions):
        self._num_steps += 1
        if self._num_steps in self._logs:
            self.extras["log"] = self._logs[self._num_steps]
        return super().step(actions)


def test_action_repeat_forwards_a_log_assigned_on_a_middle_sub_step() -> None:
    log = {"Episode_Reward/alive": torch.tensor(1.0)}
    adapter = _IsaacLabVecEnvAdapter(_LogAtStepEnv([_NO_DONE] * 8, logs={2: log}), seed=0)
    adapter.reset()
    env = ActionRepeatWrapper(adapter, repeat=4)

    first = _repeat_step(env)[4]
    second = _repeat_step(env)[4]

    assert first["log"] is log
    assert "log" not in second


class _CountingVisualEnv(_ScriptedExtrasEnv):
    """Visual env whose image pixels equal the number of steps taken plus one."""

    def __init__(self, script) -> None:
        super().__init__(script, obs=self._frame(1))
        self._value = 1

    @staticmethod
    def _frame(value: int) -> dict:
        return {"rgb_front": torch.full((NUM_ENVS, 8, 8, 3), value, dtype=torch.uint8), "state": _state()}

    def step(self, actions):
        self._value += 1
        self._obs = self._frame(self._value)
        return super().step(actions)


def test_frame_stack_over_action_repeat_stacks_decision_level_frames() -> None:
    adapter = _IsaacLabVecEnvAdapter(_CountingVisualEnv([_NO_DONE] * 8), seed=0, is_visual=True)
    wrapped = ImageFrameStackWrapper(
        ActionRepeatWrapper(adapter, repeat=4), frame_stack=3, autoreset_in_step=True
    )
    wrapped.reset()

    wrapped.step(torch.zeros(NUM_ENVS, 1))
    obs = wrapped.step(torch.zeros(NUM_ENVS, 1))[0]

    assert obs["rgb_front"].shape == (NUM_ENVS, 3, 8, 8, 3)
    assert obs["rgb_front"][:, :, 0, 0, 0].tolist() == [[1, 5, 9]] * NUM_ENVS


def _make_env_with_fake_build(monkeypatch, raw_env, **cfg_kwargs):
    from rl_garden.envs.isaaclab import env as isaaclab_env
    from rl_garden.envs.isaaclab.config import IsaacLabEnvConfig
    from rl_garden.envs.isaaclab.v3 import runtime as v3_runtime

    monkeypatch.setattr(isaaclab_env, "generation", lambda: "v3")
    monkeypatch.setattr(v3_runtime, "build", lambda cfg: raw_env)
    cfg = IsaacLabEnvConfig(env_id="RlGarden-Fake-v0", num_envs=NUM_ENVS, seed=0, **cfg_kwargs)
    return isaaclab_env.make_isaaclab_env(cfg)


def test_make_env_without_action_repeat_returns_the_adapter(monkeypatch) -> None:
    env = _make_env_with_fake_build(monkeypatch, _FakeIsaacLabEnv({"state": _state()}))

    assert type(env) is _IsaacLabVecEnvAdapter


def test_make_env_with_action_repeat_wraps_the_adapter(monkeypatch) -> None:
    env = _make_env_with_fake_build(monkeypatch, _FakeIsaacLabEnv({"state": _state()}), action_repeat=4)

    assert isinstance(env, ActionRepeatWrapper) and env.repeat == 4
    assert type(env.env) is _IsaacLabVecEnvAdapter
    assert env.num_envs == NUM_ENVS and env.single_action_space == env.env.single_action_space


def test_make_env_rejects_action_repeat_below_one(monkeypatch) -> None:
    with pytest.raises(ValueError, match="repeat must be >= 1"):
        _make_env_with_fake_build(monkeypatch, _FakeIsaacLabEnv({"state": _state()}), action_repeat=0)


@pytest.mark.parametrize("action_repeat", [1, 4])
def test_make_env_frame_stack_wraps_action_repeat(monkeypatch, action_repeat: int) -> None:
    env = _make_env_with_fake_build(
        monkeypatch,
        _FakeIsaacLabEnv(_visual_obs()),
        is_visual=True,
        frame_stack=3,
        action_repeat=action_repeat,
    )

    assert isinstance(env, ImageFrameStackWrapper)
    inner = env.env
    adapter = inner.env if action_repeat > 1 else inner
    assert isinstance(inner, ActionRepeatWrapper) is (action_repeat > 1)
    assert type(adapter) is _IsaacLabVecEnvAdapter and env.base_env is adapter
    assert env.single_observation_space["rgb_front"].shape == (3, 8, 8, 3)
    assert env.single_observation_space is adapter.single_observation_space
    assert env.num_envs == NUM_ENVS and env.post_update_sync == adapter.post_update_sync
