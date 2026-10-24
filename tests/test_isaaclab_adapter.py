"""_IsaacLabVecEnvAdapter against a duck-typed fake env (no Isaac Lab needed)."""
from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_garden.envs.isaaclab.env import _box_from_tensor, _IsaacLabVecEnvAdapter
from rl_garden.envs.wrappers import ImageFrameStackWrapper
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

    def __init__(self, script: list[tuple[list[float], list[bool], list[bool]]]) -> None:
        super().__init__({"policy": torch.zeros(NUM_ENVS, 4)})
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
