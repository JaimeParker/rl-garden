"""Cross-backend checks of the episode boundary contract
(``.agents/rules/adding-env-backend.md``, "Episode boundary contract").

Every producer is driven through at least one episode end per env;
``assert_episode_boundary`` checks rules 1-5 and 8 on every step, and
``test_episode_return_sums_raw_rewards`` compares ``episode.return`` with the
raw rewards (the step reward with ``reward_scale`` undone). A violation raises
its own ``AssertionError`` subclass, so a failure names the rule it breaks.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from test_isaaclab_adapter import NUM_ENVS, _ScriptedExtrasEnv
from test_robotwin_env import FakeExecutor

from rl_garden.common.eval_metrics import canonical_episode_metric_key
from rl_garden.envs.isaaclab.env import _IsaacLabVecEnvAdapter
from rl_garden.envs.robotwin import RoboTwinEnv, RoboTwinEnvConfig
from rl_garden.envs.vector_env import TorchVectorEnvAdapter
from rl_garden.envs.wrappers import ActionRepeatWrapper
from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper


class NonDoneFinalObservationMismatch(AssertionError):
    """Rule 4: a row that did not end has a ``final_observation`` other than ``obs``."""


class PrivateFinalObservationKey(AssertionError):
    """Rule 5: an rl-garden producer emits ``infos["_final_observation"]``."""


class ScaledEpisodeReturn(AssertionError):
    """Rule 2: ``episode.return`` is not the sum of raw env rewards."""


# --- shared checks ----------------------------------------------------------


def _env_device(env: Any) -> torch.device:
    # gymnasium's VectorWrapper does not forward attributes; walk down to the
    # env that owns ``device``.
    while not hasattr(env, "device"):
        env = env.env
    device = torch.device(env.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _tree_leaves(tree: Any, path: str) -> Iterator[tuple[str, Any]]:
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _tree_leaves(value, f"{path}[{key!r}]")
    else:
        yield path, tree


def _assert_on_device(name: str, value: Any, device: torch.device) -> None:
    assert isinstance(value, torch.Tensor), f"{name} is a {type(value).__name__}, not a tensor"
    assert value.device == device, f"{name} is on {value.device}, env device is {device}"


def _assert_batch_vector(
    name: str, value: Any, num_envs: int, device: torch.device, dtype: torch.dtype | None = None
) -> None:
    _assert_on_device(name, value, device)
    assert tuple(value.shape) == (num_envs,), f"{name} has shape {tuple(value.shape)}, not ({num_envs},)"
    if dtype is not None:
        assert value.dtype == dtype, f"{name} has dtype {value.dtype}, not {dtype}"


def _canonical_episode(episode: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    canonical = {}
    for raw_key, value in episode.items():
        key = canonical_episode_metric_key(raw_key)
        if key is not None:
            canonical[key] = value
    return canonical


def _assert_final_info(infos: dict, done: torch.Tensor, num_envs: int, device: torch.device) -> None:
    """Rules 2, 3 and 8 for ``final_info``/``_final_info``."""
    if "final_info" not in infos:
        assert not bool(done.any()), f"envs {done.nonzero().flatten().tolist()} ended without final_info"
        return
    assert "_final_info" in infos, "final_info is emitted without _final_info"
    mask = infos["_final_info"]
    _assert_batch_vector("_final_info", mask, num_envs, device, torch.bool)
    assert torch.equal(mask, done), f"_final_info {mask.tolist()} != terminated | truncated {done.tolist()}"

    final_info = infos["final_info"]
    assert isinstance(final_info, dict), f"final_info is a {type(final_info).__name__}, not a dict"
    assert "episode" in final_info, f"final_info has no 'episode', keys {sorted(final_info)}"
    episode = final_info["episode"]
    assert isinstance(episode, dict), f"final_info['episode'] is a {type(episode).__name__}, not a dict"
    for key, value in episode.items():
        _assert_batch_vector(f"final_info['episode'][{key!r}]", value, num_envs, device)
    missing = {"return", "episode_len"} - set(_canonical_episode(episode))
    assert not missing, f"final_info['episode'] lacks {sorted(missing)}, keys {sorted(episode)}"
    # Only tensor leaves are checked outside "episode": RoboTwin passes its
    # per-env instruction strings through final_info.
    for name, value in _tree_leaves(final_info, "final_info"):
        if isinstance(value, torch.Tensor):
            _assert_on_device(name, value, device)


def _assert_final_observation(obs: Any, infos: dict, done: torch.Tensor, device: torch.device) -> None:
    """Rules 4 and 8 for ``final_observation``."""
    if "final_observation" not in infos:
        return
    final_obs = infos["final_observation"]
    if isinstance(obs, dict):
        assert isinstance(final_obs, dict), f"final_observation is a {type(final_obs).__name__}, obs a dict"
        assert set(final_obs) == set(obs), f"final_observation keys {sorted(final_obs)} != obs keys {sorted(obs)}"
        pairs = [(f"[{key!r}]", final_obs[key], obs[key]) for key in obs]
    else:
        pairs = [("", final_obs, obs)]
    for key, final_value, value in pairs:
        name = f"final_observation{key}"
        _assert_on_device(name, final_value, device)
        assert final_value.shape == value.shape, f"{name} has shape {tuple(final_value.shape)}, obs {tuple(value.shape)}"
        assert final_value.dtype == value.dtype, f"{name} has dtype {final_value.dtype}, obs {value.dtype}"
    kept = ~done
    for key, final_value, value in pairs:
        if not torch.equal(final_value[kept], value[kept]):
            raise NonDoneFinalObservationMismatch(
                f"final_observation{key} differs from obs on rows that did not end "
                f"({kept.nonzero().flatten().tolist()})"
            )


def _is_maniskill_vector_env(env: Any) -> bool:
    # make_maniskill_env returns a ManiSkillVectorEnv subclass when it scales rewards.
    return any(cls.__module__.split(".")[0] == "mani_skill" for cls in type(env).__mro__)


def assert_episode_boundary(env, obs, reward, terminated, truncated, infos) -> None:
    """Rules 1-5 and 8 of the episode boundary contract for one ``step()`` result."""
    device = _env_device(env)
    num_envs = env.num_envs
    assert env.metadata.get("autoreset_mode") == AutoresetMode.SAME_STEP, (
        f"metadata['autoreset_mode'] is {env.metadata.get('autoreset_mode')!r}"
    )
    for name, value in _tree_leaves(obs, "obs"):
        _assert_on_device(name, value, device)
    _assert_batch_vector("reward", reward, num_envs, device)
    _assert_batch_vector("terminated", terminated, num_envs, device, torch.bool)
    _assert_batch_vector("truncated", truncated, num_envs, device, torch.bool)
    done = terminated | truncated
    _assert_final_info(infos, done, num_envs, device)
    _assert_final_observation(obs, infos, done, device)
    # ManiSkill's own ManiSkillVectorEnv emits the key; it passes through.
    if "_final_observation" in infos and not _is_maniskill_vector_env(env):
        raise PrivateFinalObservationKey("infos carries '_final_observation'")


def _max_action(env: Any, device: torch.device) -> torch.Tensor:
    """The batched action space's upper bound, 0 where it is unbounded."""
    high = torch.as_tensor(np.asarray(env.action_space.high), dtype=torch.float32)
    return torch.nan_to_num(high, posinf=0.0, neginf=0.0).to(device)


def run_until_episode_end(env, max_steps: int) -> Iterator[tuple]:
    """Reset ``env`` and step it with a fixed action, yielding every
    ``(obs, reward, terminated, truncated, infos)``, until each env has ended
    at least one episode; fails if that takes more than ``max_steps`` steps."""
    device = _env_device(env)
    env.reset(seed=0)
    action = _max_action(env, device)
    ended = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    for _ in range(max_steps):
        step = env.step(action)
        yield step
        ended |= step[2] | step[3]
        if bool(ended.all()):
            return
    pytest.fail(f"envs {(~ended).nonzero().flatten().tolist()} did not end within {max_steps} steps")


# --- producers --------------------------------------------------------------


class _CountingEnv(gym.Env):
    """Single env whose observation is its step count since reset and whose
    reward is half that count; terminates and/or truncates at fixed counts."""

    def __init__(self, terminate_at: int | None, truncate_at: int | None, *, dict_obs: bool) -> None:
        box = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Dict({"state": box, "aux": box}) if dict_obs else box
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._terminate_at = terminate_at
        self._truncate_at = truncate_at
        self._dict_obs = dict_obs
        self._t = 0

    def _obs(self):
        value = np.array([float(self._t)])
        return {"state": value, "aux": value * 10} if self._dict_obs else value

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t == self._terminate_at
        truncated = self._t == self._truncate_at
        return self._obs(), 0.5 * self._t, terminated, truncated, {}


def _gymnasium_adapter(*, dict_obs: bool = False, reward_scale: float = 1.0):
    def make(terminate_at, truncate_at):
        return lambda: RecordEpisodeStatistics(_CountingEnv(terminate_at, truncate_at, dict_obs=dict_obs))

    vec = SyncVectorEnv([make(2, None), make(None, 3)], autoreset_mode=AutoresetMode.SAME_STEP)
    env = TorchVectorEnvAdapter(vec, device="cpu")
    if reward_scale != 1.0:
        env = RewardScaleBiasVectorWrapper(env, scale=reward_scale)
    return env


class _CountingIsaacLabEnv(_ScriptedExtrasEnv):
    """Scripted Isaac Lab fake whose observation counts each env's steps since
    reset. On a step where some env ends, ``extras["final_obs"]`` holds the
    full-batch pre-reset observation, and it stays in ``extras`` afterwards,
    as with Isaac Lab 3.0's ``compute_final_obs``."""

    def step(self, actions):
        reward, terminated, truncated = (torch.tensor(row) for row in next(self._script))
        done = terminated | truncated
        count = self._obs["policy"] + 1
        if bool(done.any()):
            self.extras["final_obs"] = {"policy": count}
        self._obs = {"policy": torch.where(done[:, None], torch.zeros_like(count), count)}
        return self._obs, reward, terminated, truncated, self.extras

    def close(self) -> None:
        pass


_F, _T = [False, False], [True, False]


def _isaaclab_adapter():
    # env0 terminates on step 2, env1 truncates on step 4; step 3 has no end.
    script = [
        ([1.0, 2.0], _F, _F),
        ([3.0, 4.0], _T, _F),
        ([5.0, 6.0], _F, _F),
        ([7.0, 8.0], _F, [False, True]),
    ]
    return _IsaacLabVecEnvAdapter(_CountingIsaacLabEnv(script), seed=0)


def _action_repeat_over_isaaclab():
    # ActionRepeatWrapper is a gym.Wrapper, which only wraps a gym.Env such as
    # the Isaac Lab adapter. env0 terminates on sub-step 2 of decision 1,
    # env1 truncates on sub-step 2 of decision 2.
    script = [
        ([1.0, 1.0], _F, _F),
        ([2.0, 1.0], _T, _F),
        ([3.0, 1.0], _F, _F),
        ([1.0, 1.0], _F, _F),
        ([1.0, 2.0], _F, [False, True]),
        ([1.0, 1.0], _F, _F),
    ]
    return ActionRepeatWrapper(_IsaacLabVecEnvAdapter(_CountingIsaacLabEnv(script), seed=0), 3)


def _robotwin(reward_scale: float):
    # FakeExecutor ends both envs on step 2 and returns raw task rewards.
    cfg = RoboTwinEnvConfig(
        num_envs=NUM_ENVS, device="cpu", image_size=(8, 8), max_episode_steps=10, reward_scale=reward_scale
    )
    return RoboTwinEnv(cfg, executor=FakeExecutor(num_envs=NUM_ENVS))


def _mujoco_warp_pendulum():
    pytest.importorskip("mujoco_warp")
    from rl_garden.envs.mujoco_warp.tasks.inverted_pendulum_warp import (
        InvertedPendulumWarpEnv,
    )

    return InvertedPendulumWarpEnv(nworld=NUM_ENVS, device="cuda:0")


def _maniskill_pick_cube(reward_scale: float):
    pytest.importorskip("mani_skill")
    from rl_garden.envs.maniskill import make_maniskill_env
    from rl_garden.envs.maniskill.config import ManiSkillEnvConfig

    cfg = ManiSkillEnvConfig(
        env_id="PickCube-v1",
        num_envs=NUM_ENVS,
        reward_scale=reward_scale,
        ignore_terminations=True,
        record_metrics=True,
    )
    return make_maniskill_env(cfg)


@dataclass(frozen=True)
class _Producer:
    make: Callable[[], Any]
    max_steps: int
    reward_scale: float = 1.0


_FLAT = _Producer(_gymnasium_adapter, max_steps=3)
_DICT = _Producer(lambda: _gymnasium_adapter(dict_obs=True), max_steps=3)
_SCALED = _Producer(lambda: _gymnasium_adapter(reward_scale=2.0), max_steps=3, reward_scale=2.0)
_ISAACLAB = _Producer(_isaaclab_adapter, max_steps=4)
_ACTION_REPEAT = _Producer(_action_repeat_over_isaaclab, max_steps=2)
_ROBOTWIN = _Producer(lambda: _robotwin(reward_scale=2.0), max_steps=4, reward_scale=2.0)
_MUJOCO_WARP = _Producer(_mujoco_warp_pendulum, max_steps=200)
# PickCube-v1 registers max_episode_steps=50; every env truncates there.
_MANISKILL = _Producer(lambda: _maniskill_pick_cube(reward_scale=2.0), max_steps=100, reward_scale=2.0)

_GPU = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@contextmanager
def _opened(producer: _Producer):
    env = producer.make()
    try:
        yield env
    finally:
        env.close()


@pytest.mark.parametrize(
    "producer",
    [
        pytest.param(_FLAT, id="torch-adapter-box"),
        pytest.param(_DICT, id="torch-adapter-dict"),
        pytest.param(_SCALED, id="torch-adapter-reward-scale"),
        pytest.param(_ISAACLAB, id="isaaclab-adapter"),
        pytest.param(_ROBOTWIN, id="robotwin"),
        pytest.param(_ACTION_REPEAT, id="action-repeat-over-isaaclab"),
        pytest.param(_MUJOCO_WARP, id="mujoco-warp-pendulum", marks=_GPU),
        pytest.param(_MANISKILL, id="maniskill-pick-cube", marks=_GPU),
    ],
)
def test_every_step_satisfies_the_episode_boundary_contract(producer: _Producer) -> None:
    with _opened(producer) as env:
        for step in run_until_episode_end(env, producer.max_steps):
            assert_episode_boundary(env, *step)


# ActionRepeatWrapper is left out: a decision's reward drops the sub-steps
# after an env's episode end, which its episode statistics still count.
@pytest.mark.parametrize(
    "producer",
    [
        pytest.param(_FLAT, id="torch-adapter-box"),
        pytest.param(_DICT, id="torch-adapter-dict"),
        pytest.param(_SCALED, id="torch-adapter-reward-scale"),
        pytest.param(_ISAACLAB, id="isaaclab-adapter"),
        pytest.param(_ROBOTWIN, id="robotwin"),
        pytest.param(_MUJOCO_WARP, id="mujoco-warp-pendulum", marks=_GPU),
        pytest.param(_MANISKILL, id="maniskill-pick-cube", marks=_GPU),
    ],
)
def test_episode_return_sums_raw_rewards(producer: _Producer) -> None:
    with _opened(producer) as env:
        raw_return = None
        for _obs, reward, terminated, truncated, infos in run_until_episode_end(env, producer.max_steps):
            raw_reward = reward.float() / producer.reward_scale
            raw_return = raw_reward if raw_return is None else raw_return + raw_reward
            done = terminated | truncated
            if not bool(done.any()):
                continue
            episode_return = _canonical_episode(infos["final_info"]["episode"])["return"].float()
            if not torch.allclose(episode_return[done], raw_return[done], rtol=1e-5, atol=1e-5):
                raise ScaledEpisodeReturn(
                    f"episode.return {episode_return[done].tolist()} != raw reward sum "
                    f"{raw_return[done].tolist()} (reward_scale={producer.reward_scale})"
                )
            raw_return = torch.where(done, torch.zeros_like(raw_return), raw_return)
