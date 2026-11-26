"""Every ``BatchedEnvWrapper`` stacks on every kind of batched env, in any order."""
from __future__ import annotations

import pickle

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms.offline import OfflineEnvSpec
from rl_garden.envs.gymnasium import GymnasiumEnvConfig, make_gymnasium_env
from rl_garden.envs.wrappers import (
    ActionChunkWrapper,
    ActionRepeatWrapper,
    BatchedEnvWrapper,
    ImageFrameStackWrapper,
    RewardScaleBiasWrapper,
    SkillActionWrapper,
    VectorizedDictStateWrapper,
)
from rl_garden.envs.wrappers.fwbw_reset_free import FWBWResetFreeWrapper
from rl_garden.envs.wrappers.gail_reward import GAILRewardWrapper
from rl_garden.envs.wrappers.multi_stage_classifier import (
    MultiStageBinaryRewardClassifierWrapper,
)
from rl_garden.envs.wrappers.multitask_pad import MultitaskPadWrapper
from rl_garden.envs.wrappers.reward_classifier import RewardClassifierWrapper

STATE_DIM_EXTRA = 3
ACTION_DIM_EXTRA = 2


class FakeBatchedEnv(gym.Env):
    """Batched torch ``gym.Env`` (the Isaac Lab / real-robot shape): image and
    state observations, reward 1 per step, every row truncates after 3 steps."""

    def __init__(self, num_envs: int = 2, *, autoreset: bool = True, box_obs: bool = False) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.autoreset = autoreset
        self.box_obs = box_obs
        self.metadata = {"autoreset_mode": AutoresetMode.SAME_STEP} if autoreset else {}
        state = spaces.Box(-np.inf, np.inf, (2,), np.float32)
        image = spaces.Box(0, 255, (4, 4, 3), np.uint8)
        self.single_observation_space = (
            state if box_obs else spaces.Dict({"rgb_front": image, "state": state})
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.sync_calls = 0
        self.closed = False
        self._t = 0

    def post_update_sync(self) -> None:
        self.sync_calls += 1

    def _obs(self):
        state = torch.full((self.num_envs, 2), float(self._t))
        if self.box_obs:
            return state
        image = torch.full((self.num_envs, 4, 4, 3), self._t, dtype=torch.uint8)
        return {"rgb_front": image, "state": state}

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        ended = self._t == 3
        truncated = torch.full((self.num_envs,), ended)
        obs, infos = self._obs(), {}
        if ended and self.autoreset:
            infos = {
                "final_observation": obs,
                "final_info": {"episode": {"return": torch.full((self.num_envs,), 3.0)}},
                "_final_info": truncated,
            }
            self._t = 0
            obs = self._obs()
        reward = torch.ones(self.num_envs)
        return obs, reward, torch.zeros_like(truncated), truncated, infos

    def close(self) -> None:
        self.closed = True


def _adapter_env() -> VectorEnv:
    return make_gymnasium_env(
        GymnasiumEnvConfig(
            env_id="RlgTestPixels-v0", num_envs=2, seed=0, import_modules=("_gymnasium_user_envs",)
        )
    )


def _maniskill_env() -> VectorEnv:
    pytest.importorskip("mani_skill")
    from rl_garden.envs.maniskill import ManiSkillEnvConfig, make_maniskill_env

    return make_maniskill_env(
        ManiSkillEnvConfig(
            env_id="PickCube-v1",
            num_envs=2,
            rgb_cameras=("base_camera",),
            camera_width=16,
            camera_height=16,
            env_kwargs={"max_episode_steps": 3},
        )
    )


class _ZeroDecoder:
    def __init__(self, action_shape) -> None:
        self._action_shape = action_shape

    def action_log_prob(self, features):
        return torch.zeros(features.shape[0], *self._action_shape, device=features.device), None


def _pad(env):
    state_dim = env.single_observation_space["state"].shape[-1]
    action_dim = env.single_action_space.shape[-1]
    return MultitaskPadWrapper(
        env,
        obs_dim=state_dim + STATE_DIM_EXTRA,
        action_dim=action_dim + ACTION_DIM_EXTRA,
        task_idx=1,
        task_obs_dim=state_dim,
        task_action_dim=action_dim,
    )


def _stack(env):
    image_keys = [key for key in env.single_observation_space.spaces if key.startswith("rgb_")]
    return ImageFrameStackWrapper(env, frame_stack=3, image_keys=image_keys, autoreset_in_step=True)


def _scale(env):
    return RewardScaleBiasWrapper(env, scale=2.0, bias=-0.5)


def _gail(env):
    return GAILRewardWrapper(
        env, reward_fn=lambda obs, action: torch.ones(action.shape[0], device=action.device)
    )


WRAP = {
    "scale": _scale,
    "stack": _stack,
    "repeat": lambda env: ActionRepeatWrapper(env, 2),
    "chunk": lambda env: ActionChunkWrapper(env, act_steps=2),
    "pad": _pad,
    "gail": _gail,
}
# Innermost first. Every ordered pair, and the whole set in four orders.
ORDERS = [(a, b) for a in WRAP for b in WRAP if a != b] + [
    tuple(WRAP),
    tuple(reversed(WRAP)),
    ("pad", "chunk", "gail", "stack", "scale", "repeat"),
    ("repeat", "gail", "scale", "chunk", "stack", "pad"),
]
LONG_ORDERS = ORDERS[-4:]


def _check_spaces_match(env, obs) -> None:
    for key, space in env.single_observation_space.spaces.items():
        assert tuple(obs[key].shape) == (env.num_envs, *space.shape), key
        assert env.observation_space[key].shape == (env.num_envs, *space.shape), key
    assert env.action_space.shape == (env.num_envs, *env.single_action_space.shape)


def _check_composition(make_env, order) -> None:
    inner = make_env()
    env = inner
    for name in order:
        env = WRAP[name](env)

    assert isinstance(env, BatchedEnvWrapper) and not isinstance(env, VectorEnv)
    assert env.num_envs == inner.num_envs and env.unwrapped is inner.unwrapped
    assert env.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    assert torch.device(env.device) == torch.device(inner.device)
    if hasattr(inner, "post_update_sync"):
        env.post_update_sync()
        assert inner.sync_calls == 1
    else:
        assert not hasattr(env, "post_update_sync")
    with pytest.raises(AttributeError):
        _ = env.not_an_attribute

    obs, _ = env.reset(seed=0)
    _check_spaces_match(env, obs)
    action = torch.zeros(env.action_space.shape, device=env.device)
    ended = False
    for _ in range(4):
        obs, reward, terminated, truncated, infos = env.step(action)
        _check_spaces_match(env, obs)
        assert reward.shape == terminated.shape == truncated.shape == (env.num_envs,)
        if bool((terminated | truncated).any()):
            ended = True
            assert {k: v.shape for k, v in infos["final_observation"].items()} == {
                k: v.shape for k, v in obs.items()
            }
            assert bool(infos["_final_info"].any())
    assert ended

    closed = []
    inner.close = lambda: closed.append(True)
    env.close()
    assert closed == [True]
    del inner.close
    inner.close()


@pytest.mark.parametrize("order", ORDERS, ids="-".join)
@pytest.mark.parametrize("make_env", [_adapter_env, FakeBatchedEnv], ids=["adapter", "gym-env"])
def test_wrappers_compose_in_any_order(make_env, order) -> None:
    _check_composition(make_env, order)


@pytest.mark.sim
def test_wrappers_compose_in_any_order_on_maniskill() -> None:
    for order in [("stack", "scale"), ("scale", "stack"), *LONG_ORDERS]:
        _check_composition(_maniskill_env, order)


def _state_env() -> VectorEnv:
    return make_gymnasium_env(
        GymnasiumEnvConfig(
            env_id="Pendulum-v1", num_envs=2, seed=0, env_kwargs={"max_episode_steps": 3}
        )
    )


def _skill(env):
    return SkillActionWrapper(env, _ZeroDecoder(env.single_action_space.shape), horizon=2, skill_dim=4)


@pytest.mark.parametrize(
    "chain",
    [(_skill, _scale), (_scale, _skill), (_pad, _skill, _scale), (_skill, _pad), (_skill, _gail),
     (_gail, _skill), (_skill, WRAP["chunk"])],
    ids=["skill-scale", "scale-skill", "pad-skill-scale", "skill-pad", "skill-gail", "gail-skill", "skill-chunk"],
)
def test_skill_wrapper_composes(chain) -> None:
    env = _state_env()
    for wrap in chain:
        env = wrap(env)

    obs, _ = env.reset(seed=0)
    for _ in range(3):
        _check_spaces_match(env, obs)
        obs, reward, *_ = env.step(torch.zeros(env.action_space.shape))
        assert reward.shape == (2,)
    env.close()


@pytest.mark.parametrize("outer", [_scale, WRAP["repeat"], WRAP["chunk"], _gail], ids=["scale", "repeat", "chunk", "gail"])
@pytest.mark.parametrize("dict_first", [True, False])
def test_vectorized_dict_state_composes(outer, dict_first) -> None:
    wraps = [VectorizedDictStateWrapper, outer] if dict_first else [outer, VectorizedDictStateWrapper]
    inner = FakeBatchedEnv(box_obs=True)
    env = inner
    for wrap in wraps:
        env = wrap(env)

    obs, _ = env.reset()
    _check_spaces_match(env, obs)
    for _ in range(3):
        obs, _, _, truncated, infos = env.step(torch.zeros(env.action_space.shape))
        if bool(truncated.any()):
            assert set(infos["final_observation"]) == {"state"}
    env.close()
    assert inner.closed


def _classifier(obs):
    return torch.zeros(obs["state"].shape[0])


NO_AUTORESET = {
    "scale": _scale,
    "stack": _stack,
    "classifier": lambda env: RewardClassifierWrapper(env, _classifier),
    "stages": lambda env: MultiStageBinaryRewardClassifierWrapper(env, [_classifier, _classifier]),
    "fwbw": FWBWResetFreeWrapper,
}


@pytest.mark.parametrize("order", [tuple(NO_AUTORESET), tuple(reversed(NO_AUTORESET))], ids="-".join)
def test_wrappers_for_envs_without_autoreset_compose(order) -> None:
    inner = FakeBatchedEnv(num_envs=1, autoreset=False)
    env = inner
    for name in order:
        env = NO_AUTORESET[name](env)

    assert env.num_envs == 1 and "autoreset_mode" not in env.metadata
    obs, info = env.reset()
    assert info["fwbw_direction"] == "forward" and "succeed" in info
    _check_spaces_match(env, obs)
    obs, _, _, _, info = env.step(torch.zeros(1, 1))
    _check_spaces_match(env, obs)
    assert {"classifier_success_prob", "succeed", "fwbw_direction"} <= set(info)
    env.post_update_sync()
    assert inner.sync_calls == 1
    env.close()
    assert inner.closed


def test_frame_stack_then_reward_scale_equals_the_other_order() -> None:
    first, second = _scale(_stack(_adapter_env())), _stack(_scale(_adapter_env()))
    first.reset()
    second.reset()
    for _ in range(4):
        a, b = first.step(torch.zeros(2, 1)), second.step(torch.zeros(2, 1))
        assert torch.equal(a[1], b[1]) and torch.equal(a[0]["rgb_front"], b[0]["rgb_front"])
    first.close()
    second.close()


def test_env_spec_without_reset_step_or_close_is_wrapped() -> None:
    spec = OfflineEnvSpec(
        spaces.Box(-1.0, 1.0, (3,), np.float32), spaces.Box(-1.0, 1.0, (1,), np.float32)
    )
    env = VectorizedDictStateWrapper(spec)

    assert env.num_envs == 1 and set(env.single_observation_space.spaces) == {"state"}
    for name in ("metadata", "unwrapped", "spec", "post_update_sync"):
        assert not hasattr(env, name)
    env.close()


def test_wrappers_of_the_algorithm_boundary_are_picklable() -> None:
    spec = OfflineEnvSpec(
        spaces.Box(-1.0, 1.0, (3,), np.float32), spaces.Box(-1.0, 1.0, (1,), np.float32)
    )
    env = RewardScaleBiasWrapper(VectorizedDictStateWrapper(spec), scale=2.0)

    copy = pickle.loads(pickle.dumps(env))

    assert copy.num_envs == 1 and copy.reward_scale == 2.0
    assert copy.single_observation_space == env.single_observation_space


def test_subclass_may_initialise_through_gym_wrapper_only() -> None:
    class ArmClassifiers(RewardClassifierWrapper):
        def __init__(self, env) -> None:
            gym.Wrapper.__init__(self, env)

    inner = FakeBatchedEnv(num_envs=1, autoreset=False)
    env = FWBWResetFreeWrapper(ArmClassifiers(RewardScaleBiasWrapper(inner, 2.0, -0.5)))

    assert env.num_envs == 1 and env.metadata == {}
    assert env.single_observation_space is inner.single_observation_space
    assert env.observation_space is inner.observation_space
    assert env.action_space is inner.action_space
    env.close()
    assert inner.closed


def test_subclass_may_refill_frames_outside_reset() -> None:
    class SessionStack(ImageFrameStackWrapper):
        def session_observe(self):
            return self._reset_frames(self.env._obs(), env_idx=None)

    env = SessionStack(FakeBatchedEnv(num_envs=1, autoreset=False), frame_stack=3, image_keys=("rgb_front",))
    env.reset()
    env.step(torch.zeros(1, 1))

    assert env.session_observe()["rgb_front"].eq(1).all()


def test_attribute_writes_stay_on_the_wrapper() -> None:
    inner = FakeBatchedEnv()
    env = RewardScaleBiasWrapper(inner)

    env.set_wrapper_attr("sync_calls", 5)

    assert env.sync_calls == 5 and inner.sync_calls == 0


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        (("scale", "repeat"), 2 * (2.0 * 1.0 - 0.5)),
        (("repeat", "scale"), 2.0 * 2.0 - 0.5),
        (("scale", "chunk"), 2 * (2.0 * 1.0 - 0.5)),
        (("chunk", "scale"), 2.0 * 2.0 - 0.5),
    ],
    ids=["scale-repeat", "repeat-scale", "scale-chunk", "chunk-scale"],
)
def test_bias_is_added_once_per_step_below_the_transform(order, expected) -> None:
    env = FakeBatchedEnv()
    for name in order:
        env = WRAP[name](env)
    env.reset()

    assert env.step(torch.zeros(env.action_space.shape))[1].tolist() == [expected] * 2
