"""BatchedEnvWrapper and its subclasses around real gymnasium vector envs
(``VectorEnv``s, which are not ``gymnasium.Env``s)."""
from __future__ import annotations

import copy
import pickle

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv

from rl_garden.envs.gymnasium import GymnasiumEnvConfig, make_gymnasium_env
from rl_garden.envs.wrappers import (
    ActionChunkWrapper,
    ActionRepeatWrapper,
    BatchedEnvWrapper,
    ImageFrameStackWrapper,
    SkillActionWrapper,
)

NUM_ENVS = 2
SKILL_DIM = 3


class _ZeroDecoder:
    def action_log_prob(self, features):
        return torch.zeros(features.shape[0], 1, device=features.device), None


def _numpy_adapter_env() -> VectorEnv:
    """SyncVectorEnv behind TorchVectorEnvAdapter; state-only, 3-step episodes."""
    return make_gymnasium_env(
        GymnasiumEnvConfig(
            env_id="Pendulum-v1", num_envs=NUM_ENVS, seed=0, env_kwargs={"max_episode_steps": 3}
        )
    )


def _torch_native_env() -> VectorEnv:
    """A torch VectorEnv without an adapter; state-only, 3-step episodes."""
    from _gymnasium_user_envs import TorchCountingVecEnv

    return TorchCountingVecEnv(NUM_ENVS, "cpu")


def _pixels_env() -> VectorEnv:
    """rgb_front (16, 16, 3) + state (2,), reward 1 per step, 3-step episodes."""
    return make_gymnasium_env(
        GymnasiumEnvConfig(
            env_id="RlgTestPixels-v0",
            num_envs=NUM_ENVS,
            seed=0,
            import_modules=("_gymnasium_user_envs",),
        )
    )


def _frame_stack(env) -> ImageFrameStackWrapper:
    return ImageFrameStackWrapper(
        env, frame_stack=3, image_keys=("rgb_front",), autoreset_in_step=True
    )


def _skill(env) -> SkillActionWrapper:
    return SkillActionWrapper(env, _ZeroDecoder(), horizon=2, skill_dim=SKILL_DIM)


WRAPPED = [
    pytest.param(_numpy_adapter_env, lambda env: ActionChunkWrapper(env, act_steps=2), id="chunk-adapter"),
    pytest.param(_torch_native_env, lambda env: ActionChunkWrapper(env, act_steps=2), id="chunk-torch"),
    pytest.param(_numpy_adapter_env, _skill, id="skill-adapter"),
    pytest.param(_torch_native_env, _skill, id="skill-torch"),
    pytest.param(_numpy_adapter_env, lambda env: ActionRepeatWrapper(env, 2), id="repeat-adapter"),
    pytest.param(_torch_native_env, lambda env: ActionRepeatWrapper(env, 2), id="repeat-torch"),
    pytest.param(_pixels_env, _frame_stack, id="frame-stack-adapter"),
]


@pytest.mark.parametrize(("make_inner", "wrap"), WRAPPED)
def test_vector_env_is_wrapped_as_a_gym_wrapper(make_inner, wrap) -> None:
    inner = make_inner()
    assert not isinstance(inner, gym.Env)

    env = wrap(inner)

    assert isinstance(env, gym.Wrapper) and isinstance(env, BatchedEnvWrapper)
    assert env.num_envs == NUM_ENVS
    assert env.unwrapped is inner.unwrapped
    assert env.metadata is inner.metadata
    assert env.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    assert env.spec == inner.spec
    assert getattr(env, "post_update_sync", None) is None
    with pytest.raises(AttributeError):
        _ = env.not_an_attribute
    env.close()
    assert inner.unwrapped.closed


@pytest.mark.parametrize(("make_inner", "wrap"), WRAPPED)
def test_vector_env_reset_and_steps_match_the_declared_spaces(make_inner, wrap) -> None:
    env = wrap(make_inner())
    action = torch.zeros(env.action_space.shape)

    obs, _ = env.reset(seed=0)
    ended = False
    for _ in range(3):
        for key, space in env.single_observation_space.spaces.items():
            assert obs[key].shape == (NUM_ENVS, *space.shape)
            assert env.observation_space[key].shape == (NUM_ENVS, *space.shape)
        obs, reward, terminated, truncated, infos = env.step(action)
        assert reward.shape == terminated.shape == truncated.shape == (NUM_ENVS,)
        if bool(truncated.any()):
            ended = True
            assert bool(infos["_final_info"].all())
            assert "episode" in infos["final_info"]
            assert {k: v.shape for k, v in infos["final_observation"].items()} == {
                k: v.shape for k, v in obs.items()
            }
    assert ended
    env.close()


def test_redefined_spaces_are_not_replaced_by_inner_ones() -> None:
    inner = _numpy_adapter_env()

    chunk = ActionChunkWrapper(inner, act_steps=4)
    assert chunk.single_action_space.shape == (4, 1)
    assert chunk.action_space.shape == (NUM_ENVS, 4, 1)
    assert chunk.single_observation_space is inner.single_observation_space
    assert chunk.observation_space is inner.observation_space

    skill = _skill(inner)
    assert skill.single_action_space.shape == (SKILL_DIM,)
    assert skill.action_space.shape == (NUM_ENVS, SKILL_DIM)
    assert skill.raw_action_space is inner.single_action_space
    inner.close()


def test_outer_wrapper_reads_the_spaces_the_wrapper_below_redefines() -> None:
    env = ActionChunkWrapper(_frame_stack(_pixels_env()), act_steps=2)

    assert env.single_observation_space["rgb_front"].shape == (3, 16, 16, 3)
    assert env.observation_space["rgb_front"].shape == (NUM_ENVS, 3, 16, 16, 3)
    assert env.single_action_space.shape == (2, 1)
    env.close()


def test_action_chunk_forwards_post_update_sync() -> None:
    class _SyncedVecEnv(type(_torch_native_env())):
        sync_calls = 0

        def post_update_sync(self) -> None:
            self.sync_calls += 1

    inner = _SyncedVecEnv(NUM_ENVS, "cpu")
    ActionChunkWrapper(inner, act_steps=2).post_update_sync()
    assert inner.sync_calls == 1


class _PicklableBatchedEnv(gym.Env):
    num_envs = 1
    observation_space = spaces.Box(-1.0, 1.0, (1, 1), np.float32)
    action_space = spaces.Box(-1.0, 1.0, (1, 1), np.float32)


def test_lookup_before_init_does_not_recurse() -> None:
    with pytest.raises(AttributeError):
        _ = BatchedEnvWrapper.__new__(BatchedEnvWrapper).num_envs
    env = BatchedEnvWrapper(_PicklableBatchedEnv())
    assert copy.copy(env).env is env.env
    assert pickle.loads(pickle.dumps(env)).num_envs == 1
