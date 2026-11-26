"""Where each backend puts ``reward_scale``/``reward_bias`` in its wrapper stack:
once per ``step()`` of the env it returns, inside a builder's action chunk (the
table in ``docs/guides/extending.md``, section 7)."""
from __future__ import annotations

import sys
import types

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.envs.vector_env import TorchVectorEnvAdapter
from rl_garden.envs.wrappers import (
    ActionChunkWrapper,
    ActionRepeatWrapper,
    ImageFrameStackWrapper,
    RewardScaleBiasWrapper,
)

SCALE, BIAS, CHUNK = 2.0, -0.5, 2
PIXEL_STACK = (TorchVectorEnvAdapter, RewardScaleBiasWrapper, ImageFrameStackWrapper)
STATE_STACK = (TorchVectorEnvAdapter, RewardScaleBiasWrapper)


def _wrapper_types(env) -> tuple[type, ...]:
    """Wrapper classes from the adapter (or the bare vector env) outwards."""
    types_outside_in = []
    while True:
        types_outside_in.append(type(env))
        if isinstance(env, TorchVectorEnvAdapter) or not isinstance(env, gym.Wrapper):
            return tuple(reversed(types_outside_in))
        env = env.env


def _assert_bias_once_per_backend_step(make, expected_stack, *, steps: int = 2) -> None:
    """``make(**reward_kwargs)``; ``expected_stack`` omits a bare vector env."""
    transformed = make(reward_scale=SCALE, reward_bias=BIAS)
    stack = _wrapper_types(transformed)
    assert tuple(cls for cls in stack if cls in (*PIXEL_STACK, ActionRepeatWrapper)) == expected_stack

    raw, chunked = make(), ActionChunkWrapper(transformed, CHUNK)
    raw.reset(seed=0)
    chunked.reset(seed=0)
    for _ in range(steps):
        zero = torch.zeros(raw.action_space.shape, device=raw.device)
        raw_rewards = []
        for _ in range(CHUNK):
            _, reward, terminated, truncated, _ = raw.step(zero)
            assert not bool((terminated | truncated).any()), "the check covers steps without episode ends"
            raw_rewards.append(reward)
        reward = chunked.step(torch.zeros(chunked.action_space.shape, device=raw.device))[1]
        expected = sum(SCALE * raw_reward + BIAS for raw_reward in raw_rewards)
        assert torch.equal(reward, expected)
    raw.close()
    chunked.close()


def test_gymnasium() -> None:
    from rl_garden.envs.gymnasium import GymnasiumEnvConfig, make_gymnasium_env

    _assert_bias_once_per_backend_step(
        lambda **kwargs: make_gymnasium_env(
            GymnasiumEnvConfig(
                env_id="RlgTestPixels-v0", num_envs=2, seed=0, frame_stack=2,
                image_keys=("rgb_front",), import_modules=("_gymnasium_user_envs",), **kwargs,
            )
        ),
        PIXEL_STACK,
        steps=1,
    )


def test_custom() -> None:
    from rl_garden.envs.custom.config import CustomEnvConfig
    from rl_garden.envs.custom.env import make_custom_env

    _assert_bias_once_per_backend_step(
        lambda **kwargs: make_custom_env(
            CustomEnvConfig(env_id="PointReach-v0", num_envs=2, seed=0, device="cpu", **kwargs)
        ),
        STATE_STACK,
    )


def test_ogbench(monkeypatch) -> None:
    from rl_garden.envs.ogbench.config import OGBenchEnvConfig
    from rl_garden.envs.ogbench.env import make_ogbench_env

    class Pixels(gym.Env):
        observation_space = spaces.Box(0, 255, (4, 4, 3), np.uint8)
        action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros((4, 4, 3), np.uint8), {}

        def step(self, action):
            return np.zeros((4, 4, 3), np.uint8), 1.0, False, False, {"success": 0.0}

    monkeypatch.setitem(sys.modules, "ogbench", types.SimpleNamespace())
    monkeypatch.setattr("gymnasium.make", lambda env_id, **kwargs: Pixels())
    _assert_bias_once_per_backend_step(
        lambda **kwargs: make_ogbench_env(
            OGBenchEnvConfig(
                env_id="visual-x-v0", num_envs=2, seed=0, device="cpu", frame_stack=2,
                pixel_camera="front", **kwargs,
            )
        ),
        PIXEL_STACK,
    )


def test_rlbench(monkeypatch) -> None:
    from test_rlbench_env import _install_fake_rlbench, _with_images_task_class

    from rl_garden.envs.rlbench.config import RLBenchEnvConfig
    from rl_garden.envs.rlbench.env import make_rlbench_env

    _install_fake_rlbench(monkeypatch, task_class=_with_images_task_class())
    _assert_bias_once_per_backend_step(
        lambda **kwargs: make_rlbench_env(
            RLBenchEnvConfig(
                task_name="reach_target", num_envs=2, seed=0, device="cpu", frame_stack=2,
                rgb_cameras=("front",), **kwargs,
            )
        ),
        PIXEL_STACK,
    )


def test_d4rl_legacy(monkeypatch) -> None:
    from test_d4rl_legacy_env import _FakeBinaryEnv

    from rl_garden.envs.d4rl_legacy.config import D4RLLegacyEnvConfig
    from rl_garden.envs.d4rl_legacy.env import make_d4rl_legacy_env

    class LongEnv(_FakeBinaryEnv):
        def __init__(self):
            super().__init__()
            self.spec = types.SimpleNamespace(max_episode_steps=50)

        def step(self, action):
            self.steps += 1
            return np.full(3, self.steps, dtype=np.float64), float(self.steps), False, {}

    monkeypatch.setattr("rl_garden.envs.d4rl_legacy.env._make_legacy_env", lambda env_id: LongEnv())
    _assert_bias_once_per_backend_step(
        lambda **kwargs: make_d4rl_legacy_env(
            D4RLLegacyEnvConfig(env_id="antmaze-test-v2", num_envs=1, device="cpu", **kwargs)
        ),
        STATE_STACK,
    )


def test_robomimic(monkeypatch) -> None:
    from test_robomimic_env import _cfg, _FakeRobosuiteEnv

    from rl_garden.envs.robomimic.env import make_robomimic_env

    monkeypatch.setattr(
        "rl_garden.envs.robomimic.env._make_robosuite_env", lambda cfg: _FakeRobosuiteEnv()
    )
    _assert_bias_once_per_backend_step(
        lambda **kwargs: make_robomimic_env(_cfg(horizon=50, **kwargs)), STATE_STACK
    )


@pytest.mark.parametrize(("repeat", "repeat_wrappers"), [(1, ()), (3, (ActionRepeatWrapper,))])
def test_isaaclab_adds_the_bias_per_decision(monkeypatch, repeat, repeat_wrappers) -> None:
    from test_isaaclab_adapter import (
        _make_env_with_fake_build,
        _ScriptedExtrasEnv,
        _visual_obs,
    )

    def make(**kwargs):
        script = [([float(i), 1.0], [False, False], [False, False]) for i in range(20)]
        raw_env = _ScriptedExtrasEnv(script, obs=_visual_obs())
        raw_env.close = lambda: None
        return _make_env_with_fake_build(
            monkeypatch, raw_env, is_visual=True, frame_stack=2, action_repeat=repeat, **kwargs
        )

    _assert_bias_once_per_backend_step(
        make, (*repeat_wrappers, ImageFrameStackWrapper, RewardScaleBiasWrapper)
    )


@pytest.mark.sim
def test_mujoco() -> None:
    from rl_garden.envs.mujoco.config import MujocoEnvConfig
    from rl_garden.envs.mujoco.env import make_mujoco_env

    _assert_bias_once_per_backend_step(
        lambda **kwargs: make_mujoco_env(
            MujocoEnvConfig(env_id="Hopper-v4", num_envs=2, seed=0, device="cpu", **kwargs)
        ),
        STATE_STACK,
    )


@pytest.mark.sim
def test_dm_control_adds_the_bias_per_decision() -> None:
    pytest.importorskip("dm_control")
    from rl_garden.envs.dm_control.config import DMControlEnvConfig
    from rl_garden.envs.dm_control.env import make_dm_control_env

    _assert_bias_once_per_backend_step(
        lambda **kwargs: make_dm_control_env(
            DMControlEnvConfig(
                env_id="cartpole-swingup", num_envs=2, seed=0, device="cpu", action_repeat=4,
                frame_stack=2, rgb_cameras=("default",), image_size=(16, 16), **kwargs,
            )
        ),
        PIXEL_STACK,
    )


@pytest.mark.sim
def test_metaworld() -> None:
    pytest.importorskip("metaworld")
    from rl_garden.envs.metaworld.config import MetaWorldEnvConfig
    from rl_garden.envs.metaworld.env import make_metaworld_env

    _assert_bias_once_per_backend_step(
        lambda **kwargs: make_metaworld_env(
            MetaWorldEnvConfig(env_id="reach-v3", num_envs=1, seed=0, device="cpu", **kwargs)
        ),
        STATE_STACK,
    )


@pytest.mark.sim
@pytest.mark.gpu
def test_mujoco_warp() -> None:
    pytest.importorskip("mujoco_warp")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from rl_garden.envs.mujoco_warp.config import MujocoWarpEnvConfig
    from rl_garden.envs.mujoco_warp.env import make_mujoco_warp_env

    _assert_bias_once_per_backend_step(
        lambda **kwargs: make_mujoco_warp_env(
            MujocoWarpEnvConfig(
                env_id="RlGarden-InvertedPendulum-Warp-v0", num_envs=2, seed=0, device="cuda:0", **kwargs
            )
        ),
        (RewardScaleBiasWrapper,),
        steps=1,
    )


@pytest.mark.sim
def test_maniskill_stacks_frames_inside_the_vector_env() -> None:
    pytest.importorskip("mani_skill")
    from rl_garden.envs.maniskill import ManiSkillEnvConfig, make_maniskill_env

    def make(**kwargs):
        return make_maniskill_env(
            ManiSkillEnvConfig(
                env_id="PickCube-v1", num_envs=2, frame_stack=2, rgb_cameras=("base_camera",),
                camera_width=16, camera_height=16, **kwargs,
            )
        )

    _assert_bias_once_per_backend_step(make, (RewardScaleBiasWrapper,))
