"""SAC smoke tests: 256 steps of state + rgbd training without errors."""
from __future__ import annotations

import pytest
import torch

from rl_garden.algorithms import SAC
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ManiSkill smoke tests require CUDA."
)


def test_state_sac_smoke():
    cfg = ManiSkillEnvConfig(
        env_id="PickCube-v1", num_envs=4, obs_mode="state",
        partial_reset=False, ignore_terminations=True,
    )
    env = make_maniskill_env(cfg)
    try:
        agent = SAC(
            env=env, eval_env=None,
            buffer_size=1024, learning_starts=128, training_freq=64, utd=0.5,
            batch_size=64, eval_freq=0, log_freq=128,
            device="cuda", buffer_device="cuda", seed=0,
        )
        agent.learn(total_timesteps=256)
        assert agent._global_step >= 256
        assert agent._global_update > 0
    finally:
        env.close()


def test_state_env_supports_pd_ee_twist():
    cfg = ManiSkillEnvConfig(
        env_id="PickCube-v1", num_envs=1, obs_mode="state",
        control_mode="pd_ee_twist",
        partial_reset=False, ignore_terminations=True,
    )
    env = make_maniskill_env(cfg)
    try:
        assert env.action_space.shape == (1, 7)
    finally:
        env.close()


def test_rgbd_sac_smoke():
    cfg = ManiSkillEnvConfig(
        env_id="PickCube-v1", num_envs=4, obs_mode="rgb", include_state=True,
        partial_reset=False, ignore_terminations=True,
        camera_width=64, camera_height=64,
    )
    env = make_maniskill_env(cfg)
    try:
        agent = SAC(
            env=env, eval_env=None,
            buffer_size=512, learning_starts=128, training_freq=64, utd=0.25,
            batch_size=32, eval_freq=0, log_freq=128,
            device="cuda", buffer_device="cuda", seed=0,
            image_keys=("rgb",),
        )
        agent.learn(total_timesteps=256)
        assert agent._global_step >= 256
        assert agent._global_update > 0
    finally:
        env.close()
