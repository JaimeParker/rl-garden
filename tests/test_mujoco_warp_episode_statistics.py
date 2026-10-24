"""CustomMujocoWarpEnv reports per-env episode return/length in final_info."""
from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.sim,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

_MAX_STEPS = 200


@pytest.fixture
def pendulum():
    pytest.importorskip("mujoco_warp")
    from rl_garden.envs.mujoco_warp.tasks.inverted_pendulum_warp import (
        InvertedPendulumWarpEnv,
    )

    env = InvertedPendulumWarpEnv(nworld=2, device="cuda:0")
    env.reset()
    yield env
    env.close()


def _full_push(env) -> torch.Tensor:
    high = torch.as_tensor(env.single_action_space.high, device=env.device)
    return high.expand(env.num_envs, -1).contiguous()


def _check_episodes_against_step_counter(env, actions, expected_len, min_done_steps):
    """Steps ``env``; on every done step the reported return/length must
    equal the per-env step counter (InvertedPendulum pays 1 per step)."""
    done_steps = 0
    for _ in range(_MAX_STEPS):
        _, _, terminated, truncated, infos = env.step(actions)
        expected_len += 1
        done = terminated | truncated
        if not done.any():
            assert "final_info" not in infos
            continue
        episode = infos["final_info"]["episode"]
        assert torch.equal(infos["_final_info"], done)
        assert torch.equal(episode["episode_len"][done], expected_len[done])
        assert torch.equal(episode["return"][done], expected_len[done].float())
        expected_len[done] = 0
        done_steps += 1
        if done_steps == min_done_steps:
            return
    pytest.fail(f"fewer than {min_done_steps} done steps within {_MAX_STEPS} steps")


def test_done_steps_report_episode_return_and_length_across_auto_resets(pendulum) -> None:
    expected_len = torch.zeros(pendulum.num_envs, dtype=torch.int64, device=pendulum.device)

    _check_episodes_against_step_counter(pendulum, _full_push(pendulum), expected_len, 3)


def test_partial_reset_restarts_only_the_given_envs(pendulum) -> None:
    zero = torch.zeros_like(_full_push(pendulum))
    pendulum.step(zero)
    pendulum.step(zero)
    expected_len = torch.tensor([0, 2], dtype=torch.int64, device=pendulum.device)

    pendulum.reset(options={"env_idx": torch.tensor([0], device=pendulum.device)})

    _check_episodes_against_step_counter(pendulum, _full_push(pendulum), expected_len, 2)
