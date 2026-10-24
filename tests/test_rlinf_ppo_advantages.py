from __future__ import annotations

import pytest
import torch

from rl_garden.integrations.rlinf.ppo_advantages import (
    compute_advantages_from_rollout_batch,
    rlinf_rewards_include_truncation_bootstrap,
)

GAMMA = 0.5
# Two envs, T = 3 steps. V(obs_t) = 1, 2, 3 for t = 0, 1, 2; the trailing
# prev_values entry (value after the last step) is 4. V(next_obs[t]) is
# 10, 20, 30, so a wrong next-obs index shows up in the returns.
STEP_VALUES = [1.0, 2.0, 3.0, 4.0]
NEXT_OBS_VALUES = [10.0, 20.0, 30.0]
# Env 1 never ends an episode: G2 = 1 + 0.5 * 4, G1 = 1 + 0.5 * G2, ...
UNCUT_RETURNS = [2.25, 2.5, 3.0]


def _flags(env0_step_flags: list[int]) -> torch.Tensor:
    """RLinf boundary layout [T + 1, B, 1]: index 0 is the epoch-start entry,
    index t + 1 is step t's post-action flag. Env 1 never ends."""
    full = [0, *env0_step_flags]
    return torch.tensor([[[f], [0]] for f in full], dtype=torch.bool)


def _rollout_batch(
    terminated_steps: list[int],
    truncated_steps: list[int],
    rewards_env0: list[float] | None = None,
) -> dict:
    rewards_env0 = rewards_env0 or [1.0, 1.0, 1.0]
    rewards = torch.tensor([[[r], [1.0]] for r in rewards_env0])
    prev_values = torch.tensor([[[v], [v]] for v in STEP_VALUES])
    # value_head_forward below sums the state, so V(next_obs[t]) = state.
    next_states = torch.tensor([[[v], [v]] for v in NEXT_OBS_VALUES])
    return {
        "rewards": rewards,
        "terminations": _flags(terminated_steps),
        "truncations": _flags(truncated_steps),
        "dones": _flags(terminated_steps) | _flags(truncated_steps),
        "prev_values": prev_values,
        "next_obs": {"states": next_states},
    }


def _value_head_forward(states: torch.Tensor) -> torch.Tensor:
    return states.sum(dim=-1, keepdim=True)


def _returns(batch: dict, *, rewards_include_truncation_bootstrap: bool = False):
    advantages, returns = compute_advantages_from_rollout_batch(
        batch,
        _value_head_forward,
        GAMMA,
        1.0,
        rewards_include_truncation_bootstrap=rewards_include_truncation_bootstrap,
    )
    assert advantages.shape == returns.shape == (3, 2, 1)
    values = batch["prev_values"][:-1]
    assert torch.allclose(advantages, returns - values)
    return returns.squeeze(-1)


def _assert_returns(returns: torch.Tensor, env0: list[float]) -> None:
    assert torch.allclose(returns[:, 0], torch.tensor(env0))
    assert torch.allclose(returns[:, 1], torch.tensor(UNCUT_RETURNS))


def test_no_episode_end_bootstraps_from_trailing_value():
    _assert_returns(_returns(_rollout_batch([0, 0, 0], [0, 0, 0])), UNCUT_RETURNS)


def test_termination_mid_window_cuts_without_bootstrap():
    # Step 1 terminates: G1 = r1 (no V(next_obs[1]) = 20, no V(obs_2) = 3),
    # G0 = 1 + 0.5 * G1; step 2 starts a new episode: G2 = 1 + 0.5 * 4.
    returns = _returns(_rollout_batch([0, 1, 0], [0, 0, 0]))
    _assert_returns(returns, [1.5, 1.0, 3.0])


def test_truncation_mid_window_bootstraps_from_terminal_obs_and_cuts():
    # Step 1 truncates: G1 = r1 + 0.5 * V(next_obs[1]) = 1 + 10; step 2's
    # return (3) does not leak into G1.
    returns = _returns(_rollout_batch([0, 0, 0], [0, 1, 0]))
    _assert_returns(returns, [6.5, 11.0, 3.0])


def test_rlinf_reward_bootstrap_is_not_stacked():
    # RLinf already added 0.5 * V(final_obs) = 10 to step 1's reward.
    batch = _rollout_batch([0, 0, 0], [0, 1, 0], rewards_env0=[1.0, 11.0, 1.0])

    def _must_not_run(states: torch.Tensor) -> torch.Tensor:
        raise AssertionError("value head must not run when rewards carry the bootstrap")

    advantages, returns = compute_advantages_from_rollout_batch(
        batch, _must_not_run, GAMMA, 1.0, rewards_include_truncation_bootstrap=True
    )
    _assert_returns(returns.squeeze(-1), [6.5, 11.0, 3.0])
    assert advantages.shape == (3, 2, 1)


def test_termination_and_truncation_at_same_step_does_not_bootstrap():
    returns = _returns(_rollout_batch([0, 1, 0], [0, 1, 0]))
    _assert_returns(returns, [1.5, 1.0, 3.0])


def test_last_step_termination_ignores_trailing_value():
    returns = _returns(_rollout_batch([0, 0, 1], [0, 0, 0]))
    # G2 = r2 (not 1 + 0.5 * 4), G1 = 1 + 0.5, G0 = 1 + 0.75.
    _assert_returns(returns, [1.75, 1.5, 1.0])


def test_last_step_truncation_bootstraps_from_terminal_obs():
    returns = _returns(_rollout_batch([0, 0, 0], [0, 0, 1]))
    # G2 = 1 + 0.5 * V(next_obs[2]) = 1 + 15 (not the trailing value 4).
    _assert_returns(returns, [5.5, 9.0, 16.0])


def test_epoch_start_boundary_entry_does_not_cut():
    batch = _rollout_batch([0, 0, 0], [0, 0, 0])
    batch["terminations"][0] = True
    batch["truncations"][0] = True
    _assert_returns(_returns(batch), UNCUT_RETURNS)


def test_gae_lambda_below_one_matches_hand_computation():
    # Env 0 truncates at step 1; lambda = 0.5.
    # delta2 = 1 + 0.5*4 - 3 = 0;  A2 = 0
    # delta1 = 1 + 0.5*20 - 2 = 9; A1 = 9 (cut, nothing carried from step 2)
    # delta0 = 1 + 0.5*2 - 1 = 1;  A0 = 1 + 0.5*0.5*9 = 3.25
    advantages, _ = compute_advantages_from_rollout_batch(
        _rollout_batch([0, 0, 0], [0, 1, 0]),
        _value_head_forward,
        GAMMA,
        0.5,
        rewards_include_truncation_bootstrap=False,
    )
    assert torch.allclose(advantages[:, 0, 0], torch.tensor([3.25, 9.0, 0.0]))


@pytest.mark.parametrize("field", ["terminations", "truncations", "prev_values"])
def test_rejects_boundary_field_without_extra_step(field):
    batch = _rollout_batch([0, 0, 0], [0, 0, 0])
    batch[field] = batch[field][:-1]
    with pytest.raises(ValueError, match=field):
        compute_advantages_from_rollout_batch(
            batch,
            _value_head_forward,
            GAMMA,
            1.0,
            rewards_include_truncation_bootstrap=False,
        )


@pytest.mark.parametrize(
    ("auto_reset", "rollout_cfg", "expected"),
    [
        (True, {}, True),
        (True, {"collect_final_values": True}, True),
        (True, {"collect_final_values": False}, False),
        (False, {}, False),
    ],
)
def test_rlinf_rewards_include_truncation_bootstrap(auto_reset, rollout_cfg, expected):
    cfg = {"env": {"train": {"auto_reset": auto_reset}}, "rollout": rollout_cfg}
    assert rlinf_rewards_include_truncation_bootstrap(cfg) is expected
