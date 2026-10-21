"""Pure-torch check of the shared Cartpole reward (no Isaac Lab needed)."""
from __future__ import annotations

import torch

from rl_garden.envs.isaaclab.tasks_common.cartpole import _compute_rewards


def test_cartpole_reward_terms() -> None:
    pole_pos = torch.tensor([0.5, 0.0])
    pole_vel = torch.tensor([2.0, 0.0])
    cart_pos = torch.tensor([1.0, 0.0])
    cart_vel = torch.tensor([-4.0, 0.0])
    terminated = torch.tensor([False, True])

    reward = _compute_rewards(1.0, -2.0, -1.0, -0.01, -0.005, pole_pos, pole_vel, cart_pos, cart_vel, terminated)

    # env 0 alive: 1 - 0.5**2 - 0.01*4 - 0.005*2; env 1 terminated at rest: -2.
    expected = torch.tensor([1.0 - 0.25 - 0.04 - 0.01, -2.0])
    assert torch.allclose(reward, expected)
