"""Standalone GAE recurrence, shared between ``RolloutBuffer`` and RLinf's PPO adapter.

Extracted out of ``RolloutBuffer.compute_returns_and_advantage`` so the same
auto-reset-aware bootstrap correction (via ``final_values``, populated from
``infos["final_observation"]`` at steps where an episode ends mid-rollout) is
available to code that doesn't otherwise use a ``RolloutBuffer`` object --
see ``rl_garden/integrations/rlinf/ppo_actor.py``'s
``compute_advantages_and_returns`` override, which converts RLinf's own
batch layout into these plain tensors rather than building a full
``RolloutBuffer``.
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    final_values: torch.Tensor,
    last_values: torch.Tensor,
    last_dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard (non-finite-horizon) GAE with an auto-reset bootstrap fix.

    All of ``rewards``/``dones``/``values``/``final_values`` are
    ``(num_steps, num_envs)``; ``last_values``/``last_dones`` are
    ``(num_envs,)`` (the bootstrap value/done flag one step past the last
    stored step). ``final_values[step]`` is nonzero only at steps where an
    episode ended mid-rollout (auto-reset) -- it substitutes the last
    transition's own terminal-state value estimate in place of the
    post-reset next observation's value, which would otherwise wrongly
    bootstrap into the following episode. Returns ``(advantages, returns)``,
    each ``(num_steps, num_envs)``.
    """
    num_steps = rewards.shape[0]
    last_values = last_values.reshape(-1).to(rewards.device)
    last_dones = last_dones.reshape(-1).float().to(rewards.device)
    advantages = torch.zeros_like(rewards)

    last_gae_lam = torch.zeros_like(last_values)
    for step in reversed(range(num_steps)):
        if step == num_steps - 1:
            next_not_done = 1.0 - last_dones
            next_values = last_values
        else:
            next_not_done = 1.0 - dones[step + 1]
            next_values = values[step + 1]
        real_next_values = next_not_done * next_values + final_values[step]
        delta = rewards[step] + gamma * real_next_values - values[step]
        last_gae_lam = delta + gamma * gae_lambda * next_not_done * last_gae_lam
        advantages[step] = last_gae_lam

    returns = advantages + values
    return advantages, returns
