"""RLinf-independent conversion: RLinf's rollout batch -> auto-reset-correct GAE.

Kept separate from ``ppo_actor.py`` (which needs RLinf installed just to
import ``EmbodiedFSDPActor``) so this conversion logic is unit-testable
without RLinf, mirroring ``sac_actor.trajectory_batch_to_sample``'s role in
Phase 2.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from rl_garden.buffers.gae import compute_gae


def rlinf_rewards_include_truncation_bootstrap(cfg: Any) -> bool:
    """Whether RLinf already added ``gamma * V(final_obs)`` to truncated rewards.

    RLinf's collector (``EnvTransition.compute_rewards`` in
    ``rlinf/data/schema/embodied_types.py``) adds the truncation bootstrap to
    the reward of every auto-reset truncation whenever the rollout produced
    terminal values, i.e. ``env.train.auto_reset`` is on and
    ``rollout.collect_final_values`` is left at its default ``True``. Under
    ``algorithm.bootstrap_type: always`` it adds it at terminations too.
    ``collect_final_values: False`` also drops the trailing ``prev_values``
    entry, which ``compute_advantages_from_rollout_batch`` rejects.
    """
    auto_reset = bool(cfg["env"]["train"]["auto_reset"])
    collect_final_values = bool(cfg["rollout"].get("collect_final_values", True))
    return auto_reset and collect_final_values


def compute_advantages_from_rollout_batch(
    rollout_batch: dict[str, Any],
    value_head_forward: Callable[[torch.Tensor], torch.Tensor],
    gamma: float,
    gae_lambda: float,
    *,
    rewards_include_truncation_bootstrap: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert RLinf's rollout batch into auto-reset-correct advantages/returns.

    RLinf batch layout (``num_action_chunks == 1``, after
    ``process_nested_dict_for_adv``), as built by ``Trajectory.from_steps``
    (``rlinf/data/schema/embodied_types.py``) and consumed by RLinf's own
    GAE (``rlinf/algorithms/advantages.py``, ``dones[step + 1]`` /
    ``values[step + 1]`` for step ``t``):

    - ``rewards``: ``[T, B, 1]``, ``rewards[t]`` is step ``t``'s reward.
    - ``terminations``/``truncations``/``dones``: ``[T + 1, B, 1]`` with a
      LEADING entry -- index 0 is the epoch-start boundary (always zero,
      ``EnvWorker.bootstrap_step``), index ``t + 1`` is step ``t``'s
      post-action flag. This is exactly ``compute_gae``'s episode-start
      convention, so the flags are passed through unshifted.
    - ``prev_values``: ``[T + 1, B, 1]`` with a TRAILING entry --
      ``prev_values[t] = V(obs_t)``, ``prev_values[T]`` is the value of the
      observation after the last step.
    - ``next_obs``: ``[T, B, ...]`` (``rollout.collect_transitions: True``),
      the pre-reset observation after step ``t``.

    Every termination or truncation cuts the GAE recursion. The truncation
    bootstrap ``gamma * V(next_obs[t])`` (bootstrap at truncation only, 0 at
    termination) comes from exactly one place: RLinf's rewards when
    ``rewards_include_truncation_bootstrap`` (see
    ``rlinf_rewards_include_truncation_bootstrap``), otherwise
    ``final_values`` computed here from ``next_obs`` via
    ``value_head_forward`` (flat ``[N, obs_dim]`` -> ``[N, 1]``).

    Returns ``(advantages, returns)``, each ``[T, B, 1]``.
    """
    rewards = rollout_batch["rewards"].squeeze(-1)  # [T, B]
    terminations_full = rollout_batch["terminations"].squeeze(-1).bool()  # [T+1, B]
    truncations_full = rollout_batch["truncations"].squeeze(-1).bool()  # [T+1, B]
    values_full = rollout_batch["prev_values"].squeeze(-1)  # [T+1, B]
    t_steps = rewards.shape[0]
    for name, tensor in (
        ("terminations", terminations_full),
        ("truncations", truncations_full),
        ("prev_values", values_full),
    ):
        if tensor.shape[0] != t_steps + 1:
            raise ValueError(
                f"rollout_batch[{name!r}] must have T + 1 = {t_steps + 1} steps "
                f"(rewards has T = {t_steps}), got {tensor.shape[0]}."
            )

    episode_starts_full = terminations_full | truncations_full
    dones = episode_starts_full[:-1].float()
    last_dones = episode_starts_full[-1]
    values = values_full[:-1]
    last_values = values_full[-1]
    truncated_steps = truncations_full[1:] & ~terminations_full[1:]  # [T, B]

    if rewards_include_truncation_bootstrap or not truncated_steps.any():
        final_values = torch.zeros_like(values)
    else:
        next_states = rollout_batch["next_obs"]["states"]  # [T, B, obs_dim]
        _, batch_size, obs_dim = next_states.shape
        with torch.no_grad():
            next_values = value_head_forward(
                next_states.reshape(t_steps * batch_size, obs_dim)
            ).reshape(t_steps, batch_size)
        # value_head_forward runs the model's own forward pass, which moves
        # its inputs to the model's device -- rollout_batch's tensors
        # (received over Ray channels) are not guaranteed to be on that
        # device (a live 6017 launch hit "found at least two devices").
        # compute_gae runs on rewards' device, so move the per-step inputs
        # to next_values' device once.
        device = next_values.device
        rewards = rewards.to(device)
        dones = dones.to(device)
        values = values.to(device)
        last_values = last_values.to(device)
        last_dones = last_dones.to(device)
        final_values = truncated_steps.to(device) * next_values

    advantages, returns = compute_gae(
        rewards, dones, values, final_values, last_values, last_dones, gamma, gae_lambda
    )
    return advantages.unsqueeze(-1), returns.unsqueeze(-1)
