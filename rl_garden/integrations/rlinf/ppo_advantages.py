"""RLinf-independent conversion: RLinf's rollout batch -> auto-reset-correct GAE.

Kept separate from ``ppo_actor.py`` (which needs RLinf installed just to
import ``EmbodiedFSDPActor``) so this conversion logic is unit-testable
without RLinf, mirroring ``sac_actor.trajectory_batch_to_sample``'s role in
Phase 2.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from rl_garden.buffers.gae import compute_gae


def compute_advantages_from_rollout_batch(
    rollout_batch: dict[str, Any],
    value_head_forward: Callable[[torch.Tensor], torch.Tensor],
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert RLinf's rollout batch into auto-reset-correct advantages/returns.

    Batch shapes, verified directly against ``EmbodiedFSDPActor
    ._process_received_rollout_batch``/``process_nested_dict_for_train``
    (``RLinf/rlinf/workers/actor/fsdp_actor_worker.py``): after
    ``recv_rollout_trajectories``, ``rollout_batch["rewards"]`` is
    ``[T, B, num_action_chunks]``, while ``["terminations"]``/
    ``["prev_values"]`` carry one extra *trailing* bootstrap-step entry,
    ``[T+1, B, num_action_chunks]`` (confirmed by
    ``process_nested_dict_for_train``'s own ``value[:-1]`` stripping of
    exactly these fields before training). ``dones`` is sourced from
    ``terminations`` specifically, not the combined
    ``terminations|truncations`` field, mirroring the established
    convention in ``sac_actor.trajectory_batch_to_sample``.

    RLinf's stock ``"gae"`` registry entry
    (``rlinf/algorithms/advantages.py``) does not support auto-reset by its
    own docstring's admission -- it bootstraps every non-terminal step from
    ``values[step + 1]``, which under ``env.train.auto_reset: True`` is the
    value of the *post-reset* observation, wrongly bootstrapping a
    just-ended episode's last transition into the next episode. This fixes
    it via a separate ``final_values`` array, computed by forwarding
    ``value_head_forward`` over ``next_obs`` (populated only when
    ``rollout.collect_transitions: True`` -- the pre-reset terminal
    observation, not the post-reset one).

    ``value_head_forward`` takes a flat ``[N, obs_dim]`` tensor and returns
    ``[N, 1]`` values -- a thin closure over the model's value head,
    injected so this function stays RLinf/model-independent and directly
    testable (see ``tests/test_rlinf_ppo_advantages.py``).

    Returns ``(advantages, returns)``, each ``[T, B, 1]`` -- matching
    ``rollout_batch``'s own layout, ready to write back for
    ``train_micro_batch``'s downstream reads.
    """
    rewards = rollout_batch["rewards"].squeeze(-1)  # [T, B]
    terminations_full = rollout_batch["terminations"].squeeze(-1)  # [T+1, B]
    values_full = rollout_batch["prev_values"].squeeze(-1)  # [T+1, B]
    next_states = rollout_batch["next_obs"]["states"]  # [T, B, obs_dim]

    dones = terminations_full[:-1].float()
    last_dones = terminations_full[-1]
    values = values_full[:-1]
    last_values = values_full[-1]

    t_steps, batch_size, obs_dim = next_states.shape
    with torch.no_grad():
        next_values = value_head_forward(
            next_states.reshape(t_steps * batch_size, obs_dim)
        ).reshape(t_steps, batch_size)

    # value_head_forward runs the model's own forward pass, which moves its
    # inputs to the model's device internally (RLGardenPPOModel.
    # default_forward) -- rollout_batch's tensors (received over Ray
    # channels from the rollout/env workers) are not guaranteed to already
    # be on that same device. Normalize onto next_values' device rather
    # than moving the model's output back to CPU, since GAE's per-step
    # reverse loop (compute_gae) is cheaper to run once on whichever device
    # the value forward already used. Confirmed a real, not hypothetical,
    # mismatch by a live launch on 6017 (RuntimeError: found at least two
    # devices, cuda:0 and cpu).
    device = next_values.device
    rewards = rewards.to(device)
    dones = dones.to(device)
    values = values.to(device)
    last_values = last_values.to(device)
    last_dones = last_dones.to(device)
    final_values = dones * next_values

    advantages, returns = compute_gae(
        rewards, dones, values, final_values, last_values, last_dones, gamma, gae_lambda
    )
    return advantages.unsqueeze(-1), returns.unsqueeze(-1)
