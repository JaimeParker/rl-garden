"""Receding-horizon rollout queue for action-chunked off-policy algorithms
(Q-chunking / QC).

Matches QC's own reference rollout (``3rd_party/qc/main.py``,
``main_online.py``'s ``action_queue``): the actor is invoked once every
``horizon_length`` raw env steps per env, not every step -- the other
``horizon_length - 1`` actions are replayed from a cached chunk.
Generalizes QC's single-env Python-list queue to a GPU-batched,
per-env-staggered-reset-aware queue: each env's cursor advances
independently, and a fresh chunk is computed for an env the moment it needs
one (either its cursor reached the end, or it just terminated/truncated),
all within ONE batched forward pass per ``_rollout_action`` call (the policy
is never invoked per-env).

``env.step()`` stays exactly 1:1 with ``OffPolicyAlgorithm.learn()``'s
existing per-step loop -- no env wrapper, no changes to ``off_policy.py`` at
all; this only overrides ``_rollout_action``/``_on_env_reset``/
``_post_rollout_step``.

Before ``learning_has_started``, chunking is bypassed entirely (falls
through to the base class's normal ``_rollout_action``, i.e. one fresh
independent random action per raw step) -- matches QC's own warmup behavior
exactly (``main_online.py:179-188`` samples a single raw random action per
step during warmup, not a random chunk).

Compute note: for a heavily-vectorized ``num_envs``, some env needs a
refill on almost every step (envs desync via different episode lengths), so
this rarely skips a batched forward pass entirely -- unlike QC's single-env
reference, where the H-step gap between forward passes is the actual
compute win. This is an inherent, expected consequence of vectorization
(rl-garden's normal SAC/RLPD rollout already does one batched forward pass
per step regardless of chunking), not a shortcut: the *executed action
sequence* per env still exactly matches QC's receding-horizon semantics,
which is what the RL objective depends on -- discarding a same-cost batched
forward pass's unused outputs for envs mid-chunk is simpler and no slower
than dynamically sub-batching only the envs that need a refill.

Host must define ``self.horizon_length``, ``self.num_envs``, ``self.device``,
``self.env.single_action_space`` and implement ``_sample_action_chunk(obs)
-> Tensor[(num_envs, horizon_length, *action_shape)]``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch


class ChunkedRolloutMixin:
    def _init_chunked_rollout(self) -> None:
        self._chunk_queue: Optional[torch.Tensor] = None
        self._chunk_cursor: Optional[torch.Tensor] = None

    def _sample_action_chunk(self, obs) -> torch.Tensor:
        raise NotImplementedError

    def _on_env_reset(self, obs) -> None:
        super()._on_env_reset(obs)
        act_shape = tuple(self.env.single_action_space.shape)
        self._chunk_queue = torch.zeros(
            (self.num_envs, self.horizon_length) + act_shape, device=self.device
        )
        # >= horizon_length forces an immediate refill on the first call.
        self._chunk_cursor = torch.full(
            (self.num_envs,), self.horizon_length, dtype=torch.long, device=self.device
        )

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        if not learning_has_started:
            return super()._rollout_action(obs, learning_has_started)

        needs_refill = self._chunk_cursor >= self.horizon_length
        if bool(needs_refill.any()):
            with torch.no_grad():
                new_chunk = self._sample_action_chunk(
                    self._obs_to_policy_device(obs)
                ).detach()
            mask = needs_refill.view((self.num_envs,) + (1,) * (new_chunk.ndim - 1))
            self._chunk_queue = torch.where(
                mask, new_chunk.to(self._chunk_queue.device), self._chunk_queue
            )
            self._chunk_cursor = torch.where(
                needs_refill, torch.zeros_like(self._chunk_cursor), self._chunk_cursor
            )

        env_idx = torch.arange(self.num_envs, device=self._chunk_cursor.device)
        actions = self._chunk_queue[env_idx, self._chunk_cursor]
        self._chunk_cursor = self._chunk_cursor + 1
        return actions, actions, None

    def _post_rollout_step(
        self,
        action_context: Optional[dict[str, Any]],
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos,
    ) -> None:
        super()._post_rollout_step(action_context, terminations, truncations, infos)
        done = (terminations | truncations).to(device=self._chunk_cursor.device)
        self._chunk_cursor = torch.where(
            done, torch.full_like(self._chunk_cursor, self.horizon_length), self._chunk_cursor
        )
