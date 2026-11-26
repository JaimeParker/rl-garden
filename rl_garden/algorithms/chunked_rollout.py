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
``_post_rollout_step``/``_replay_buffer_step_kwargs`` and the eval hooks.

Evaluation executes chunks the same way (``3rd_party/qc/evaluation.py:82-99``),
from a queue of its own (``ChunkedEvalMixin``): an evaluation in the middle of
training leaves the rollout's queue and cursors alone.

Every rollout row also stores where its episode ends, terminated or truncated
(``terminals=float(done)``, ``3rd_party/qc/main.py:270``), so no chunk window
joins two episodes.

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

Host must define ``self.horizon_length``, ``self.env.single_action_space`` and
implement ``_sample_action_chunk(obs) -> Tensor[(num_envs, horizon_length,
*action_shape)]``; its ``_eval_action`` returns the flat chunk its policy predicts.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import torch


class ActionChunkQueue:
    """One chunk and one cursor per env, sized by the first sampled chunk."""

    def __init__(self, horizon_length: int) -> None:
        self.horizon_length = horizon_length
        self.chunks: Optional[torch.Tensor] = None
        self.cursor: Optional[torch.Tensor] = None

    def next_action(self, sample_chunks: Callable[[], torch.Tensor]) -> torch.Tensor:
        """The next action of every env's chunk. ``sample_chunks`` is called
        when some env needs a new one; the others keep theirs."""
        if self.cursor is None:
            self.chunks = sample_chunks()
            self.cursor = torch.zeros(
                self.chunks.shape[0], dtype=torch.long, device=self.chunks.device
            )
        else:
            needs_refill = self.cursor >= self.horizon_length
            if bool(needs_refill.any()):
                new_chunks = sample_chunks()
                mask = needs_refill.view((-1,) + (1,) * (new_chunks.ndim - 1))
                self.chunks = torch.where(mask, new_chunks, self.chunks)
                self.cursor = torch.where(
                    needs_refill, torch.zeros_like(self.cursor), self.cursor
                )
        env_idx = torch.arange(self.cursor.shape[0], device=self.cursor.device)
        actions = self.chunks[env_idx, self.cursor]
        self.cursor = self.cursor + 1
        return actions

    def restart(self, done: torch.Tensor) -> None:
        """Drop the rest of the chunk of every ``done`` env."""
        if self.cursor is not None:
            done = done.reshape(-1).to(self.cursor.device)
            self.cursor = self.cursor.masked_fill(done, self.horizon_length)


class ChunkedEvalMixin:
    """Executes the flat chunk ``_eval_action`` returns one action per eval env
    step, in every evaluation loop. Alone, it serves an algorithm that never
    rolls out (QAM, QGF: ``3rd_party/qam/evaluation.py:91-100``,
    ``3rd_party/qgf/utils/evaluation.py:234-264``)."""

    _eval_chunks: Optional[ActionChunkQueue] = None

    def _eval_start_hook(self) -> None:
        super()._eval_start_hook()
        self._eval_chunks = ActionChunkQueue(self.horizon_length)

    def _eval_action_and_critic_action(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """The critic's action is the whole chunk the executed action is from:
        the one chosen for ``obs`` on an env's decision steps, where every
        chunk starts, and an earlier one in between."""

        def predict_chunks() -> torch.Tensor:
            flat = self._eval_action(obs)
            action_shape = self.env.single_action_space.shape
            return flat.reshape(flat.shape[0], self.horizon_length, *action_shape)

        action = self._eval_chunks.next_action(predict_chunks)
        return action, self._eval_chunks.chunks.flatten(start_dim=1)

    def _eval_step_hook(
        self, obs_before, critic_action, rewards, terminations, truncations, infos
    ) -> None:
        super()._eval_step_hook(
            obs_before, critic_action, rewards, terminations, truncations, infos
        )
        self._eval_chunks.restart(terminations | truncations)


class ChunkedRolloutMixin(ChunkedEvalMixin):
    def _init_chunked_rollout(self) -> None:
        self._rollout_chunks: Optional[ActionChunkQueue] = None

    def _sample_action_chunk(self, obs) -> torch.Tensor:
        raise NotImplementedError

    def _on_env_reset(self, obs) -> None:
        super()._on_env_reset(obs)
        self._rollout_chunks = ActionChunkQueue(self.horizon_length)

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        if not learning_has_started:
            return super()._rollout_action(obs, learning_has_started)

        with torch.no_grad():
            actions = self._rollout_chunks.next_action(
                lambda: self._sample_action_chunk(self._obs_to_policy_device(obs))
            )
        return actions, actions, None

    def _replay_buffer_step_kwargs(
        self, terminations: torch.Tensor, truncations: torch.Tensor
    ) -> dict[str, Any]:
        return {
            **super()._replay_buffer_step_kwargs(terminations, truncations),
            "episode_end": terminations | truncations,
        }

    def _post_rollout_step(
        self,
        action_context: Optional[dict[str, Any]],
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos,
    ) -> None:
        super()._post_rollout_step(action_context, terminations, truncations, infos)
        self._rollout_chunks.restart(terminations | truncations)
