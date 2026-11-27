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
all. ``ChunkedRolloutMixin`` owns the rollout hooks; its parent
``ActionChunkingMixin`` owns action-space tiling, replay construction,
training-batch flattening, evaluation queues and horizon metadata.

Evaluation executes chunks the same way (``3rd_party/qc/evaluation.py:82-99``),
from a queue of its own (``ActionChunkingMixin``): an evaluation in the middle of
training leaves the rollout's queue and cursors alone.

Every rollout row also stores where its episode ends, terminated or truncated
(``terminals=float(done)``, ``3rd_party/qc/main.py:270``), so no chunk window
joins two episodes.

Before ``learning_has_started``, when no initial training phase is active,
chunking is bypassed entirely (falls
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

Put the mixin leftmost. The host calls ``_init_action_chunking(horizon_length)``
before model setup and provides normalized Dict observations, a flat Box action
space, ``num_envs``, ``buffer_size``, ``buffer_device``, ``gamma`` and ``device``.
Build the policy with ``_policy_action_space()`` and provide a cooperative
``_sample_train_batch`` returning a chunked sample dataclass. Losses retain
their own validity weighting; SAC-specific glue stays in ACRLPDCore.
The default rollout sampler reshapes ``policy.predict(obs, deterministic=False)``
to ``(B, H, *action_shape)``; ``_eval_action`` returns a flat chunk.
"""
from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers.chunked_replay_buffer import ChunkedReplayBuffer


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


class ActionChunkingMixin:
    """Shared action space, replay, batch and metadata plumbing for chunked hosts.

    Executes the flat chunk ``_eval_action`` returns one action per eval env
    step, in every evaluation loop. Alone, it serves an algorithm that never
    rolls out (QAM, QGF: ``3rd_party/qam/evaluation.py:91-100``,
    ``3rd_party/qgf/utils/evaluation.py:234-264``)."""

    _eval_chunks: Optional[ActionChunkQueue] = None

    def _init_action_chunking(self, horizon_length: int) -> None:
        if horizon_length < 1:
            raise ValueError(f"horizon_length must be >= 1, got {horizon_length}")
        self.horizon_length = horizon_length

    def _policy_action_space(self) -> spaces.Box:
        action_space = self.env.single_action_space
        assert isinstance(action_space, spaces.Box), (
            f"{type(self).__name__} requires a flat Box action space."
        )
        low = np.tile(np.asarray(action_space.low, dtype=np.float32).reshape(-1), self.horizon_length)
        high = np.tile(np.asarray(action_space.high, dtype=np.float32).reshape(-1), self.horizon_length)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _chunked_replay_buffer(self, *, num_envs: int, buffer_size: int) -> ChunkedReplayBuffer:
        return ChunkedReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=num_envs,
            buffer_size=buffer_size,
            horizon_length=self.horizon_length,
            gamma=self.gamma,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    def _build_replay_buffer(self) -> ChunkedReplayBuffer:
        return self._chunked_replay_buffer(num_envs=self.num_envs, buffer_size=self.buffer_size)

    def _sample_train_batch(self, batch_size: int):
        data = super()._sample_train_batch(batch_size)
        flat_actions = data.actions.reshape(data.actions.shape[0], -1)
        return dataclasses.replace(data, actions=flat_actions)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {**super()._checkpoint_metadata(), "horizon_length": self.horizon_length}

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


class ChunkedRolloutMixin(ActionChunkingMixin):
    _rollout_chunks: Optional[ActionChunkQueue] = None

    def _sample_action_chunk(self, obs) -> torch.Tensor:
        flat_action = self.policy.predict(obs, deterministic=False)
        action_shape = self.env.single_action_space.shape
        return flat_action.reshape(flat_action.shape[0], self.horizon_length, *action_shape)

    def _on_env_reset(self, obs) -> None:
        super()._on_env_reset(obs)
        self._rollout_chunks = ActionChunkQueue(self.horizon_length)

    def _rollout_action(
        self, obs, learning_has_started: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]:
        if not learning_has_started and self._active_initial_training_phase() is None:
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
