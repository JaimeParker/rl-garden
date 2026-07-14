"""LearnerLoop: drives an existing ``OffPolicyAlgorithm`` from transitions
received over the network instead of ``OffPolicyAlgorithm.learn()``'s
single-process rollout+update loop (which assumes it owns the env directly
and isn't a fit for a real robot stepped by a separate actor process).

Algorithm-agnostic by construction: only touches the public surface any
``OffPolicyAlgorithm`` subclass (``SAC``, ``RLPD``, ``TD3``, ...) already
exposes -- ``replay_buffer.add(...)``, ``train(gradient_steps)``,
``policy.state_dict()``, and the checkpoint-related attributes/methods
(``checkpoint_dir``, ``checkpoint_freq``, ``save_replay_buffer``,
``save_final_checkpoint``, ``global_update``, ``save(...)``) -- so no
base-class change beyond the read-only ``global_update`` property was needed
to build this.

Also periodically saves the full agent state (weights + optimizer), since
``OffPolicyAlgorithm.learn()``'s own periodic-checkpoint machinery
(``_maybe_save_periodic_checkpoint``/``_save_checkpoint``,
``rl_garden/algorithms/off_policy.py``) is only ever called from within
``learn()``'s own rollout loop, which real-world training never runs.
Without this, a learner crash loses all trained weights/optimizer state --
only the replay/demo buffers survive (see ``HilSerlLearnerLoop``'s pkl
snapshots). Cadence mirrors HIL-SERL's own ``learner()``
(``3rd_party/hil-serl/examples/train_rlpd.py:314-355``): keyed on gradient
update count, not received-transition count. Resuming weights on startup
reuses the existing ``--load_checkpoint`` flag
(``build_rlpd``/``build_rlpd_hybrid`` already call ``agent.load(...)`` before
handing the agent to this loop) -- no new resume logic needed here.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

import torch

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.real_world.sync import LearnerSyncServer

_TRANSITION_TENSOR_KEYS = ("obs", "next_obs", "action", "reward", "done")


class LearnerLoop:
    """Owns a full algorithm instance and trains it from actor-supplied data.

    ``agent.replay_buffer.add()`` (invoked from the HTTP server's request
    thread on every received transition) and ``agent.train()`` (invoked from
    :meth:`run`'s loop) both touch the replay buffer, so both are serialized
    under one lock -- mirrors the thread-safety SERL gets from its own
    ``MemoryEfficientReplayBufferDataStore``.
    """

    def __init__(
        self,
        agent: OffPolicyAlgorithm,
        host: str,
        port: int,
        train_freq: int = 1,
        publish_freq: int = 100,
        idle_poll_interval: float = 0.1,
    ) -> None:
        self.agent = agent
        self.train_freq = train_freq
        self.publish_freq = publish_freq
        self.idle_poll_interval = idle_poll_interval

        self._lock = threading.Lock()
        self._received = 0
        self._last_checkpoint_update = 0
        self._server = LearnerSyncServer(host, port, on_transition=self._on_transition)

    @property
    def received_transitions(self) -> int:
        with self._lock:
            return self._received

    def _on_transition(self, transition: dict[str, Any]) -> None:
        device = self.agent.buffer_device
        tensors = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in transition.items()
            if k in _TRANSITION_TENSOR_KEYS
        }
        extra = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in transition.items()
            if k not in _TRANSITION_TENSOR_KEYS
        }
        with self._lock:
            self.agent.replay_buffer.add(
                tensors["obs"],
                tensors["next_obs"],
                tensors["action"],
                tensors["reward"],
                tensors["done"],
                **extra,
            )
            self._received += 1

    def _refresh_offline_data(self) -> None:
        """Hook for methods that need to periodically re-read a growing
        on-disk dataset (e.g. HIL-SERL's demo/correction data). No-op by
        default -- SERL doesn't need it."""

    def _train_step(self, compute_info: bool = False) -> dict[str, float]:
        gradient_steps = max(1, int(self.train_freq * self.agent.utd))
        with self._lock:
            info = self.agent.train(gradient_steps, compute_info=compute_info)
        self._maybe_save_periodic_checkpoint()
        return info

    def _maybe_save_periodic_checkpoint(self) -> None:
        agent = self.agent
        if agent.checkpoint_dir is None or agent.checkpoint_freq <= 0:
            return
        update = agent.global_update
        if update // agent.checkpoint_freq <= self._last_checkpoint_update // agent.checkpoint_freq:
            return
        agent.save(
            Path(agent.checkpoint_dir) / f"checkpoint_{update}.pt",
            include_replay_buffer=agent.save_replay_buffer,
        )
        self._last_checkpoint_update = update

    def run(self, total_transitions: Optional[int] = None) -> None:
        """Runs until ``total_transitions`` have been received (or forever,
        if ``None``, until :meth:`stop` is called from another thread)."""
        self._server.start()
        self._stop = False
        try:
            update = 0
            while not self._stop:
                if total_transitions is not None and self.received_transitions >= total_transitions:
                    break
                if self.received_transitions < self.agent.learning_starts:
                    time.sleep(self.idle_poll_interval)
                    continue
                self._refresh_offline_data()
                self._train_step()
                update += 1
                if update % self.publish_freq == 0:
                    self._server.publish_params(self.agent.policy.state_dict())
        finally:
            self._server.stop()
            agent = self.agent
            if agent.checkpoint_dir is not None and agent.save_final_checkpoint:
                agent.save(
                    Path(agent.checkpoint_dir) / "final.pt",
                    include_replay_buffer=agent.save_replay_buffer,
                )

    def stop(self) -> None:
        self._stop = True
