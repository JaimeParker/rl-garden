"""HIL-SERL's LearnerLoop -- this migration targets HIL-SERL's ``train_rlpd.py``
capability set (online RLPD + demo mixing + HITL + reward classifier), which
uses RLPD's existing static ``offline_dataset_path`` loading, not a
continuously growing on-disk dataset. So ``_refresh_offline_data()`` stays
the base class's no-op default here too; HG-DAgger's growing-dataset problem
is out of scope for this round (see docs/robot_infra_roadmap.md).

Overrides ``_on_transition`` to route transitions the way HIL-SERL's own
actor loop does (``3rd_party/hil-serl/examples/train_rlpd.py:178-193``):
*every* transition goes into the online replay buffer, and human-intervened
transitions are *additionally* copied into the growing demo buffer
(``DemoInterventionMixin``, see ``rl_garden/buffers/demo_intervention.py``)
-- the demo buffer is a duplicated subset of the online buffer, not a
disjoint partition of it. Also periodically pickle-snapshots both buffers to
disk for crash recovery -- ``LearnerLoop.run()`` never triggers the existing
``save_replay_buffer``/``include_replay_buffer`` checkpoint mechanism
(``rl_garden/algorithms/off_policy.py``), so this pkl snapshot is the only
buffer persistence path for real-world hil_serl training. Mirrors HIL-SERL's
own mechanism (``train_rlpd.py:140-233``): transitions accumulated since the
last snapshot are pickled to ``<checkpoint_dir>/buffer/transitions_{n}.pkl``
(every transition) and ``<checkpoint_dir>/demo_buffer/transitions_{n}.pkl``
(intervened only) every ``buffer_period`` received transitions, and both
directories are glob-reloaded on startup.

Also loads ``demo_dataset_paths`` (pre-collected demo transitions, HIL-SERL's
``--demo_path`` equivalent -- ``train_rlpd.py:458-466``) into the demo buffer
on every start, unconditionally, before the crash-recovery snapshot reload --
matching HIL-SERL's own always-reload-``demo_path`` behavior exactly. Uses
the same pkl transition-dict format as the buffer snapshots above (not
HIL-SERL's own numpy dict shape), so a previous run's
``demo_buffer/*.pkl`` snapshot files can be reused directly as seed data.
"""
from __future__ import annotations

import glob
import os
import pickle
from typing import Any, Sequence

import torch

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.real_world.learner_loop import _TRANSITION_TENSOR_KEYS, LearnerLoop


class HilSerlLearnerLoop(LearnerLoop):
    def __init__(
        self,
        agent: OffPolicyAlgorithm,
        host: str,
        port: int,
        checkpoint_dir: str,
        buffer_period: int = 1000,
        demo_dataset_paths: Sequence[str] = (),
        train_freq: int = 1,
        publish_freq: int = 100,
        idle_poll_interval: float = 0.1,
    ) -> None:
        super().__init__(agent, host, port, train_freq, publish_freq, idle_poll_interval)
        self._checkpoint_dir = checkpoint_dir
        self._buffer_period = buffer_period
        self._pending_online: list[dict[str, Any]] = []
        self._pending_demo: list[dict[str, Any]] = []
        self._step_since_snapshot = 0
        self._load_demo_datasets(demo_dataset_paths)
        self._reload_snapshots()

    def _load_demo_datasets(self, paths: Sequence[str]) -> None:
        # Not counted toward `_received`/`received_transitions` -- that
        # gates LearnerLoop.run()'s learning_starts wait, and HIL-SERL's own
        # analogous gate (`len(replay_buffer) < training_starts`) only ever
        # counts the online buffer, never `--demo_path` preload.
        device = self.agent.buffer_device
        for pattern in paths:
            for pkl_path in sorted(glob.glob(pattern)):
                with open(pkl_path, "rb") as f:
                    transitions = pickle.load(f)
                for transition in transitions:
                    self._add_transition(self.agent.add_demo_transition, transition, device)

    def _reload_snapshots(self) -> None:
        device = self.agent.buffer_device
        # "buffer" holds every transition ever received (mirrors HIL-SERL's
        # data_store); "demo_buffer" is the intervened subset, duplicated
        # into it, not a disjoint remainder -- so only count a transition as
        # received once, off the "buffer" directory, or reloading both would
        # double-count every intervened transition.
        sources = (
            (os.path.join(self._checkpoint_dir, "buffer"), self.agent.replay_buffer.add, True),
            (os.path.join(self._checkpoint_dir, "demo_buffer"), self.agent.add_demo_transition, False),
        )
        for directory, adder, count_received in sources:
            for pkl_path in sorted(glob.glob(os.path.join(directory, "*.pkl"))):
                with open(pkl_path, "rb") as f:
                    transitions = pickle.load(f)
                for transition in transitions:
                    self._add_transition(adder, transition, device)
                    if count_received:
                        self._received += 1

    @staticmethod
    def _add_transition(adder, transition: dict[str, Any], device) -> None:
        tensors = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in transition.items()
            if k in _TRANSITION_TENSOR_KEYS
        }
        extra = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in transition.items()
            if k not in _TRANSITION_TENSOR_KEYS
        }
        adder(tensors["obs"], tensors["next_obs"], tensors["action"], tensors["reward"], tensors["done"], **extra)

    def _on_transition(self, transition: dict[str, Any]) -> None:
        intervened = bool(transition.pop("intervened", False))
        device = self.agent.buffer_device
        with self._lock:
            self._add_transition(self.agent.replay_buffer.add, transition, device)
            self._pending_online.append(transition)
            if intervened:
                self._add_transition(self.agent.add_demo_transition, transition, device)
                self._pending_demo.append(transition)
            self._received += 1
            self._step_since_snapshot += 1
            if self._step_since_snapshot >= self._buffer_period:
                self._snapshot()

    def _snapshot(self) -> None:
        buffer_dir = os.path.join(self._checkpoint_dir, "buffer")
        demo_dir = os.path.join(self._checkpoint_dir, "demo_buffer")
        os.makedirs(buffer_dir, exist_ok=True)
        os.makedirs(demo_dir, exist_ok=True)
        step = self._received
        with open(os.path.join(buffer_dir, f"transitions_{step}.pkl"), "wb") as f:
            pickle.dump(self._pending_online, f)
        with open(os.path.join(demo_dir, f"transitions_{step}.pkl"), "wb") as f:
            pickle.dump(self._pending_demo, f)
        self._pending_online = []
        self._pending_demo = []
        self._step_since_snapshot = 0
