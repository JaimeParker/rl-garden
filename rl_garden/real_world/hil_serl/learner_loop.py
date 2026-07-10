"""HIL-SERL's LearnerLoop -- this migration targets HIL-SERL's ``train_rlpd.py``
capability set (online RLPD + demo mixing + HITL + reward classifier), which
uses RLPD's existing static ``offline_dataset_path`` loading, not a
continuously growing on-disk dataset. So ``_refresh_offline_data()`` stays
the base class's no-op default here too; HG-DAgger's growing-dataset problem
is out of scope for this round (see docs/robot_infra_roadmap.md).

Overrides ``_on_transition`` to route human-intervened transitions into the
growing demo buffer (``DemoInterventionMixin``, see
``rl_garden/buffers/demo_intervention.py``) instead of the plain online
replay buffer, and to periodically pickle-snapshot both buffers to disk for
crash recovery -- ``LearnerLoop.run()`` never triggers the existing
``save_replay_buffer``/``include_replay_buffer`` checkpoint mechanism
(``rl_garden/algorithms/off_policy.py``), so this pkl snapshot is the only
buffer persistence path for real-world hil_serl training. Mirrors HIL-SERL's
own mechanism (``3rd_party/hil-serl/examples/train_rlpd.py:140-233``):
transitions accumulated since the last snapshot are pickled to
``<checkpoint_dir>/buffer/transitions_{n}.pkl`` and
``<checkpoint_dir>/demo_buffer/transitions_{n}.pkl`` every ``buffer_period``
received transitions, and both directories are glob-reloaded on startup.
"""
from __future__ import annotations

import glob
import os
import pickle
from typing import Any

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
        self._reload_snapshots()

    def _reload_snapshots(self) -> None:
        device = self.agent.buffer_device
        sources = (
            (os.path.join(self._checkpoint_dir, "buffer"), self.agent.replay_buffer.add),
            (os.path.join(self._checkpoint_dir, "demo_buffer"), self.agent.add_demo_transition),
        )
        for directory, adder in sources:
            for pkl_path in sorted(glob.glob(os.path.join(directory, "*.pkl"))):
                with open(pkl_path, "rb") as f:
                    transitions = pickle.load(f)
                for transition in transitions:
                    self._add_transition(adder, transition, device)
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
        adder = self.agent.add_demo_transition if intervened else self.agent.replay_buffer.add
        with self._lock:
            self._add_transition(adder, transition, device)
            self._received += 1
            (self._pending_demo if intervened else self._pending_online).append(transition)
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
