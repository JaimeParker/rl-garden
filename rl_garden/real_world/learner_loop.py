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
_EVAL_STEP_METRIC = "steps/eval_received_transitions"


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
        monitor_interval: float = 5.0,
        eval_freq: Optional[int] = None,
    ) -> None:
        self.agent = agent
        self.train_freq = train_freq
        self.publish_freq = publish_freq
        self.idle_poll_interval = idle_poll_interval
        self.monitor_interval = monitor_interval
        self.eval_freq = (
            int(getattr(agent, "eval_freq", 0)) if eval_freq is None else int(eval_freq)
        )

        self._lock = threading.Lock()
        self._received = 0
        self._transition_log_period = 1000
        self._last_checkpoint_update = 0
        self._last_monitor_ts: Optional[float] = None
        self._last_monitor_received = 0
        self._last_monitor_update = 0
        self._last_train_log_step = 0
        self._last_eval_step = 0
        self._server = LearnerSyncServer(host, port, on_transition=self._on_transition)
        self._define_wandb_step_metrics()

    @property
    def received_transitions(self) -> int:
        with self._lock:
            return self._received

    def _on_transition(self, transition: dict[str, Any]) -> None:
        transition = dict(transition)
        episode_metrics = transition.pop("episode_metrics", None)
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
            self._log_received_transition_count()
            received = self._received
        self._log_episode_metrics(episode_metrics, received)

    def _log_received_transition_count(self) -> None:
        if (
            self._transition_log_period > 0
            and self._received % self._transition_log_period == 0
        ):
            print(
                f"[sync] learner received_transitions={self._received}",
                flush=True,
            )

    def _refresh_offline_data(self) -> None:
        """Hook for methods that need to periodically re-read a growing
        on-disk dataset (e.g. HIL-SERL's demo/correction data). No-op by
        default -- SERL doesn't need it."""

    def _should_log_train_info(self, step: int) -> bool:
        if getattr(self.agent, "logger", None) is None:
            return False
        log_freq = int(getattr(self.agent, "log_freq", 0))
        if log_freq <= 0:
            return False
        return self._last_train_log_step // log_freq < step // log_freq

    def _log_train_info(self, info: dict[str, float], step: int) -> None:
        if not info:
            return
        logger = getattr(self.agent, "logger", None)
        if logger is None:
            return
        logger.log_metrics(info, step)

    def _log_episode_metrics(self, metrics: Any, step: int) -> None:
        if not isinstance(metrics, dict):
            return
        logger = getattr(self.agent, "logger", None)
        if logger is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.add_scalar(f"train/{key}", float(value), step)

    def _train_step(self, compute_info: Optional[bool] = None) -> dict[str, float]:
        gradient_steps = max(1, int(self.train_freq * self.agent.utd))
        step = self.received_transitions
        if compute_info is None:
            compute_info = self._should_log_train_info(step)
        with self._lock:
            info = self.agent.train(gradient_steps, compute_info=compute_info)
        if compute_info:
            self._log_train_info(info, step)
            self._last_train_log_step = step
        self._maybe_evaluate(step)
        self._maybe_save_periodic_checkpoint()
        return info

    def _maybe_evaluate(self, step: int) -> None:
        if self.eval_freq <= 0:
            return
        if getattr(self.agent, "eval_env", None) is None:
            return
        if self._last_eval_step // self.eval_freq >= step // self.eval_freq:
            return

        start = time.perf_counter()
        metrics = self.agent._evaluate()
        eval_time = time.perf_counter() - start
        self._last_eval_step = step

        logger = getattr(self.agent, "logger", None)
        if logger is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.add_scalar(
                        f"eval/{key}",
                        float(value),
                        step,
                        step_metric=_EVAL_STEP_METRIC,
                    )
            logger.add_scalar(
                "time/eval_time",
                eval_time,
                step,
                step_metric=_EVAL_STEP_METRIC,
            )

        if bool(getattr(self.agent, "std_log", False)):
            eval_return = metrics.get("return")
            eval_success = metrics.get("success_at_end", metrics.get("success_once"))
            print(
                "[eval] "
                f"step={step} "
                f"return={self._fmt_metric(eval_return)} "
                f"success_at_end={self._fmt_metric(eval_success)}",
                flush=True,
            )

    @staticmethod
    def _fmt_metric(value: Any) -> str:
        if value is None:
            return "nan"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "nan"

    def _define_wandb_step_metrics(self) -> None:
        logger = getattr(self.agent, "logger", None)
        define_metric = getattr(logger, "define_metric", None)
        if define_metric is None:
            return
        define_metric(_EVAL_STEP_METRIC)
        define_metric("eval/*", step_metric=_EVAL_STEP_METRIC)
        define_metric("time/eval_time", step_metric=_EVAL_STEP_METRIC)

    def _maybe_save_periodic_checkpoint(self) -> None:
        agent = self.agent
        if agent.checkpoint_dir is None or agent.checkpoint_freq <= 0:
            return
        update = agent.global_update
        if (
            update // agent.checkpoint_freq
            <= self._last_checkpoint_update // agent.checkpoint_freq
        ):
            return
        agent.save(
            Path(agent.checkpoint_dir) / f"checkpoint_{update}.pt",
            include_replay_buffer=agent.save_replay_buffer,
        )
        self._last_checkpoint_update = update

    def _maybe_log_monitor(self) -> None:
        if self.monitor_interval <= 0:
            return
        now = time.monotonic()
        if self._last_monitor_ts is None:
            self._last_monitor_ts = now
            self._last_monitor_received = self.received_transitions
            self._last_monitor_update = self.agent.global_update
            return
        elapsed = now - self._last_monitor_ts
        if elapsed < self.monitor_interval:
            return
        with self._lock:
            received = self._received
            replay_len = len(self.agent.replay_buffer)
        update = self.agent.global_update
        server_stats = self._server.connection_stats()
        received_rate = (received - self._last_monitor_received) / max(elapsed, 1e-6)
        update_rate = (update - self._last_monitor_update) / max(elapsed, 1e-6)
        print(
            "[sync] learner link "
            f"received={received} received_rate={received_rate:.2f}/s "
            f"replay_len={replay_len} "
            f"global_update={update} update_rate={update_rate:.2f}/s "
            f"published_policy_version={self._server.published_version} "
            f"waiting_for_learning_starts={received < self.agent.learning_starts} "
            f"http_transition_posts={server_stats['transition_posts']} "
            f"http_policy_gets={server_stats['policy_param_gets']} "
            f"last_http_transition={self._format_age(server_stats['last_transition_post_ts'], now)} "
            f"last_http_policy_get={self._format_age(server_stats['last_policy_param_get_ts'], now)}",
            flush=True,
        )
        self._last_monitor_ts = now
        self._last_monitor_received = received
        self._last_monitor_update = update

    @staticmethod
    def _format_age(timestamp: Optional[float], now: float) -> str:
        if timestamp is None:
            return "never"
        return f"{max(0.0, now - timestamp):.1f}s_ago"

    def run(self, total_transitions: Optional[int] = None) -> None:
        """Runs until ``total_transitions`` have been received (or forever,
        if ``None``, until :meth:`stop` is called from another thread)."""
        self._server.start()
        self._stop = False
        try:
            update = 0
            while not self._stop:
                self._maybe_log_monitor()
                if (
                    total_transitions is not None
                    and self.received_transitions >= total_transitions
                ):
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
