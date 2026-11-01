"""``ModelBasedStepMixin``: the three-hook gradient step shared by the online
(``ModelBasedAlgorithm``) and offline (``OfflineModelBasedAlgorithm``)
model-based hosts.

One gradient step is: sample a batch (``_sample_train_batch``), then
``_update_model(batch) -> (metrics, posterior)``, then
``_update_actor_critic(batch, posterior) -> metrics``, then
``_update_targets()``. Subclasses override the three hooks (and, when the
batch does not come straight from ``self.replay_buffer``, the sampler) --
never ``train()``/``_gradient_step()``. ``posterior`` is the LIVE
(graph-attached) ``State`` returned by ``WorldModel.model_loss``;
``_update_actor_critic`` detaches it itself if its actor update must not
backprop into an already-stepped model (TD-MPC2's case). A hook may be a
no-op for algorithms that do not need it.

The host class supplies ``self.policy`` (with a ``world_model``),
``self.replay_buffer``, ``self.batch_size`` and ``self._global_update``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rl_garden.world_models.base import State, WorldModel


class ModelBasedStepMixin(ABC):
    @property
    def world_model(self) -> WorldModel:
        return self.policy.world_model

    @abstractmethod
    def _update_model(self, batch: Any) -> tuple[dict[str, float], State]:
        """Trains the world model (and, for TD-MPC2, the critic -- see
        ``TDMPC2Core._update_model``) on ``batch``. Returns ``(metrics,
        posterior)`` where ``posterior`` is the LIVE ``State``
        ``_update_actor_critic`` reads."""

    @abstractmethod
    def _update_actor_critic(self, batch: Any, posterior: State) -> dict[str, float]:
        """Trains the actor (and, for algorithms that don't fold the critic
        into ``_update_model``, the critic) from ``posterior``."""

    @abstractmethod
    def _update_targets(self) -> None:
        """Polyak/hard target-network update(s), run once per gradient step
        after ``_update_model``/``_update_actor_critic``."""

    def _sample_train_batch(self) -> Any:
        return self.replay_buffer.sample(self.batch_size)

    def _gradient_step(self) -> dict[str, float]:
        self._global_update += 1
        batch = self._sample_train_batch()
        model_metrics, posterior = self._update_model(batch)
        actor_critic_metrics = self._update_actor_critic(batch, posterior)
        self._update_targets()
        return {**model_metrics, **actor_critic_metrics}

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        del compute_info  # every step's info dict is floats only -- cheap, always computed.
        info: dict[str, float] = {}
        for _ in range(gradient_steps):
            info = self._gradient_step()
        return info
