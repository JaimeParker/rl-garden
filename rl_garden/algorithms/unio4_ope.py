"""UniO4OPE: Uni-O4 with dynamics-model OPE (off-policy evaluation) gating
-- a faithful-reproduction variant of ``UniO4`` (Milestone A of
``~/.claude/plans/git-diff-diff-haiku-subagent-cuddly-glade.md``).

Ported from ``3rd_party/Uni-O4/{main.py:280-330, abppo.py:314-322,
dynamics_eval.py::rollout}``, verified against source directly -- see the
approved plan for the specific numbers this port pins (no ``penalty_coef``
wired through by upstream, ``rollout_length~=1000``,
``rollout_batch_size=512``, gating cadence ``eval_step=100`` independent of
the real-eval cadence ``eval_freq``).

``UniO4``'s own ``_log_eval_metrics`` gates ``old_actors`` sync on real-env
eval return -- a disclosed, working, but methodologically different choice
from upstream, which gates exclusively on a dynamics-model OPE estimate
(its own real-env eval, ``off_evaluate``, is monitoring-only, never gates).
``UniO4OPE`` keeps real-env eval running for monitoring (same as upstream's
``off_evaluate``, and unmodified in ``UniO4`` itself) but overrides
``_maybe_sync_old_actors_from_eval`` (extracted in ``unio4.py`` as a pure,
behavior-preserving refactor -- see that file) as a no-op: sync now comes
exclusively from ``_ope_gating_check``, called periodically from ``train()``
once Phase IMPROVE has started.

The dynamics ensemble is trained lazily, once, on the first Phase-IMPROVE
``train()`` call rather than eagerly in ``_setup_model()`` -- at
``_setup_model()`` time (during ``UniO4.__init__``) the offline dataset
hasn't been loaded into ``self.replay_buffer`` yet (``rl_garden/training/
offline/_runner.py::_run_offline`` builds the agent, buffer empty, *then*
calls ``load_offline_dataset``); by Phase IMPROVE, Phase CRITIC and Phase
BC_ENSEMBLE have both already consumed that data, so it is unquestionably
present. This needed no changes to the shared offline runner.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch

from rl_garden.algorithms.unio4 import UniO4
from rl_garden.common.optim import make_optimizer
from rl_garden.models.dynamics import EnsembleDynamicsModel, get_termination_fn, rollout_q_mean, train_ensemble


class UniO4OPE(UniO4):
    _compatible_checkpoint_algorithms = ("UniO4OPE",)

    def __init__(
        self,
        env: Any,
        *,
        task: str,
        dynamics_hidden_dims: Sequence[int] = (200, 200, 200, 200),
        dynamics_n_ensemble: int = 7,
        dynamics_n_elites: int = 5,
        dynamics_lr: float = 1e-3,
        dynamics_weight_decay: Sequence[float] = (2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4),
        dynamics_max_epochs_since_update: int = 5,
        dynamics_max_epochs: Optional[int] = None,
        dynamics_batch_size: int = 256,
        dynamics_holdout_ratio: float = 0.2,
        ope_rollout_length: int = 1000,
        ope_rollout_batch_size: int = 512,
        ope_gating_freq: int = 100,
        **unio4_kwargs: Any,
    ) -> None:
        """``task``: env-family string (e.g. ``"halfcheetah-medium-v2"``)
        used only to pick a ``termination_fn`` -- ``OfflineEnvSpec`` carries
        no env id, so this can't be inferred from ``env``."""
        self._termination_fn = get_termination_fn(task)

        self.dynamics_hidden_dims = list(dynamics_hidden_dims)
        self.dynamics_n_ensemble = dynamics_n_ensemble
        self.dynamics_n_elites = dynamics_n_elites
        self.dynamics_lr = dynamics_lr
        self.dynamics_weight_decay = list(dynamics_weight_decay)
        self.dynamics_max_epochs_since_update = dynamics_max_epochs_since_update
        self.dynamics_max_epochs = dynamics_max_epochs
        self.dynamics_batch_size = dynamics_batch_size
        self.dynamics_holdout_ratio = dynamics_holdout_ratio
        self.ope_rollout_length = ope_rollout_length
        self.ope_rollout_batch_size = ope_rollout_batch_size
        self.ope_gating_freq = ope_gating_freq

        super().__init__(env, **unio4_kwargs)

        self._dynamics_trained = False
        self._last_ope_check = 0
        self._best_ope_score = [float("-inf")] * self.num_policies
        self.dynamics_input_mean: Optional[torch.Tensor] = None
        self.dynamics_input_std: Optional[torch.Tensor] = None

    def _setup_model(self) -> None:
        super()._setup_model()
        obs_dim = int(np.prod(self.env.single_observation_space.shape))
        action_dim = int(np.prod(self.env.single_action_space.shape))
        self.dynamics_model = EnsembleDynamicsModel(
            obs_dim,
            action_dim,
            self.dynamics_hidden_dims,
            num_ensemble=self.dynamics_n_ensemble,
            num_elites=self.dynamics_n_elites,
            weight_decays=self.dynamics_weight_decay,
        ).to(self.device)
        self.dynamics_optimizer = make_optimizer(
            list(self.dynamics_model.parameters()), lr=self.dynamics_lr
        )

    # --- real-eval sync neutralized, monitoring preserved (see docstring) ---

    def _maybe_sync_old_actors_from_eval(
        self, metrics: dict[str, float], step: int
    ) -> None:
        return

    # --- dynamics model + OPE gating ---

    def _valid_buffer_rows(self) -> int:
        buf = self.replay_buffer
        return buf.per_env_buffer_size if buf.full else buf.pos

    def _train_dynamics_model(self) -> None:
        buf = self.replay_buffer
        valid = self._valid_buffer_rows()
        obs = buf.obs[:valid].reshape(-1, buf.obs.shape[-1]).to(self.device)
        next_obs = buf.next_obs[:valid].reshape(-1, buf.next_obs.shape[-1]).to(self.device)
        actions = buf.actions[:valid].reshape(-1, buf.actions.shape[-1]).to(self.device)
        rewards = buf.rewards[:valid].reshape(-1).to(self.device)

        metrics, input_mean, input_std = train_ensemble(
            self.dynamics_model,
            self.dynamics_optimizer,
            obs,
            actions,
            next_obs,
            rewards,
            max_epochs_since_update=self.dynamics_max_epochs_since_update,
            max_epochs=self.dynamics_max_epochs,
            batch_size=self.dynamics_batch_size,
            holdout_ratio=self.dynamics_holdout_ratio,
        )
        self.dynamics_input_mean = input_mean
        self.dynamics_input_std = input_std
        self._dynamics_trained = True
        if self.logger is not None:
            for key, value in metrics.items():
                self.logger.add_scalar(f"dynamics/{key}", value, self._global_update)

    def _ope_gating_check(self) -> dict[str, float]:
        buf = self.replay_buffer
        valid = self._valid_buffer_rows()
        obs_flat = buf.obs[:valid].reshape(-1, buf.obs.shape[-1])
        idx = torch.randint(
            0, obs_flat.shape[0], (self.ope_rollout_batch_size,), device=obs_flat.device
        )
        init_obs = obs_flat[idx].to(self.device)

        metrics: dict[str, float] = {}
        for i in range(self.num_policies):
            score = rollout_q_mean(
                self.actors[i],
                self.q_net,
                self.dynamics_model,
                self._termination_fn,
                init_obs,
                self.ope_rollout_length,
                input_mean=self.dynamics_input_mean,
                input_std=self.dynamics_input_std,
            )
            metrics[f"ope_q_mean_{i}"] = score
            if score > self._best_ope_score[i]:
                self._best_ope_score[i] = score
                self.old_actors[i].load_state_dict(self.actors[i].state_dict())
                metrics[f"ope_old_policy_synced_{i}"] = 1.0
        if self.logger is not None:
            for key, value in metrics.items():
                self.logger.add_scalar(f"ope/{key}", value, self._global_update)
        return metrics

    # --- training loop ---

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        metrics = super().train(gradient_steps, compute_info=compute_info)
        if self._phase_step >= self._improve_phase_start:
            if not self._dynamics_trained:
                self._train_dynamics_model()
            elif self._global_update - self._last_ope_check >= self.ope_gating_freq:
                self._ope_gating_check()
                self._last_ope_check = self._global_update
        return metrics

    # --- checkpointing ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return (*super()._optimizer_names(), "dynamics_optimizer")

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "dynamics_hidden_dims": self.dynamics_hidden_dims,
            "dynamics_n_ensemble": self.dynamics_n_ensemble,
            "dynamics_n_elites": self.dynamics_n_elites,
            "dynamics_weight_decay": self.dynamics_weight_decay,
            "ope_rollout_length": self.ope_rollout_length,
            "ope_rollout_batch_size": self.ope_rollout_batch_size,
            "ope_gating_freq": self.ope_gating_freq,
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            **super()._extra_checkpoint_state(),
            "dynamics_model": self.dynamics_model.state_dict(),
            "dynamics_input_mean": self.dynamics_input_mean,
            "dynamics_input_std": self.dynamics_input_std,
            "dynamics_trained": self._dynamics_trained,
            "best_ope_score": self._best_ope_score,
            "last_ope_check": self._last_ope_check,
        }

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_checkpoint_state(state)
        self.dynamics_model.load_state_dict(state["dynamics_model"])
        self.dynamics_input_mean = state.get("dynamics_input_mean")
        self.dynamics_input_std = state.get("dynamics_input_std")
        self._dynamics_trained = state.get("dynamics_trained", False)
        self._best_ope_score = state.get("best_ope_score", self._best_ope_score)
        self._last_ope_check = state.get("last_ope_check", 0)
