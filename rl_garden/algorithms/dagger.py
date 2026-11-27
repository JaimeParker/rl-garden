"""DAgger (Ross et al. 2011): interactive imitation learning via a
beta-mixed scripted-expert rollout that always labels the aggregated dataset
with the expert's action, alternated with BC-style retraining.

Composes two existing, unmodified pieces rather than introducing new
machinery: ``BC`` (``algorithms/bc.py``, policy/loss/vision-dispatch
machinery, reused as-is) and ``DemoInterventionMixin``
(``buffers/demo_intervention.py``, the incrementally-growing buffer HIL-SERL
already uses for human-intervention data, reused here for expert-labeled
data instead). See ``.agents/local/imitation-learning-expansion-dagger-notes.md``
for the design rationale and comparison against ``HumanCompatibleAI/imitation``'s
own DAgger implementation.

``DemoInterventionMixin`` must come first in the MRO (``DAgger(DemoInterventionMixin,
BC)``, matching ``RLPDHybrid(DemoInterventionMixin, RLPD)``'s existing
precedent): ``BC._sample_train_batch(self)`` (no ``batch_size`` arg) is the
only such signature in this codebase -- every other algorithm, including
``PriorDataReplayMixin._sample_train_batch(self, batch_size)`` (inherited via
this mixin), takes the batch size explicitly. With ``BC`` first, its no-arg
override would shadow the mixin's mixing logic entirely and this would
silently degrade into "always sample the permanently-empty online buffer."
``train()`` below is therefore a near-copy of ``BC.train()`` that calls
``self._sample_train_batch(self.batch_size)`` explicitly, matching the
convention every non-BC algorithm in this codebase already uses.

``PriorDataReplayMixin._build_prior_data_buffer`` hardcodes ``num_envs=1``
(built for a static single-stream offline dataset) and, in its n-step
branches, reads ``self.nstep`` (only ever set by the SAC family -- ``BC``
has no such attribute). DAgger overrides it below to use
``num_envs=self.num_envs`` (rollout steps a real, possibly multi-env VecEnv
every round) and drops the n-step branches entirely -- DAgger has no n-step
return machinery, so ``self.nstep`` is never read.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Optional

import torch

from rl_garden.algorithms.bc import BC
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.buffers.demo_intervention import DemoInterventionMixin
from rl_garden.common.eval_metrics import masked_episode_metrics
from rl_garden.common.scripted_expert import ScriptedExpert


class DAgger(DemoInterventionMixin, BC):
    """DAgger: rollout the current policy under beta-mixed expert control,
    always aggregate the *expert's* action as the training label, and
    periodically retrain via BC on the growing aggregated dataset."""

    _compatible_checkpoint_algorithms = ("DAgger",)

    def __init__(
        self,
        env: Any,
        expert: ScriptedExpert,
        eval_env: Optional[Any] = None,
        *,
        demo_buffer_size: int = 100_000,
        beta_rounds: int = 15,
        rollout_steps_per_round: int = 1_000,
        gradient_steps_per_round: int = 100,
        buffer_size: int = 1,
        **bc_kwargs: Any,
    ) -> None:
        if rollout_steps_per_round <= 0:
            raise ValueError(
                f"rollout_steps_per_round must be positive, got {rollout_steps_per_round}."
            )
        if gradient_steps_per_round <= 0:
            raise ValueError(
                f"gradient_steps_per_round must be positive, got {gradient_steps_per_round}."
            )
        # BC builds its own self.replay_buffer unconditionally in
        # _setup_model() but DAgger never writes to it -- all real data goes
        # through self.offline_replay_buffer (below) instead. Kept tiny
        # rather than exposed to the caller, same reasoning DiffusionBC uses
        # for its own unused replay_buffer (buffer_size=1 there too).
        # ReplayBuffer (BC's buffer, always Dict now) validates
        # per_env_buffer_size = buffer_size // num_envs >= 1 at construction,
        # so floor buffer_size at num_envs even though this buffer is never
        # actually added to.
        self._init_prior_data_params()
        buffer_size = max(buffer_size, getattr(env, "num_envs", 1))
        super().__init__(env=env, eval_env=eval_env, buffer_size=buffer_size, **bc_kwargs)

        self.expert = expert
        self.beta_rounds = beta_rounds
        self.rollout_steps_per_round = rollout_steps_per_round
        self.gradient_steps_per_round = gradient_steps_per_round
        self._round_num = 0
        self._rollout_obs = None
        self.init_demo_buffer(demo_buffer_size, demo_data_ratio=1.0)

    # --- growing buffer sizing (see module docstring) ---

    def _build_prior_data_buffer(self, buffer_size: int):
        # obs_space is always Dict (boundary normalization is unconditional).
        return ReplayBuffer(
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )

    # --- beta schedule (Ross et al. 2011, linear decay) ---

    def beta_schedule(self, round_num: int) -> float:
        if self.beta_rounds <= 0:
            return 0.0
        return max(0.0, 1.0 - round_num / self.beta_rounds)

    # --- checkpointing ---

    def _checkpoint_includes_replay_buffer(self) -> bool:
        return self.save_replay_buffer

    def _training_state_dict(self) -> dict[str, Any]:
        return {**super()._training_state_dict(), "round_num": self._round_num}

    def _load_training_state_dict(self, state: dict[str, Any]) -> None:
        super()._load_training_state_dict(state)
        self._round_num = int(state.get("round_num", 0))

    # --- rollout + aggregate ---

    def _to_storage_device(self, value):
        storage_device = self.offline_replay_buffer.storage_device
        if isinstance(value, dict):
            return {key: leaf.to(storage_device) for key, leaf in value.items()}
        return value.to(storage_device)

    def collect_round(self, num_steps: int, beta: float) -> dict[str, float]:
        """Roll out ``num_steps`` env steps under beta-mixed control and
        aggregate expert-labeled transitions. Returns the mean of every
        episode statistic that finished during the round."""
        if self._rollout_obs is None:
            obs, _ = self.env.reset(seed=self.seed)
            self._rollout_obs = obs
        obs = self._rollout_obs
        self.policy.eval()
        finished_episodes: dict[str, list[torch.Tensor]] = defaultdict(list)
        for _ in range(num_steps):
            with torch.no_grad():
                expert_action = self.expert(obs).to(self.device)
                policy_action = self.policy.predict(
                    self._obs_to_policy_device(obs), deterministic=True
                )
            use_expert = torch.rand(self.num_envs, device=self.device) < beta
            mask = use_expert.view(-1, *([1] * (expert_action.dim() - 1)))
            executed_action = torch.where(mask, expert_action, policy_action)

            next_obs, reward, terminated, truncated, infos = self.env.step(executed_action)
            for key, values in masked_episode_metrics(infos).items():
                finished_episodes[key].append(values.detach().float().reshape(-1))
            done = (terminated | truncated).float()
            # Label is always the expert's action regardless of which action
            # was actually executed -- this is what makes it DAgger rather
            # than plain on-policy BC (Ross et al. 2011; confirmed against
            # imitation's InteractiveTrajectoryCollector semantics).
            self.add_demo_transition(
                self._to_storage_device(obs),
                self._to_storage_device(next_obs),
                self._to_storage_device(expert_action),
                self._to_storage_device(reward),
                self._to_storage_device(done),
            )
            obs = next_obs
        self._rollout_obs = obs
        return {
            key: float(torch.cat(values).mean().item())
            for key, values in finished_episodes.items()
        }

    # --- training (see module docstring for why this can't be inherited) ---

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        if gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be positive, got {gradient_steps}.")
        metrics_sum: dict[str, float] = {}
        self.policy.train()
        for _ in range(gradient_steps):
            self._global_update += 1
            data = self._sample_train_batch(self.batch_size)

            self.actor_optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._compute_losses(data)
            loss.backward()
            self._clip_grad_norm()
            self.actor_optimizer.step()
            self._step_schedulers()

            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        return {key: value / gradient_steps for key, value in metrics_sum.items()}

    # --- outer loop ---

    def learn(self, total_timesteps: int) -> "DAgger":
        self._on_training_start(total_timesteps)
        while self._global_step < total_timesteps:
            previous_step = self._global_step
            beta = self.beta_schedule(self._round_num)
            rollout_start = time.perf_counter()
            episode_metrics = self.collect_round(self.rollout_steps_per_round, beta)
            rollout_time = time.perf_counter() - rollout_start
            self._global_step += self.rollout_steps_per_round * self.num_envs
            losses = self.train(self.gradient_steps_per_round)
            self._round_num += 1
            self._log_round(
                total_timesteps, beta, losses, episode_metrics, rollout_time
            )
            self._maybe_save_periodic_checkpoint(previous_step)
        self._maybe_save_final_checkpoint()
        return self

    def _log_round(
        self,
        total_timesteps: int,
        beta: float,
        losses: dict[str, float],
        episode_metrics: dict[str, float],
        rollout_time: float,
    ) -> None:
        demo_size = len(self.offline_replay_buffer)
        step = self._global_step
        if self.logger is not None:
            self._log_update_metrics(losses, step)
            self.logger.add_scalar("dagger/beta", beta, step)
            self.logger.add_scalar("dagger/demo_buffer_size", demo_size, step)
            self.logger.add_scalar("dagger/round", self._round_num, step)
            self.logger.add_scalar("time/rollout_time", rollout_time, step)
            for key, value in episode_metrics.items():
                self._log_rollout_metric(key, value, step)
        if self.std_log:
            episode_return = self._first_metric(episode_metrics, ("return",))
            success = self._first_metric(episode_metrics, ("success_at_end", "success_once"))
            loss = self._first_metric(losses, ("actor_loss", "loss"))
            print(
                "[train] "
                f"step={step}/{total_timesteps} round={self._round_num} "
                f"beta={beta:.3f} demo_size={demo_size} "
                f"actor_loss={self._fmt_metric(loss)} "
                f"return={self._fmt_metric(episode_return)} "
                f"success_at_end={self._fmt_metric(success)}",
                flush=True,
            )
