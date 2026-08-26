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

from typing import Any, Optional

import torch
from gymnasium import spaces

from rl_garden.algorithms.bc import BC
from rl_garden.buffers.dict_buffer import DictReplayBuffer
from rl_garden.buffers.demo_intervention import DemoInterventionMixin
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
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
        self._init_prior_data_params()
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
        obs_space = self.env.single_observation_space
        if isinstance(obs_space, spaces.Dict):
            return DictReplayBuffer(
                observation_space=obs_space,
                action_space=self.env.single_action_space,
                num_envs=self.num_envs,
                buffer_size=buffer_size,
                storage_device=self.buffer_device,
                sample_device=self.device,
            )
        return TensorReplayBuffer(
            observation_space=obs_space,
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

    # --- rollout + aggregate ---

    def collect_round(self, num_steps: int, beta: float) -> None:
        if self._rollout_obs is None:
            obs, _ = self.env.reset(seed=self.seed)
            self._rollout_obs = obs
        obs = self._rollout_obs
        self.policy.eval()
        for _ in range(num_steps):
            with torch.no_grad():
                expert_action = self.expert(obs)
                policy_action = self.policy.predict(obs, deterministic=True)
            use_expert = torch.rand(self.num_envs, device=expert_action.device) < beta
            mask = use_expert.view(-1, *([1] * (expert_action.dim() - 1)))
            executed_action = torch.where(mask, expert_action, policy_action)

            next_obs, reward, terminated, truncated, _info = self.env.step(executed_action)
            done = (terminated | truncated).float()
            # Label is always the expert's action regardless of which action
            # was actually executed -- this is what makes it DAgger rather
            # than plain on-policy BC (Ross et al. 2011; confirmed against
            # imitation's InteractiveTrajectoryCollector semantics).
            self.add_demo_transition(obs, next_obs, expert_action, reward, done)
            obs = next_obs
        self._rollout_obs = obs

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
            beta = self.beta_schedule(self._round_num)
            self.collect_round(self.rollout_steps_per_round, beta)
            self._global_step += self.rollout_steps_per_round * self.num_envs
            self.train(self.gradient_steps_per_round)
            self._round_num += 1
        return self
