"""Torch-native rollout buffers for on-policy algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from gymnasium import spaces

from rl_garden.buffers.replay_buffer import DictArray
from rl_garden.buffers.gae import compute_gae
from rl_garden.common.obs_utils import flatten_leading_dims, index_obs
from rl_garden.common.types import Obs


@dataclass
class RolloutBufferSample:
    obs: Obs
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_mean: Optional[torch.Tensor] = None
    old_log_std: Optional[torch.Tensor] = None
    decision_mask: torch.Tensor | None = None
    component_mask: torch.Tensor | None = None


class RolloutBuffer:
    """Fixed-size on-policy buffer with ``(num_steps, num_envs, ...)`` layout.

    ``dones[t]`` is the AFTER-step flag: the episode ended as a result of
    acting on ``obs[t]`` (``terminations | truncations`` of step ``t``), so
    step ``t``'s own GAE target is cut by ``dones[t]``.
    ``compute_returns_and_advantage``'s ``last_dones`` is the done after the
    last stored step, i.e. the same flag as ``dones[-1]``.

    ``store_decisions=True`` adds ``decisions`` ((T, N) bool): row ``t`` is
    true where the action stored at step ``t`` took effect in the env (the
    env's ``infos["decision"]``). A window whose every ``add()`` passed a
    ``decision`` uses SMDP GAE (see ``compute_gae``'s ``step_lambda``) and
    yields ``decision_mask`` in its samples; a window without any keeps the
    plain path.

    It also adds, for partially effective (multi-rate) action components,
    ``component_mask`` ((T, N) + action shape bool, the env's
    ``infos["decision_components"]``) and ``component_log_probs`` (same shape,
    float32, the rollout log-prob of every action component). A window whose
    every ``add()`` passed both stores, as each step's behaviour log-prob, the
    sum over the components that took effect (computed once in
    ``compute_returns_and_advantage``) and yields ``component_mask`` in its
    samples; its GAE uses the decisions (any component took effect) like a
    whole-step window.

    See ``docs/guides/decision-level-ppo.md`` for the decision rules.
    """

    def __init__(
        self,
        observation_space: spaces.Box | spaces.Dict,
        action_space: spaces.Box,
        num_steps: int,
        num_envs: int,
        device: torch.device | str = "cuda",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        store_dist_params: bool = False,
        store_decisions: bool = False,
    ) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if not isinstance(action_space, spaces.Box):
            raise TypeError("RolloutBuffer only supports Box action spaces.")
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.pos = 0
        self.full = False
        self.generator_ready = False

        shape = (num_steps, num_envs)
        if isinstance(observation_space, spaces.Dict):
            self.obs = DictArray(shape, observation_space, device=self.device)
        elif isinstance(observation_space, spaces.Box):
            self.obs = torch.zeros(
                shape + tuple(observation_space.shape),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise TypeError(
                "RolloutBuffer supports Box or Dict observations, got "
                f"{type(observation_space)}."
            )
        self.actions = torch.zeros(
            shape + tuple(action_space.shape), dtype=torch.float32, device=self.device
        )
        self.log_probs = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.final_values = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.store_dist_params = store_dist_params
        if store_dist_params:
            action_shape = shape + tuple(action_space.shape)
            self.means = torch.zeros(action_shape, dtype=torch.float32, device=self.device)
            self.log_stds = torch.zeros(action_shape, dtype=torch.float32, device=self.device)
        self.store_decisions = store_decisions
        # Host-side count of add() calls that passed a decision this window.
        self._decision_rows = 0
        # Host-side count of add() calls that passed a component mask.
        self._component_rows = 0
        if store_decisions:
            # Rows added without a decision count as decisions.
            self.decisions = torch.ones(shape, dtype=torch.bool, device=self.device)
            component_shape = shape + tuple(action_space.shape)
            self.component_mask = torch.ones(
                component_shape, dtype=torch.bool, device=self.device
            )
            self.component_log_probs = torch.zeros(
                component_shape, dtype=torch.float32, device=self.device
            )

    @property
    def buffer_size(self) -> int:
        return self.num_steps * self.num_envs

    @property
    def window_has_decisions(self) -> bool:
        return self.store_decisions and self._decision_rows > 0

    @property
    def window_has_components(self) -> bool:
        return self.store_decisions and self._component_rows > 0

    def reset(self) -> None:
        self.pos = 0
        self.full = False
        self.generator_ready = False
        self.final_values.zero_()
        self._decision_rows = 0
        self._component_rows = 0
        if self.store_decisions:
            self.decisions.fill_(True)
            self.component_mask.fill_(True)

    def add(
        self,
        obs: Obs,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        final_values: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
        log_std: torch.Tensor | None = None,
        decision: torch.Tensor | None = None,
        component_mask: torch.Tensor | None = None,
        component_log_probs: torch.Tensor | None = None,
    ) -> None:
        if self.pos >= self.num_steps:
            raise RuntimeError(
                "RolloutBuffer is full; call reset() before adding more."
            )
        if decision is not None and not self.store_decisions:
            raise ValueError(
                "RolloutBuffer.add() got a decision but the buffer was built with "
                "store_decisions=False."
            )
        if (component_mask is None) != (component_log_probs is None):
            raise ValueError(
                "RolloutBuffer.add() takes component_mask and component_log_probs "
                "together."
            )
        if component_mask is not None:
            if not self.store_decisions:
                raise ValueError(
                    "RolloutBuffer.add() got a component mask but the buffer was built "
                    "with store_decisions=False."
                )
            if decision is None:
                raise ValueError(
                    "RolloutBuffer.add() got a component mask without the decision it "
                    "implies (any component took effect)."
                )

        if isinstance(self.obs, DictArray):
            assert isinstance(obs, dict)
            self.obs[self.pos] = {
                key: value.to(self.device) for key, value in obs.items()
            }
        else:
            assert isinstance(obs, torch.Tensor)
            self.obs[self.pos] = obs.to(self.device)
        self.actions[self.pos] = actions.to(self.device)
        self.rewards[self.pos] = rewards.reshape(self.num_envs).to(self.device)
        self.dones[self.pos] = dones.reshape(self.num_envs).float().to(self.device)
        self.values[self.pos] = values.reshape(self.num_envs).to(self.device)
        self.log_probs[self.pos] = log_probs.reshape(self.num_envs).to(self.device)
        if final_values is not None:
            self.final_values[self.pos] = final_values.reshape(self.num_envs).to(
                self.device
            )
        if self.store_dist_params:
            assert mean is not None and log_std is not None
            self.means[self.pos] = mean.to(self.device)
            self.log_stds[self.pos] = log_std.to(self.device)
        if decision is not None:
            self.decisions[self.pos] = decision.reshape(self.num_envs).to(
                self.device, torch.bool
            )
            self._decision_rows += 1
        if component_mask is not None:
            component_shape = self.component_mask.shape[1:]
            self.component_mask[self.pos] = component_mask.reshape(component_shape).to(
                self.device, torch.bool
            )
            self.component_log_probs[self.pos] = component_log_probs.reshape(
                component_shape
            ).to(self.device)
            self._component_rows += 1

        self.pos += 1
        self.full = self.pos == self.num_steps
        self.generator_ready = False

    def compute_returns_and_advantage(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        *,
        finite_horizon_gae: bool = False,
    ) -> None:
        if self.window_has_decisions:
            if self._decision_rows != self.num_steps:
                raise ValueError(
                    f"{self._decision_rows} of {self.num_steps} rollout steps reported a "
                    "decision: the env must report infos['decision'] on every step of a "
                    "run or on none (episode boundary contract)."
                )
            if finite_horizon_gae:
                raise ValueError("finite_horizon_gae does not support decision steps.")
        if self.window_has_components:
            if self._component_rows != self.num_steps:
                raise ValueError(
                    f"{self._component_rows} of {self.num_steps} rollout steps reported "
                    "a component mask: the env must report "
                    "infos['decision_components'] on every step of a run or on none "
                    "(episode boundary contract)."
                )
            # Behaviour log-prob of a step: the components that took effect.
            # ``where`` (not a multiply) drops the other components exactly.
            self.log_probs.copy_(
                torch.where(self.component_mask, self.component_log_probs, 0.0)
                .reshape(self.num_steps, self.num_envs, -1)
                .sum(-1)
            )
        last_values = last_values.reshape(self.num_envs).to(self.device)
        last_dones = last_dones.reshape(self.num_envs).float().to(self.device)
        advantages = torch.zeros_like(self.rewards)

        if finite_horizon_gae:
            lam_coef_sum = torch.zeros(self.num_envs, device=self.device)
            reward_term_sum = torch.zeros(self.num_envs, device=self.device)
            value_term_sum = torch.zeros(self.num_envs, device=self.device)
            for step in reversed(range(self.num_steps)):
                if step == self.num_steps - 1:
                    next_not_done = 1.0 - last_dones
                    next_values = last_values
                else:
                    next_not_done = 1.0 - self.dones[step]
                    next_values = self.values[step + 1]
                real_next_values = next_not_done * next_values + self.final_values[step]
                lam_coef_sum = lam_coef_sum * next_not_done
                reward_term_sum = reward_term_sum * next_not_done
                value_term_sum = value_term_sum * next_not_done
                lam_coef_sum = 1.0 + self.gae_lambda * lam_coef_sum
                reward_term_sum = (
                    self.gae_lambda * self.gamma * reward_term_sum
                    + lam_coef_sum * self.rewards[step]
                )
                value_term_sum = (
                    self.gae_lambda * self.gamma * value_term_sum
                    + self.gamma * real_next_values
                )
                advantages[step] = (
                    reward_term_sum + value_term_sum
                ) / lam_coef_sum - self.values[step]
            self.advantages = advantages
            self.returns = self.advantages + self.values
        else:
            # compute_gae takes episode-start flags: starts[t + 1] = dones[t].
            # starts[0] is never read by compute_gae, so it stays zero.
            episode_starts = torch.zeros_like(self.dones)
            episode_starts[1:] = self.dones[:-1]
            if self.window_has_decisions:
                # lambda once per decision: inside a hold (next step is not a
                # decision) the advantage carries back undecayed.
                step_lambda = torch.full_like(self.rewards, self.gae_lambda)
                step_lambda[:-1].masked_fill_(~self.decisions[1:], 1.0)
                self.advantages, self.returns = compute_gae(
                    self.rewards,
                    episode_starts,
                    self.values,
                    self.final_values,
                    last_values,
                    last_dones,
                    self.gamma,
                    self.gae_lambda,
                    step_lambda=step_lambda,
                )
            else:
                self.advantages, self.returns = compute_gae(
                    self.rewards,
                    episode_starts,
                    self.values,
                    self.final_values,
                    last_values,
                    last_dones,
                    self.gamma,
                    self.gae_lambda,
                )

        self.generator_ready = False

    def get(self, batch_size: int | None = None) -> Iterator[RolloutBufferSample]:
        if not self.full:
            raise RuntimeError("RolloutBuffer must be full before sampling.")
        if batch_size is None:
            batch_size = self.buffer_size
        flat_obs = flatten_leading_dims(self.obs)
        flat_actions = self.actions.reshape((-1,) + self.actions.shape[2:])
        flat_values = self.values.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)
        flat_means = (
            self.means.reshape((-1,) + self.means.shape[2:])
            if self.store_dist_params
            else None
        )
        flat_log_stds = (
            self.log_stds.reshape((-1,) + self.log_stds.shape[2:])
            if self.store_dist_params
            else None
        )
        flat_decisions = (
            self.decisions.reshape(-1) if self.window_has_decisions else None
        )
        flat_component_mask = (
            self.component_mask.reshape((-1,) + self.component_mask.shape[2:])
            if self.window_has_components
            else None
        )

        indices = torch.randperm(self.buffer_size, device=self.device)
        for start in range(0, self.buffer_size, batch_size):
            mb_inds = indices[start : start + batch_size]
            yield RolloutBufferSample(
                obs=index_obs(flat_obs, mb_inds),
                actions=flat_actions[mb_inds],
                old_values=flat_values[mb_inds],
                old_log_prob=flat_log_probs[mb_inds],
                advantages=flat_advantages[mb_inds],
                returns=flat_returns[mb_inds],
                old_mean=flat_means[mb_inds] if flat_means is not None else None,
                old_log_std=flat_log_stds[mb_inds] if flat_log_stds is not None else None,
                decision_mask=(
                    flat_decisions[mb_inds] if flat_decisions is not None else None
                ),
                component_mask=(
                    flat_component_mask[mb_inds] if flat_component_mask is not None else None
                ),
            )
