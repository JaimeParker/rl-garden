"""Toy vector envs with event-triggered (fixed-hold) decisions and with
partially effective action components, shared by the decision-level PPO
tests."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode

# Four envs with fixed holds of 1, 2, 3 and 5 steps over a 12-step window.
# Episodes end (after-step flag) in the middle of a hold: env 1 is truncated
# after step 4 (hold 4-5), env 2 terminates after step 7 (hold 6-8). Env 1's
# hold 11-12 and env 3's hold 10-14 run past the end of the window.
HOLDS = (1, 2, 3, 5)
NUM_ENVS = len(HOLDS)
NUM_STEPS = 12
EPISODE_ENDS = {(6, 0): "terminated", (4, 1): "truncated", (7, 2): "terminated"}


def _scripted_decisions(
    holds: tuple[int, ...] = HOLDS,
    num_steps: int = NUM_STEPS,
    episode_ends: dict[tuple[int, int], str] = EPISODE_ENDS,
) -> torch.Tensor:
    """Every episode starts with a decision, then one decision per hold."""
    decisions = torch.zeros(num_steps, len(holds), dtype=torch.bool)
    for env, hold in enumerate(holds):
        since_start = 0
        for step in range(num_steps):
            decisions[step, env] = since_start % hold == 0
            since_start = 0 if (step, env) in episode_ends else since_start + 1
    return decisions


def _scripted_flags(
    kind: str,
    num_envs: int = NUM_ENVS,
    num_steps: int = NUM_STEPS,
    episode_ends: dict[tuple[int, int], str] = EPISODE_ENDS,
) -> torch.Tensor:
    flags = torch.zeros(num_steps, num_envs, dtype=torch.bool)
    for (step, env), end in episode_ends.items():
        if step < num_steps:
            flags[step, env] = end == kind
    return flags


class _FixedHoldVecEnv(gym.Env):
    """Batched torch env with per-env fixed holds; reports ``infos["decision"]``
    on every step (unless told otherwise) and ends episodes in the middle of a
    hold per ``episode_ends``. Every tensor lives on ``device`` and ``step()``
    performs no host-device synchronization."""

    single_observation_space = spaces.Dict(
        {"state": spaces.Box(low=-100.0, high=100.0, shape=(4,), dtype=np.float32)}
    )
    single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def __init__(
        self,
        emit_decisions: bool = True,
        drop_key_at_step: int | None = None,
        *,
        holds: tuple[int, ...] = HOLDS,
        episode_ends: dict[tuple[int, int], str] = EPISODE_ENDS,
        max_steps: int = 64,
        device: str | torch.device = "cpu",
    ) -> None:
        self.metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}
        self.num_envs = len(holds)
        self.device = torch.device(device)
        self.observation_space = self.single_observation_space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_envs, 2), dtype=np.float32
        )
        self.emit_decisions = emit_decisions
        self.drop_key_at_step = drop_key_at_step
        self.holds = torch.tensor(holds, dtype=torch.long, device=self.device)
        self.terminated = _scripted_flags(
            "terminated", self.num_envs, max_steps, episode_ends
        ).to(self.device)
        self.truncated = _scripted_flags(
            "truncated", self.num_envs, max_steps, episode_ends
        ).to(self.device)
        self.env_index = torch.arange(self.num_envs, dtype=torch.float32, device=self.device)

    def _state(self) -> torch.Tensor:
        columns = [
            self._since_start.float(),
            self.env_index,
            torch.full((self.num_envs,), float(self._t), device=self.device),
            torch.ones(self.num_envs, device=self.device),
        ]
        return torch.stack(columns, dim=-1) / 10.0

    def reset(self, *, seed=None, options=None):
        del seed, options
        self._t = 0
        self._since_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_state = self._state()
        return {"state": self.last_state}, {}

    def step(self, action):
        del action
        step = self._t
        terminated = self.terminated[step].clone()
        truncated = self.truncated[step].clone()
        done = terminated | truncated
        decision = self._since_start % self.holds == 0
        rewards = 0.1 * (step + 1) + 0.5 * self.env_index
        self._t += 1
        self._since_start = self._since_start + 1
        terminal_state = self._state() + 1.0
        self._since_start = torch.where(done, 0, self._since_start)
        self.last_state = self._state()
        final_state = torch.where(done[:, None], terminal_state, self.last_state)
        infos = {"final_observation": {"state": final_state}}
        if self.emit_decisions and step != self.drop_key_at_step:
            infos["decision"] = decision
        return {"state": self.last_state}, rewards, terminated, truncated, infos

    def close(self) -> None:
        return None


# Per-env cycles of component masks for a 3-dimensional action, restarted at
# every episode start (row 0 of each cycle is a decision). All-false rows are
# steps without a decision. Env 3 never uses component 2; the others do.
COMPONENT_CYCLES = (
    ((1, 1, 0), (0, 0, 1), (1, 1, 1), (1, 0, 0)),
    ((1, 1, 1), (0, 0, 0), (0, 0, 1), (0, 0, 0), (1, 0, 0)),
    ((0, 0, 1), (1, 1, 0), (0, 0, 0)),
    ((1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 0, 0)),
)
# No env ever uses component 2.
COMPONENT_CYCLES_WITHOUT_LAST = (
    ((1, 1, 0), (0, 1, 0), (1, 0, 0)),
    ((1, 0, 0), (0, 0, 0), (0, 1, 0)),
    ((0, 1, 0), (1, 1, 0), (0, 0, 0)),
    ((1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 0, 0)),
)
NUM_COMPONENTS = 3


def _scripted_components(
    cycles: tuple = COMPONENT_CYCLES,
    num_steps: int = NUM_STEPS,
    episode_ends: dict[tuple[int, int], str] = EPISODE_ENDS,
) -> torch.Tensor:
    """(T, N, A) bool component masks; each cycle restarts with every episode."""
    num_components = len(cycles[0][0])
    components = torch.zeros(num_steps, len(cycles), num_components, dtype=torch.bool)
    for env, cycle in enumerate(cycles):
        since_start = 0
        for step in range(num_steps):
            components[step, env] = torch.tensor(cycle[since_start % len(cycle)], dtype=torch.bool)
            since_start = 0 if (step, env) in episode_ends else since_start + 1
    return components


class _ComponentMaskVecEnv(_FixedHoldVecEnv):
    """``_FixedHoldVecEnv`` with a 3-dimensional action whose components take
    effect per the scripted ``cycles``. ``report`` selects what ``step()``
    puts in ``infos``: ``"components"`` the component mask
    (``decision_components``), ``"decision"`` only whether any component took
    effect, ``"broadcast"`` that decision broadcast to every component (as
    ``decision_components``), ``"none"`` nothing. ``drop_key_at_step`` omits
    the key on that step; ``switch_key_at_step`` reports ``decision`` instead
    on that step."""

    single_action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(NUM_COMPONENTS,), dtype=np.float32
    )

    def __init__(
        self,
        report: str = "components",
        *,
        cycles: tuple = COMPONENT_CYCLES,
        episode_ends: dict[tuple[int, int], str] = EPISODE_ENDS,
        drop_key_at_step: int | None = None,
        switch_key_at_step: int | None = None,
        max_steps: int = 64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            emit_decisions=False,
            holds=(1,) * len(cycles),
            episode_ends=episode_ends,
            max_steps=max_steps,
            device=device,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_envs, NUM_COMPONENTS), dtype=np.float32
        )
        if report not in ("components", "decision", "broadcast", "none"):
            raise ValueError(f"unknown report {report!r}")
        self.report = report
        self.switch_key_at_step = switch_key_at_step
        self.drop_component_key_at_step = drop_key_at_step
        self.components = _scripted_components(cycles, max_steps, episode_ends).to(self.device)

    def step(self, action):
        step = self._t
        obs, rewards, terminated, truncated, infos = super().step(action)
        components = self.components[step]
        decision = components.any(dim=-1)
        if self.report == "none" or step == self.drop_component_key_at_step:
            return obs, rewards, terminated, truncated, infos
        if self.report == "decision" or step == self.switch_key_at_step:
            infos["decision"] = decision
        elif self.report == "broadcast":
            infos["decision_components"] = decision[:, None].expand_as(components)
        else:
            infos["decision_components"] = components
        return obs, rewards, terminated, truncated, infos
