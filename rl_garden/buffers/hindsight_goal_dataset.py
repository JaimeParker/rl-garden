"""Hindsight goal relabeling over a static offline trajectory dataset, ported
from ``3rd_party/HILP/hilp_gcrl/src/dataset_utils.py::GCDataset``.

A stateful, resample-every-call sampler (unlike
``chunked_dataset.py::load_h5_dataset_as_chunks``'s one-shot static
materialization) -- the reference resamples goals fresh every gradient step.
Not HILP-specific: goal-conditioned hindsight relabeling is a broadly useful
technique, so this is a standalone, generically-named component. The
``success`` -> reward-convention transform is deliberately left to the
caller (the sampler only outputs raw 0/1 ``success``), keeping it reusable
by an algorithm with a different reward convention than HILP's own
``reward = success - 1.0``.

Goal sampling is two sequential Bernoulli overrides, not a flat three-way
categorical draw (verified by reading ``GCDataset.sample_goals`` directly):
start from a uniform-random goal, override with a same-trajectory geometric
future goal at probability ``p_trajgoal/(1-p_currgoal)``, then override with
the current state at probability ``p_currgoal``. With the default
``p_currgoal=0`` these framings coincide, but they diverge for any
``p_currgoal>0`` -- implemented here exactly as the reference nests them.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
from dataclasses import dataclass

from rl_garden.buffers._dataset_common import _concat
from rl_garden.buffers._h5_common import _read_node, _require_h5py
from rl_garden.buffers.h5_dataset import _load_traj_transitions
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs


@dataclass
class HindsightGoalSample:
    obs: Obs
    next_obs: Obs
    actions: torch.Tensor
    goals: Obs
    success: torch.Tensor  # (B,), 0/1 float -- reward-convention-agnostic


class HindsightGoalDataset:
    def __init__(
        self,
        path: str | Path,
        *,
        p_currgoal: float = 0.0,
        p_trajgoal: float = 0.625,
        p_randomgoal: float = 0.375,
        discount: float = 0.99,
        device: torch.device | str = "cpu",
        num_traj: Optional[int] = None,
    ) -> None:
        if abs(p_currgoal + p_trajgoal + p_randomgoal - 1.0) > 1e-6:
            raise ValueError(
                "p_currgoal + p_trajgoal + p_randomgoal must sum to 1, got "
                f"{p_currgoal + p_trajgoal + p_randomgoal}."
            )
        self.p_currgoal = p_currgoal
        self.p_trajgoal = p_trajgoal
        self.p_randomgoal = p_randomgoal
        self.discount = discount
        self.device = torch.device(device)

        h5py = _require_h5py()
        path = Path(path)
        obs_parts: list[Obs] = []
        next_obs_parts: list[Obs] = []
        action_parts: list[torch.Tensor] = []
        terminal_locs: list[int] = []
        running_total = 0

        with h5py.File(path, "r") as f:
            keys = sorted(
                (key for key in f if key.startswith("traj_")),
                key=lambda key: int(key.split("_")[-1]),
            )
            if num_traj is not None:
                keys = keys[:num_traj]
            for key in keys:
                traj = _read_node(f[key])
                if not isinstance(traj, dict):
                    continue
                obs, next_obs, actions, _rewards, _dones, _episode_ends = (
                    _load_traj_transitions(traj, self.device)
                )
                length = actions.shape[0]
                if length < 1:
                    continue
                obs_parts.append(obs)
                next_obs_parts.append(next_obs)
                action_parts.append(actions)
                # Each `traj_N` H5 group is exactly one episode in this
                # format -- its own last transition is this trajectory's
                # boundary, independent of whatever the loaded `dones` array
                # says (avoids depending on `dones` granularity/correctness
                # for boundary-finding).
                terminal_locs.append(running_total + length - 1)
                running_total += length

        if not action_parts:
            raise ValueError(f"No usable trajectories found in {path}.")

        self._obs = _concat(obs_parts)
        self._next_obs = _concat(next_obs_parts)
        self._actions = torch.cat(action_parts, dim=0)
        self.size = running_total
        self.terminal_locs = torch.tensor(
            sorted(terminal_locs), device=self.device, dtype=torch.long
        )

    def sample(self, batch_size: int) -> HindsightGoalSample:
        indx = torch.randint(0, self.size, (batch_size,), device=self.device)
        loc = torch.searchsorted(self.terminal_locs, indx)
        final_state_indx = self.terminal_locs[loc]

        u = torch.rand(batch_size, device=self.device)
        offset = torch.ceil(torch.log(1.0 - u) / math.log(self.discount)).long()
        middle_goal_indx = torch.minimum(indx + offset, final_state_indx)

        goal_indx = torch.randint(0, self.size, (batch_size,), device=self.device)
        # p_currgoal == 1.0 makes the trajgoal probability 0/0 -- moot since
        # the p_currgoal override below always fires in that case anyway.
        traj_prob = (
            0.0 if self.p_currgoal >= 1.0 else self.p_trajgoal / (1.0 - self.p_currgoal)
        )
        use_traj = torch.rand(batch_size, device=self.device) < traj_prob
        goal_indx = torch.where(use_traj, middle_goal_indx, goal_indx)
        use_curr = torch.rand(batch_size, device=self.device) < self.p_currgoal
        goal_indx = torch.where(use_curr, indx, goal_indx)

        success = (indx == goal_indx).float()
        return HindsightGoalSample(
            obs=index_obs(self._obs, indx),
            next_obs=index_obs(self._next_obs, indx),
            actions=self._actions[indx],
            goals=index_obs(self._obs, goal_indx),
            success=success,
        )
