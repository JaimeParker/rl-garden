"""Helpers shared by batched-vector-env wrappers that run several inner
``step()`` calls per outer step and report one SAME_STEP transition
(``action_chunk.py``, ``skill_action_wrapper.py``, ``action_repeat.py``).
"""
from __future__ import annotations

from typing import Any

import torch

from rl_garden.common.types import Obs


def _tree_where(mask: torch.Tensor, new: Obs, old: Obs) -> Obs:
    if isinstance(new, dict):
        return {key: _tree_where(mask, new[key], old[key]) for key in new}
    view_shape = (mask.shape[0],) + (1,) * (new.dim() - 1)
    return torch.where(mask.view(view_shape), new, old)


def _tree_clone(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {key: _tree_clone(value) for key, value in tree.items()}
    return tree.clone()


def _tree_structure(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {key: _tree_structure(value) for key, value in tree.items()}
    return None


def _freeze_newly_done(name: str, newly_done: torch.Tensor, new: Any, frozen: Any) -> Any:
    """``new`` cloned on its first sub-step (``frozen is None``), else ``new``
    on the ``newly_done`` rows and ``frozen`` elsewhere."""
    if frozen is None:
        return _tree_clone(new)
    if _tree_structure(new) != _tree_structure(frozen):
        raise ValueError(
            f"infos[{name!r}] keys changed between sub-steps: "
            f"{_tree_structure(frozen)!r} then {_tree_structure(new)!r}."
        )
    return _tree_where(newly_done, new, frozen)


class ChunkAccumulator:
    """Aggregates the sub-steps of one outer step of a batched SAME_STEP
    autoreset env into one transition.

    Call ``begin()`` once per outer step, ``add()`` after every inner
    ``step()``, then ``finish(last_infos)`` with the infos of the last inner
    step. Per env:

    - ``reward`` sums the sub-steps up to and including the env's first
      episode end in the chunk; later sub-steps (already in the new episode)
      are not counted.
    - ``terminated``/``truncated`` are the flags of that first episode end.
    - ``final_info``/``final_observation`` are reported only if the inner env
      reported them on some sub-step of the chunk, and then on every outer
      step. They are cloned on their first sub-step and frozen at the sub-step
      each env ended on; ``_final_info``/``_final_observation`` mark every env
      that ended in the chunk. A key set that changes between sub-steps raises
      ``ValueError``.
    - ``log`` is the whole dict from the last sub-step that reported one. Every
      other info key comes from the last sub-step.
    - A second episode end of the same env within one chunk is dropped: its
      flags, reward and ``final_*`` stay those of the first. Drops are counted
      per env in ``dropped_episode_ends`` ((N,) int64 on the env device,
      allocated on the first ``add()``, cumulative since then, not cleared by
      ``begin()``).

    No host sync: all masking is tensor ops on the device of the inner
    ``terminated``/``truncated`` tensors.
    """

    def __init__(self) -> None:
        self.dropped_episode_ends: torch.Tensor | None = None
        self.begin()

    def begin(self) -> None:
        self._active: torch.Tensor | None = None
        self._reward_sum: torch.Tensor | None = None
        self._terminated: torch.Tensor | None = None
        self._truncated: torch.Tensor | None = None
        self._final_info: Any = None
        self._final_observation: Any = None
        self._log: Any = None

    def add(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict[str, Any],
    ) -> None:
        done = terminated | truncated
        if self._active is None:
            self._active = torch.ones_like(done, dtype=torch.bool)
            self._reward_sum = torch.zeros_like(reward)
            self._terminated = torch.zeros_like(self._active)
            self._truncated = torch.zeros_like(self._active)
            if self.dropped_episode_ends is None:
                self.dropped_episode_ends = torch.zeros_like(done, dtype=torch.int64)
        active = self._active
        newly_done = active & done
        self._reward_sum = self._reward_sum + reward * active.to(reward.dtype)
        self._terminated = self._terminated | (terminated & newly_done)
        self._truncated = self._truncated | (truncated & newly_done)
        self.dropped_episode_ends += done & ~active
        self._active = active & ~done
        if "final_info" in infos:
            self._final_info = _freeze_newly_done("final_info", newly_done, infos["final_info"], self._final_info)
        if "final_observation" in infos:
            self._final_observation = _freeze_newly_done(
                "final_observation", newly_done, infos["final_observation"], self._final_observation
            )
        if "log" in infos:
            self._log = infos["log"]

    def finish(
        self, last_infos: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self._active is None:
            raise RuntimeError("ChunkAccumulator.finish() called before any add() since begin().")
        out_infos = {
            key: value
            for key, value in last_infos.items()
            if key != "log" and not key.startswith(("final_", "_final_"))
        }
        ended = ~self._active
        if self._final_info is not None:
            out_infos["final_info"] = self._final_info
            out_infos["_final_info"] = ended
        if self._final_observation is not None:
            out_infos["final_observation"] = self._final_observation
            out_infos["_final_observation"] = ended
        if self._log is not None:
            out_infos["log"] = self._log
        return self._reward_sum, self._terminated, self._truncated, out_infos
