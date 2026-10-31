"""Consumer side of the optional per-step decision keys (episode boundary
contract rules 9-10 in ``.agents/rules/adding-env-backend.md``).

An env with event-triggered (semi-Markov) decisions is stepped and its policy
queried on every step, but the sampled action takes effect only on the steps
the env flags: ``infos["decision"]`` ((N,) bool) or, per action component,
``infos["decision_components"]`` ((N,) + action shape, bool). Validation lives
here, in the consumer, so it covers every backend; adapters forward the keys
unchanged. All checks use tensor metadata and host-side dict lookups only.

See ``docs/guides/decision-level-ppo.md`` for how the flags are used in training.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

DECISION_KEY = "decision"
DECISION_COMPONENTS_KEY = "decision_components"
DECISION_KEYS = (DECISION_KEY, DECISION_COMPONENTS_KEY)

_KEY_SET_UNSEEN = object()


def _present_decision_key(infos: dict[str, Any]) -> str | None:
    present = [key for key in DECISION_KEYS if key in infos]
    if len(present) > 1:
        raise ValueError(
            f"infos holds both {DECISION_KEY!r} and {DECISION_COMPONENTS_KEY!r}; an env "
            "reports exactly one of them."
        )
    return present[0] if present else None


def warn_if_decision_keys_ignored(infos: dict[str, Any]) -> bool:
    """Warn when an env reports decision keys that the algorithm was not
    configured to use. Returns whether it warned, so a caller that warns once
    keeps that state itself."""
    key = next((key for key in DECISION_KEYS if key in infos), None)
    if key is None:
        return False
    warnings.warn(
        f"The env reports infos[{key!r}] but decision-level training is off; every "
        "step is treated as a decision.",
        UserWarning,
        stacklevel=2,
    )
    return True


class DecisionMaskReader:
    """Reads and validates the optional per-step decision keys an env emits
    (contract rules 9-10)."""

    def __init__(
        self,
        num_envs: int,
        action_shape: tuple[int, ...],
        device: torch.device | str,
    ) -> None:
        self.num_envs = int(num_envs)
        self.action_shape = tuple(action_shape)
        self.device = torch.device(device)
        self._key: Any = _KEY_SET_UNSEEN
        self._steps_read = 0

    def _expected_shape(self, key: str) -> tuple[int, ...]:
        if key == DECISION_KEY:
            return (self.num_envs,)
        return (self.num_envs,) + self.action_shape

    def read(
        self, infos: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """``(decision, component_mask)`` of this step on ``device``, or None
        when the env reports no decision key. ``decision`` is the (N,) bool
        decision mask. ``component_mask`` is None for a ``decision`` key; for a
        ``decision_components`` key it is the (N,) + action shape bool mask of
        the components that took effect, and ``decision`` is "any component
        took effect"."""
        key = _present_decision_key(infos)
        step = self._steps_read
        self._steps_read += 1
        if self._key is _KEY_SET_UNSEEN:
            self._key = key
        elif key != self._key:
            raise ValueError(
                f"Decision key set changed at step {step}: earlier steps reported "
                f"{self._key!r}, this step reports {key!r}. An env reports its "
                "decision key on every step of a run or never."
            )
        if key is None:
            return None
        value = infos[key]
        expected_shape = self._expected_shape(key)
        # One error type for every contract violation of the decision keys.
        if not isinstance(value, torch.Tensor):
            raise ValueError(  # noqa: TRY004
                f"infos[{key!r}] must be a torch.Tensor, got {type(value).__name__}."
            )
        if value.dtype != torch.bool:
            raise ValueError(f"infos[{key!r}] must have dtype torch.bool, got {value.dtype}.")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"infos[{key!r}] must have shape {expected_shape}, got {tuple(value.shape)}."
            )
        value = value.to(self.device, torch.bool)
        if key == DECISION_COMPONENTS_KEY:
            return value.reshape(self.num_envs, -1).any(dim=-1), value
        return value, None
