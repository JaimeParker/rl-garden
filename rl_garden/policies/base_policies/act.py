"""ACT base-policy adapter for residual RL."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.policies.base_policies.base import BasePolicyOutput, BasePolicyProvider

if TYPE_CHECKING:
    from rl_garden.models.act.provider import StateObsGetter
else:
    StateObsGetter = Any


class ACTBasePolicy(BasePolicyProvider):
    """Wrap ``ACTBaseActionProvider`` behind the residual base-policy interface."""

    def __init__(
        self,
        provider,
        *,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(observation_space, action_space, device=device)
        self.provider = provider
        self.to(device)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        ckpt_path: str | Path | None = None,
        stats_path: str | Path | None = None,
        env: Any | None = None,
        state_obs_getter: Optional[StateObsGetter] = None,
        temporal_agg: bool = True,
        temporal_agg_k: float = 0.01,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> "ACTBasePolicy":
        from rl_garden.models.act import ACTBaseActionProvider

        provider = ACTBaseActionProvider.from_checkpoint(
            observation_space=observation_space,
            action_space=action_space,
            ckpt_path=ckpt_path,
            stats_path=stats_path,
            env=env,
            state_obs_getter=state_obs_getter,
            temporal_agg=temporal_agg,
            temporal_agg_k=temporal_agg_k,
            strict=strict,
            device=device,
        )
        return cls(
            provider,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

    @property
    def checkpoint_path(self):
        return self.provider.checkpoint_path

    @property
    def spec(self):
        return self.provider.spec

    @property
    def config(self):
        return self.provider.config

    def to(self, *args, **kwargs):  # type: ignore[override]
        module = super().to(*args, **kwargs)
        self.provider.to(self.device)
        return module

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        self.provider.reset(env_ids=env_ids)

    def bind_env(self, env: Any) -> None:
        self.provider.bind_env(env)

    @torch.no_grad()
    def select_action(self, obs: Obs) -> BasePolicyOutput:
        actions = self.provider.select_action(self._obs_to_device(obs))
        return BasePolicyOutput(actions=self._format_actions(actions))
