"""TD3 algorithm: DDPG + delayed policy updates + decoupled target policy
smoothing + Q1-only actor loss.

Clipped double-Q learning (twin critics, min-for-target) is already present in
``DDPG`` via ``DrQv2Critic`` and does not need to be reimplemented here.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from rl_garden.algorithms.ddpg import DDPG
from rl_garden.hyperparameters import resolve_params
from rl_garden.hyperparameters.td3 import TD3Params



class TD3(DDPG):
    """TD3 (Fujimoto et al. 2018) as a thin extension of ``DDPG``.

    Overrides three hooks on top of ``DDPG``'s existing twin-Q critic:

    * ``_target_action_noise``: fixed ``(target_noise_std, target_noise_clip)``
      decoupled from the exploration schedule, instead of reusing it.
    * ``_should_update_actor_and_target``: delays actor + target updates to
      every ``policy_freq`` gradient steps.
    * ``_actor_q_value``: uses the first critic only (canonical TD3 actor
      loss), instead of DDPG's ``min(q1, q2)``.
    """

    params_class = TD3Params

    _compatible_checkpoint_algorithms = ("TD3",)

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        *,
        params: Optional[TD3Params] = None,
        **ddpg_kwargs: Any,
    ) -> None:
        parameter_names = self.params_class.__dataclass_fields__
        overrides = {name: ddpg_kwargs.pop(name) for name in tuple(ddpg_kwargs)
                     if name in parameter_names}
        params = resolve_params(self.params_class, params, overrides)
        self.policy_freq = params.policy_freq
        self.target_noise_std = params.target_noise_std
        self.target_noise_clip = params.target_noise_clip

        super().__init__(env, eval_env, params=params, **ddpg_kwargs)

    def _target_action_noise(self) -> tuple[float, float]:
        return self.target_noise_std, self.target_noise_clip

    def _should_update_actor_and_target(self) -> bool:
        return self._global_update % self.policy_freq == 0

    def _actor_q_value(self, q_actor_all: torch.Tensor) -> torch.Tensor:
        return q_actor_all[0]

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
        }
