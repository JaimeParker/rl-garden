"""``TDMPC2Policy``: the ``BasePolicy`` wrapper around ``WorldModel``.

Exists purely so ``BaseAlgorithm``'s already-implemented ``save()``/``load()``/
``state_dict()``/``_evaluate()`` keep working unmodified for a model-based
algorithm whose action selection is a planner, not a single forward pass
through an actor network (see ``.agents/rules/adding-algorithm.md``'s "do not
reimplement save()/load()" rule).
"""
from __future__ import annotations

import torch

from rl_garden.algorithms.tdmpc2 import planner as planner_mod
from rl_garden.algorithms.tdmpc2.planner import PlannerConfig
from rl_garden.algorithms.tdmpc2.world_model import WorldModel
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.policies.base import BasePolicy


class TDMPC2Policy(BasePolicy):
    def __init__(
        self,
        world_model: WorldModel,
        planner_cfg: PlannerConfig,
        discount: float,
        use_planner: bool = True,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.planner_cfg = planner_cfg
        self.discount = discount
        self.use_planner = use_planner
        self._prev_mean: torch.Tensor | None = None
        self._t0 = True

    @property
    def features_extractor(self) -> BaseFeaturesExtractor:
        return self.world_model.encoder

    def reset_episode(self) -> None:
        """Call at the start of every episode (training rollout or eval)."""
        self._prev_mean = None
        self._t0 = True

    def notify_step_done(self, done: bool) -> None:
        """Call after every env step with whether that step ended the episode
        (terminated or truncated) -- the *next* ``predict()`` call should then
        plan with ``t0=True`` / a cold planning mean."""
        if done:
            self.reset_episode()

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        if self.use_planner:
            action, self._prev_mean = planner_mod.plan(
                self.world_model,
                obs,
                self._prev_mean,
                self.discount,
                self.planner_cfg,
                self._t0,
                eval_mode=deterministic,
            )
            action = action.unsqueeze(0)
        else:
            with torch.no_grad():
                z = self.world_model.encode(obs)
                action, info = self.world_model.pi(z)
                if deterministic:
                    action = info["mean"]
        self._t0 = False
        return action
