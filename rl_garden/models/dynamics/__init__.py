"""Ensemble forward-dynamics model for OPE (off-policy evaluation) gating.
Currently consumed by `rl_garden.algorithms.unio4_ope.UniO4OPE`.
"""

from rl_garden.models.dynamics.network import EnsembleDynamicsModel, EnsembleLinear
from rl_garden.models.dynamics.termination_fns import TerminationFn, get_termination_fn
from rl_garden.models.dynamics.trainer import rollout_q_mean, train_ensemble

__all__ = [
    "EnsembleDynamicsModel",
    "EnsembleLinear",
    "TerminationFn",
    "get_termination_fn",
    "rollout_q_mean",
    "train_ensemble",
]
