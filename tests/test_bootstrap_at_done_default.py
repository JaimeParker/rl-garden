"""Regression test: bootstrap_at_done's literal default must be 'truncated'.

Every off-policy algorithm __init__ and off2on CLI args dataclass currently
redeclares this default independently rather than inheriting a shared one
(see rl_garden/algorithms/off_policy.py's own default plus each subclass).
This pins the exact set of declarations changed from 'always' to
'truncated' so a future edit to any one of them doesn't silently drift.
"""
import importlib
import inspect
from dataclasses import fields

import pytest

_ALGORITHM_CLASSES = [
    ("rl_garden.algorithms.off_policy", "OffPolicyAlgorithm"),
    ("rl_garden.algorithms.sac", "SAC"),
    ("rl_garden.algorithms.ddpg", "DDPG"),
    ("rl_garden.algorithms.iql", "_IQLRolloutTrainingShell"),
    ("rl_garden.algorithms.cql", "_CQLRolloutTrainingShell"),
    ("rl_garden.algorithms.awac", "_AWACRolloutTrainingShell"),
    ("rl_garden.algorithms.spot", "_SPOTRolloutTrainingShell"),
    ("rl_garden.algorithms.wsrl", "WSRL"),
    ("rl_garden.algorithms.acfql", "_ACFQLRolloutTrainingShell"),
    ("rl_garden.algorithms.off2on_iql", "Off2OnIQL"),
    ("rl_garden.algorithms.off2on_calql", "Off2OnCalQL"),
    ("rl_garden.algorithms.off2on_awac", "Off2OnAWAC"),
    ("rl_garden.algorithms.off2on_spot", "Off2OnSPOT"),
    ("rl_garden.algorithms.off2on_so2", "Off2OnSO2"),
]

_ARGS_CLASSES = [
    ("rl_garden.training.off2on.iql", "IQLOff2OnArgs"),
    ("rl_garden.training.off2on.calql", "CalQLOff2OnArgs"),
    ("rl_garden.training.off2on.wsrl", "WSRLOff2OnArgs"),
    ("rl_garden.training.off2on.so2", "SO2Off2OnArgs"),
]


@pytest.mark.parametrize("modpath,clsname", _ALGORITHM_CLASSES, ids=[c for _, c in _ALGORITHM_CLASSES])
def test_algorithm_bootstrap_at_done_default_is_truncated(modpath, clsname):
    cls = getattr(importlib.import_module(modpath), clsname)
    default = inspect.signature(cls.__init__).parameters["bootstrap_at_done"].default
    assert default == "truncated"


@pytest.mark.parametrize("modpath,clsname", _ARGS_CLASSES, ids=[c for _, c in _ARGS_CLASSES])
def test_off2on_args_bootstrap_at_done_default_is_truncated(modpath, clsname):
    cls = getattr(importlib.import_module(modpath), clsname)
    field = next(f for f in fields(cls) if f.name == "bootstrap_at_done")
    assert field.default == "truncated"
