"""Regression test for OfflineRLAlgorithm._evaluate()'s final_info masking.

Verifies the fix directly at the _evaluate() call site (not just indirectly
through the Minari adapter): unmasked garbage/in-progress per-env metric
values must not leak into the averaged eval metrics, and the lockstep case
(every env finishes on the same step, e.g. ManiSkill-style fixed episode
length) must be unaffected.
"""
from types import SimpleNamespace

import torch

from rl_garden.algorithms.offline import OfflineRLAlgorithm


class _StubPolicy:
    def eval(self):
        pass

    def train(self):
        pass

    def predict(self, obs, deterministic=True):
        return torch.zeros(obs.shape[0], 1)


class _ScriptedEvalEnv:
    """Replays a fixed sequence of step() results."""

    def __init__(self, steps):
        self._steps = list(steps)
        self._i = 0

    def reset(self):
        return torch.zeros(2, 1), {}

    def step(self, action):
        result = self._steps[self._i]
        self._i += 1
        return result


class _MinimalOfflineAlgo(OfflineRLAlgorithm):
    def _setup_model(self) -> None:
        pass

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        return {}


def _make_algo(eval_env, num_eval_steps):
    algo = _MinimalOfflineAlgo(
        env=SimpleNamespace(num_envs=2),
        eval_env=eval_env,
        num_eval_steps=num_eval_steps,
    )
    algo.policy = _StubPolicy()
    return algo


def test_evaluate_excludes_non_final_garbage_values_from_partial_termination():
    # Step 1: only env 0 finishes (r=10.0); env 1's "999.0" is an in-progress,
    # not-yet-final running value that must be excluded from the mean.
    step1_infos = {
        "final_info": {"episode": {"r": torch.tensor([10.0, 999.0])}},
        "_final_info": torch.tensor([True, False]),
    }
    # Step 2: only env 1 finishes (r=20.0); env 0's "999.0" must be excluded.
    step2_infos = {
        "final_info": {"episode": {"r": torch.tensor([999.0, 20.0])}},
        "_final_info": torch.tensor([False, True]),
    }
    obs = torch.zeros(2, 1)
    env = _ScriptedEvalEnv(
        [
            (obs, torch.zeros(2), torch.zeros(2), torch.zeros(2), step1_infos),
            (obs, torch.zeros(2), torch.zeros(2), torch.zeros(2), step2_infos),
        ]
    )
    algo = _make_algo(env, num_eval_steps=2)

    out = algo._evaluate()

    # Correct: mean of the two real completions (10.0, 20.0) == 15.0.
    # A regression to the unmasked bug would instead average in the 999.0
    # garbage entries and produce a wildly different number.
    assert out["return"] == 15.0


def test_evaluate_lockstep_all_envs_done_together_matches_unmasked_result():
    # Every env finishes on the same step (e.g. ManiSkill fixed-episode-length
    # eval): the mask is all-True, so masking must be a no-op and reproduce
    # exactly the pre-fix behavior (no regression for the common case).
    step_infos = {
        "final_info": {"episode": {"r": torch.tensor([4.0, 8.0])}},
        "_final_info": torch.tensor([True, True]),
    }
    obs = torch.zeros(2, 1)
    env = _ScriptedEvalEnv(
        [(obs, torch.zeros(2), torch.zeros(2), torch.zeros(2), step_infos)]
    )
    algo = _make_algo(env, num_eval_steps=1)

    out = algo._evaluate()

    assert out["return"] == 6.0  # mean(4.0, 8.0), identical to pre-fix torch.stack().mean()
