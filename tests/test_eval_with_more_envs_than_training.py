"""Evaluation builds the agent with ``--num-eval-envs`` envs; a constructor check
on the training env count must not reject it. Every evaluable algorithm trains a
few steps with 2 envs and is evaluated with 64."""
from __future__ import annotations

import pytest
import test_checkpoint_eval_is_self_contained as self_contained

from rl_garden.training.evaluation import EvalCheckpointArgs, evaluate_checkpoint

shared_inputs = self_contained.shared_inputs
cpu_only_and_restored_rng = self_contained.cpu_only_and_restored_rng

FAST_LANE = {("online", "dreamer_v3"), ("online", "ncrl")}
# (phase, algorithm) -> why it is not evaluated with 64 envs.
EXEMPTIONS = {
    ("online", "tdmpc2"): "its CEM planner acts in one env (num_eval_envs == 1 required)",
}


@pytest.mark.parametrize(
    "phase,name",
    [
        pytest.param(*key, id="-".join(key), marks=() if key in FAST_LANE else pytest.mark.slow)
        for key in self_contained.CASES
        if key not in EXEMPTIONS
    ],
)
def test_evaluation_with_more_envs_than_training(phase, name, tmp_path, shared_inputs, monkeypatch):
    checkpoint, _, _ = self_contained._train(phase, name, shared_inputs, tmp_path, monkeypatch)
    result = evaluate_checkpoint(
        EvalCheckpointArgs(
            checkpoint_path=str(checkpoint), phase=phase, algorithm=name, num_eval_envs=64,
            num_eval_episodes=1, device="cpu",
        )
    )
    assert result["metrics"]["eval_steps"] > 0
