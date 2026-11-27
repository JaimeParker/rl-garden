"""An off2on warmup (``--warmup-steps``) pauses the updates of every algorithm
that takes one, as WSRL's loop does for every agent
(``3rd_party/wsrl/finetune.py:337-342``)."""
from __future__ import annotations

import copy

import pytest
import torch

try:
    from tests._cli import run_entry
    from tests.test_entrypoint_builds import (
        PHASES, build_argv, cpu_only_and_restored_rng, inputs_for, shared_inputs,
    )
    from tests.test_replay_buffer_state import assert_tree_equal
except ModuleNotFoundError as exc:
    if exc.name != "tests._cli":
        raise
    from _cli import run_entry
    from test_entrypoint_builds import (
        PHASES, build_argv, cpu_only_and_restored_rng, inputs_for, shared_inputs,
    )
    from test_replay_buffer_state import assert_tree_equal

from rl_garden.training import inspection

__all__ = ["cpu_only_and_restored_rng", "shared_inputs"]  # fixtures

#: The off2on entrypoints whose builders pass ``--warmup-steps`` to the agent.
FAST = ["awac", "iql", "spot", "wsrl"]
SLOW = ["calql"]
#: 2 offline steps, then 16 online env steps of 8-step episodes.
RUN = {
    "env_id": "rlg-entrypoint-build-halfcheetah-pendulum-v0",
    "training_freq": "4", "utd": "0.5", "num_steps": "8", "learning_starts": "0",
    "eval_freq": "0", "online_replay_mode": "append",
    "vae_iterations": "2", "bc_pretrain_steps": "2", "hidden_layers": "2", "num_samples": "2",
    "num_offline_steps": "2", "num_online_steps": "16",
}


def _offline_and_final(name, warmup_steps, tmp_path, shared):
    log_dir = tmp_path / f"warmup_{warmup_steps}"
    inputs = {**RUN, **inputs_for("off2on", name, shared, tmp_path), "warmup_steps": warmup_steps}
    result = run_entry(PHASES["off2on"][0], build_argv("off2on", name, log_dir, inputs=inputs))
    assert result.returncode == 0, result.stderr[-1500:]
    (checkpoints,) = log_dir.glob("*/checkpoints")
    return tuple(
        torch.load(checkpoints / name, map_location="cpu") for name in ("offline_final.pt", "final.pt")
    )


@pytest.mark.parametrize(
    "name", [*FAST, *(pytest.param(name, marks=pytest.mark.slow) for name in SLOW)]
)
def test_a_warmup_longer_than_the_online_stage_updates_nothing(
    name, tmp_path, shared_inputs  # noqa: F811
):
    for warmup_steps, pauses in (("1000", True), ("0", False)):
        offline, final = _offline_and_final(name, warmup_steps, tmp_path, shared_inputs)
        policy, updated = offline["state"]["policy"], final["state"]["policy"]
        unchanged = [key for key in policy if torch.equal(policy[key], updated[key])]
        assert (len(unchanged) == len(policy)) == pauses, (warmup_steps, len(unchanged))
        same_count = offline["state"]["global_update"] == final["state"]["global_update"]
        assert same_count == pauses, warmup_steps


def _update_state(agent) -> dict:
    """Everything a gradient step advances besides the policy: every optimizer
    (``_optimizer_state_dicts``, the checkpoint hook) and every LR scheduler."""
    return copy.deepcopy({
        "optimizers": agent._optimizer_state_dicts(),
        "schedulers": [
            None if scheduler is None else scheduler.state_dict()
            for scheduler in getattr(agent, "_lr_schedulers", [])
        ],
    })


def _rng_state(agent) -> dict:
    state = {"cpu": torch.get_rng_state()}
    if agent.device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(agent.device)
    return state


@pytest.mark.parametrize(
    "name", [*FAST, *(pytest.param(name, marks=pytest.mark.slow) for name in SLOW)]
)
def test_a_paused_train_consumes_no_random_numbers_and_advances_no_optimizer(
    name, tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    built, at_switch = [], []
    construct = inspection.construct_agent

    def capture(*args, **kwargs):
        agent = construct(*args, **kwargs)
        built.append(agent)
        switch = agent.switch_to_online_mode

        def switch_then_snapshot(*switch_args, **switch_kwargs):
            result = switch(*switch_args, **switch_kwargs)
            # SPOT rebuilds its optimizers here: only a later snapshot is the baseline.
            at_switch.append(_update_state(agent))
            return result

        agent.switch_to_online_mode = switch_then_snapshot
        return agent

    inputs = {**RUN, **inputs_for("off2on", name, shared_inputs, tmp_path), "warmup_steps": "1000"}
    argv = build_argv("off2on", name, tmp_path / "paused", inputs=inputs)
    with monkeypatch.context() as patch:
        patch.setattr(inspection, "construct_agent", capture)
        result = run_entry(PHASES["off2on"][0], argv)
    assert result.returncode == 0, result.stderr[-1500:]
    (agent,) = built
    (switched,) = at_switch
    assert agent._active_initial_training_phase() is not None

    # The loop's own paused train() calls moved nothing.
    assert_tree_equal(_update_state(agent), switched)

    rng, updates = _rng_state(agent), _update_state(agent)
    for compute_info in (False, True):
        metrics = agent.train(agent.grad_steps_per_iteration, compute_info=compute_info)
        assert metrics == {}
        assert_tree_equal(_rng_state(agent), rng)
        assert_tree_equal(_update_state(agent), updates)
