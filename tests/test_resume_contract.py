"""Registry-wide exact checkpoint preservation and continued training.

CPU compatibility fixtures are shared with the registry build/evaluation tests.
The checkpoint basename is identical in separate A/B directories, so references
to replay and sidecars can be compared without normalizing or excluding paths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import test_checkpoint_eval_is_self_contained as evaluation
import test_entrypoint_builds as builds
import test_off2on_resume as off2on
import torch

from rl_garden.algorithms.offline import run_offline_pretraining
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.on_policy import OnPolicyAlgorithm
from rl_garden.buffers.base import BaseReplayBuffer

shared_inputs = builds.shared_inputs
cpu_only_and_restored_rng = builds.cpu_only_and_restored_rng
pytestmark = pytest.mark.slow

ENTRY_EXEMPTIONS = {}
NOT_APPLICABLE = {
    ("online", "gail"): "D4RL-only demonstrations; rl_garden/algorithms/gail.py:138 requires demo_env_id",
}
# Explicit runner ownership: checkpoint A must contain both keys iff listed.
# Writers: training/online/runner.py:167, offline/runner.py:196,
# off2on/runner.py:571; bespoke offline runners do not attach run_config.
RUN_CONFIG_ENTRIES = {
    *{("online", name) for name in (
        "acrlpd", "dagger", "diffusion_cm_distill_online", "dppo", "dreamer_v3",
        "drqv2", "explore", "flash_sac", "flow_ppo", "ncrl", "policy_distillation",
        "ppo", "recurrent_ppo", "recurrent_sac", "rlpd", "rlpd_hybrid", "sac",
        "sac_flow", "supe", "supermimic", "td3", "tdmpc2", "transformer_ppo", "transformer_sac",
    )},
    *{("offline", name) for name in (
        "awac", "bc", "bcq", "bppo", "calql", "cql", "edac", "fac", "fino", "floq",
        "flow_bc", "fql", "idql", "iql", "mean_flow_bc", "plas", "qam", "qgf",
        "rebrac", "spot", "td3_bc", "unio4", "unio4_ope", "value_flows", "wsrl",
    )},
    *{("off2on", name) for name in (
        "acfql", "awac", "calql", "fac", "fino", "floq", "fql", "iql", "spot", "value_flows", "wsrl",
    )},
}
KNOWN_GAPS = {}
# Non-growing counters identify completed phases or phase origins, not new updates.
COUNTER_RULES = {
    "offline": "learn(2) adds 2 global steps and 2 updates; offline WSRL uses its runner's offline loop",
    "off2on-offline": "run_offline_pretraining(2, gradient_steps=1) adds 2 steps and 2 updates",
    "ncrl": "ceil(learning_starts / iteration_steps) refill iterations, then one more; "
            "two update iterations including the threshold-crossing iteration (online replay is not saved)",
    "off-policy": "two iterations add 2*steps_per_env*num_envs steps and 2*grad_steps_per_iteration updates",
    "on-policy": "one rollout adds num_steps*num_envs steps and one update",
    "policy_distillation": "one rollout adds num_steps*num_envs steps and num_learning_epochs*num_minibatches updates",
    "dagger": "one round adds rollout_steps_per_round*num_envs steps, gradient_steps_per_round updates, and one round",
    "bppo/unio4": "phase_step and phase_b_step each add the new offline step count",
    "fac": "phase_step stays at the completed BC-prelude budget (fac.py:666-669); offline BC scheduler stays fixed (fac.py:713)",
    "initial_phase_start_step/online_start_step/spot_online_update_start": "fixed origins, preserved by C1",
}
EXCLUDED_KEYS = {
    f"root.metadata.{key}": (
        "Runner-owned per-run provenance, deliberately not restored by load; "
        "resume arguments come from the CLI (docs/guides/checkpoint.md:116-118). "
        "Writers: rl_garden/training/online/runner.py:167, "
        "rl_garden/training/offline/runner.py:196, rl_garden/training/off2on/runner.py:571; "
        "serialization: rl_garden/algorithms/base_algorithm.py:424-426."
    )
    for key in ("run_config", "run_config_digest")
}
CASES = sorted(builds.REGISTERED - ENTRY_EXEMPTIONS.keys())


def differences(left, right, path="root"):
    """Exact leaves: tensor dtype/shape, identical NaN masks, exact non-NaNs."""
    if path in EXCLUDED_KEYS:
        return []
    if type(left) is not type(right):
        return [f"{path}: type {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, torch.Tensor):
        equal = left.dtype == right.dtype and left.shape == right.shape
        if equal:
            left_nan, right_nan = torch.isnan(left), torch.isnan(right)
            equal = (torch.equal(left_nan, right_nan)
                     and torch.equal(left[~left_nan], right[~right_nan]))
        return [] if equal else [f"{path}: tensors differ"]
    if isinstance(left, np.ndarray):
        equal = left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
        return [] if equal else [f"{path}: arrays differ"]
    if isinstance(left, dict):
        result = [f"{path}.{key}: missing key" for key in left.keys() ^ right.keys()
                  if f"{path}.{key}" not in EXCLUDED_KEYS]
        for key in left.keys() & right.keys():
            result.extend(differences(left[key], right[key], f"{path}.{key}"))
        return result
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        return [error for index, (a, b) in enumerate(zip(left, right))
                for error in differences(a, b, f"{path}.{index}")]
    return [] if left == right else [f"{path}: {left!r} != {right!r}"]


def _read(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _run(phase, name, argv, monkeypatch, *, dry=False):
    with evaluation._captured_agents(monkeypatch) as agents:
        result = builds.run_entry(builds.PHASES[phase][0], [*argv, *(["--dry-run"] if dry else [])])
    assert result.returncode == 0, result.stderr[-4000:]
    assert agents
    return agents[-1]


def _arguments(phase, name, shared, root):
    key = phase, name
    inputs = builds.inputs_for(phase, name, shared, root)
    inputs.update(evaluation.OPTIONAL_INPUTS.get(key, lambda s: {})(shared))
    if key == ("online", "ncrl"):
        artifacts = evaluation.make_pretrain_artifacts(root / "ncrl")
        inputs.update(pretrained_world_model=str(artifacts["checkpoint"]),
                      retrieval_db=str(artifacts["retrieval_db"]),
                      prior_dataset_dir=str(artifacts["dataset_dir"]))
    values = {**evaluation.SMALL_TRAINING, **builds.OVERRIDES.get(key, {}),
              **evaluation.TRAINING_OVERRIDES.get(key, {}), **inputs,
              "checkpoint_freq": "4", "save_replay_buffer": name != "ncrl",
              "lr_schedule": "linear_warmup", "lr_warmup_steps": "100"}
    if name == "unio4_ope":
        # The tiny dynamics/OPE settings of test_unio4_ope._make_agent.
        values.update(num_offline_steps="6", ope_gating_freq="1",
                      dynamics_hidden_dims="8 8", dynamics_weight_decay="1e-5 1e-5 1e-5",
                      dynamics_n_ensemble="2", dynamics_n_elites="1", dynamics_max_epochs="2",
                      dynamics_max_epochs_since_update="1", dynamics_batch_size="32",
                      ope_rollout_length="5", ope_rollout_batch_size="8")
    if name == "pwm":
        values["horizon"] = "2"
    if name == "supermimic":
        from test_supermimic_smoke import SMALL
        values.update({key: " ".join(map(str, value)) if isinstance(value, tuple)
                       else str(value) for key, value in SMALL.items() if key != "device"})
        values["std_log"] = False
    return builds.build_argv(phase, name, root / "runs", with_eval=key in builds.EVAL_ONLY,
                             inputs=values)


def _assert_trained(checkpoint):
    state = checkpoint["state"]
    assert state["global_step"] > 0
    assert state["global_update"] > 0
    for name in ("phase_step", "phase_b_step", "round_num"):
        if name in state["training_state"]:
            assert state["training_state"][name] > 0, f"{name} never advanced"
    for name in ("last_ope_check", "guided_steps"):
        if name in state["extra"]:
            assert state["extra"][name] > 0, f"{name} never advanced"
    assert state["optimizers"], "no checkpointed optimizer"
    for name, optimizer in state["optimizers"].items():
        assert optimizer["state"], f"{name} never stepped"
        steps = [slot["step"] for slot in optimizer["state"].values() if "step" in slot]
        assert steps and all(step > 0 for step in steps), f"{name}: nonpositive optimizer step"


def _roundtrip(agent, rebuilt, root, source=None):
    include = isinstance(getattr(agent, "replay_buffer", None), BaseReplayBuffer)
    include = include or any(slot.buffer is not None for slot in agent._sidecar_buffers())
    path = source or agent.save(root / "A" / "checkpoint.pt", include_replay_buffer=include)
    before = _read(path)
    _assert_trained(before)
    rebuilt.load(path, load_replay_buffer=True)
    after_path = rebuilt.save(root / "B" / path.name, include_replay_buffer=include)
    after = _read(after_path)
    errors = differences(before, after)
    references = {before["metadata"]["replay_buffer_path"]} - {None}
    for slot in rebuilt._sidecar_buffers():
        reference = before["state"]["extra"]
        for key in slot.reference_key:
            reference = reference.get(key) if isinstance(reference, dict) else None
        if reference is not None:
            references.add(reference)
    assert {p.name for p in after_path.parent.glob("*.pt")} == references | {path.name}
    for second in after_path.parent.glob("*.pt"):
        if second == after_path:
            continue
        first = path.parent / second.name
        assert first.exists(), second.name
        errors.extend(differences(_read(first), _read(second), second.name))
    return before, errors


def _check_c1(key, before, errors):
    expected = {"run_config", "run_config_digest"} if key in RUN_CONFIG_ENTRIES else set()
    assert before["metadata"].keys() & {"run_config", "run_config_digest"} == expected, (
        f"{key}: checkpoint A runner metadata ownership changed"
    )
    assert not errors, f"C1 differences for {key}:\n" + "\n".join(errors)


def _schedulers(agent):
    for name, value in vars(agent).items():
        candidates = value if isinstance(value, (list, tuple)) else (value,)
        for index, scheduler in enumerate(candidates):
            if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
                yield f"{name}.{index}", scheduler


def _continue(agent, before, root, monkeypatch, *, offline=False, scheduler_gap=False):
    state = before["state"]
    start, updates = state["global_step"], state["global_update"]
    scheduler_states = {}
    for name, scheduler in _schedulers(agent):
        scheduler_states[name] = (scheduler, scheduler.last_epoch, [0])
        if scheduler_gap:
            assert scheduler.last_epoch == 0, f"{name}: stale missing-scheduler gap"
        else:
            assert scheduler.last_epoch > 0, f"{name} never stepped"
        calls = scheduler_states[name][2]
        step = type(scheduler).step
        def counted(self, *args, _step=step, _calls=calls, _scheduler=scheduler, **kwargs):
            if self is _scheduler:
                _calls[0] += 1
            return _step(self, *args, **kwargs)
        monkeypatch.setattr(type(scheduler), "step", counted)
    agent.checkpoint_dir = str(root / "continued")
    agent.checkpoint_freq = 1
    agent.logger = None
    if offline:
        amount, expected_updates = 2, 2
        if isinstance(agent, OffPolicyAlgorithm):
            run_offline_pretraining(agent, num_steps=amount, gradient_steps=1,
                                   checkpoint_dir=agent.checkpoint_dir, checkpoint_freq=1,
                                   std_log=False, log_freq=0)
        else:
            agent.learn(amount)
    else:
        if isinstance(agent, OnPolicyAlgorithm):
            amount, expected_updates = agent.num_steps * agent.num_envs, 1
            if type(agent).__name__ == "PolicyDistillation":
                expected_updates = agent.num_learning_epochs * agent.num_minibatches
        elif isinstance(agent, OffPolicyAlgorithm):
            amount = 2 * agent.steps_per_env * agent.num_envs
            expected_updates = 2 * agent.grad_steps_per_iteration
            if type(agent).__name__ == "NCRL":
                iteration_steps = agent.steps_per_env * agent.num_envs
                refill_iterations = max(1, (agent.learning_starts + iteration_steps - 1) // iteration_steps)
                amount = (refill_iterations + 1) * iteration_steps
        elif type(agent).__name__ == "DAgger":
            amount = agent.rollout_steps_per_round * agent.num_envs
            expected_updates = agent.gradient_steps_per_round
        else:
            amount = agent.num_steps * agent.num_envs
            expected_updates = agent.num_learning_epochs * agent.num_minibatches
        agent.learn(start + amount)
    assert agent._global_step == start + amount
    assert agent._global_update == updates + expected_updates
    assert agent._global_update > updates
    periodic = list(Path(agent.checkpoint_dir).glob("checkpoint_*.pt"))
    assert periodic, "continuation crossed no checkpoint boundary"
    for path in periodic:
        named_step = int(path.stem.removeprefix("checkpoint_"))
        assert named_step > start, path.name
        assert _read(path)["state"]["global_step"] == named_step
    for path in Path(agent.checkpoint_dir).glob("*.pt"):
        saved = _read(path)
        if "state" in saved and "global_step" in saved["state"]:
            assert saved["state"]["global_step"] > start, path.name
    for name, (scheduler, epoch, calls) in scheduler_states.items():
        frozen_bc = (type(agent).__name__ in {"FAC", "Off2OnFAC"}
                     and offline and name == "_lr_schedulers.2")
        if frozen_bc:
            assert calls[0] == 0, "FAC offline BC prelude must stay finished (fac.py:713)"
        else:
            assert calls[0] > 0, f"{name} did not advance"
        assert scheduler.last_epoch == epoch + calls[0], name
        assert scheduler.get_last_lr() == [g["lr"] for g in scheduler.optimizer.param_groups]
    after = agent.state_dict()
    for name in ("phase_step", "phase_b_step", "round_num"):
        if name in state["training_state"]:
            old = state["training_state"][name]
            assert old > 0, f"{name} never advanced"
            increment = 1 if name == "round_num" else amount
            if type(agent).__name__ in {"FAC", "Off2OnFAC"} and name == "phase_step":
                increment = 0  # fac.py:666-669 advances only the completed BC prelude.
            assert after["training_state"][name] == old + increment
    if "last_ope_check" in state["extra"]:
        assert after["extra"]["last_ope_check"] == state["extra"]["last_ope_check"] + expected_updates
    return after


@pytest.mark.parametrize("phase,name", [key for key in CASES if key[0] != "off2on"],
                         ids=[f"{p}-{n}" for p, n in CASES if p != "off2on"])
def test_resume_contract(phase, name, tmp_path, shared_inputs, monkeypatch):
    argv = _arguments(phase, name, shared_inputs, tmp_path)
    if (phase, name) == ("online", "gail"):
        with pytest.raises(ValueError, match="demo_env_id must be set to a D4RL dataset id"):
            _run(phase, name, argv, monkeypatch)
        return
    agent = _run(phase, name, argv, monkeypatch)
    for scheduler_name, scheduler in _schedulers(agent):
        assert scheduler.last_epoch > 0, f"{scheduler_name} never stepped before save"
    rebuilt = _run(phase, name, argv, monkeypatch, dry=True)
    before, errors = _roundtrip(agent, rebuilt, tmp_path)
    _check_c1((phase, name), before, errors)
    gap = KNOWN_GAPS.get((phase, name), {})
    if "continuation" in gap:
        with pytest.raises(RuntimeError, match=gap["continuation"][0]):
            _continue(rebuilt, before, tmp_path, monkeypatch)
        return
    if "scheduler" in gap:
        original = dict(_schedulers(agent))
        restored = dict(_schedulers(rebuilt))
        assert original.keys() == restored.keys() and original
        for key, scheduler in original.items():
            assert scheduler.last_epoch > 0
            assert restored[key].last_epoch == 0, "stale missing-scheduler gap"
    after = _continue(rebuilt, before, tmp_path, monkeypatch, offline=phase == "offline",
                      scheduler_gap="scheduler" in gap)
    print("CONTRACT", phase, name, "C1", errors, "COUNTERS", before["state"]["training_state"],
          after["training_state"])
    _check_c1((phase, name), before, errors)


@pytest.mark.parametrize("name,mode", [(n, m) for n in off2on.ALGORITHMS
                                       for m in off2on.MODES])
@pytest.mark.parametrize("stage", ["offline", "online"])
def test_off2on_resume_contract(name, mode, stage, tmp_path, shared_inputs, monkeypatch):
    values = {"num_offline_steps": "4", "num_online_steps": "32", "checkpoint_freq": "4",
              "lr_schedule": "linear_warmup", "lr_warmup_steps": "100"}
    agent, path = off2on._trained(name, mode, shared_inputs, tmp_path, monkeypatch, **values)
    if stage == "offline":
        path = path.with_name("offline_final.pt")
    inputs = off2on._inputs(name, shared_inputs, tmp_path,
                          **{**off2on.MODES[mode], **off2on.EMPTY_MC.get((name, mode), {}),
                             **off2on.OFFLINE_ONLINE, **values})
    rebuilt, result = off2on._run(name, tmp_path / "rebuild", inputs, monkeypatch, "--dry-run")
    assert result.returncode == 0, result.stderr[-4000:]
    before, errors = _roundtrip(agent, rebuilt, tmp_path, source=path)
    _check_c1(("off2on", name), before, errors)
    _continue(rebuilt, before, tmp_path, monkeypatch, offline=stage == "offline")
    print("CONTRACT", name, mode, stage, errors)
    _check_c1(("off2on", name), before, errors)


def test_registry_tables_are_current():
    assert ENTRY_EXEMPTIONS.keys() <= builds.REGISTERED
    assert set(NOT_APPLICABLE) == {
        ("online", "gail"),
    }, "every applicability entry needs its corresponding runtime witness"
    assert NOT_APPLICABLE.keys() <= builds.REGISTERED
    assert KNOWN_GAPS.keys() <= builds.REGISTERED
    assert set(CASES) | ENTRY_EXEMPTIONS.keys() == builds.REGISTERED
    assert set(EXCLUDED_KEYS) == {
        "root.metadata.run_config", "root.metadata.run_config_digest",
    }
    assert all(EXCLUDED_KEYS.values())
    assert RUN_CONFIG_ENTRIES <= builds.REGISTERED
    assert not (RUN_CONFIG_ENTRIES & (ENTRY_EXEMPTIONS.keys() | NOT_APPLICABLE.keys()))
    assert all(NOT_APPLICABLE.values())
    assert all(set(gap) and set(gap) <= {"scheduler", "continuation"}
               for gap in KNOWN_GAPS.values()), "every gap needs a supported runtime witness"
    assert not (KNOWN_GAPS.keys() & ENTRY_EXEMPTIONS.keys())


def test_tensor_leaf_equality():
    original = torch.tensor([float("nan"), 1.0, 2.0])
    assert not differences(original, original.clone())
    for changed in (
        torch.tensor([1.0, float("nan"), 2.0]),
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([float("nan"), 1.0, 3.0]),
        original.double(),
        original.reshape(1, 3),
    ):
        assert differences(original, changed) == ["root: tensors differ"]
