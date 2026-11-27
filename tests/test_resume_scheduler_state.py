"""Scheduler round trips and pre-scheduler checkpoints on CPU."""
import copy

import pytest
import test_edac_core as edac
import test_idql as idql
import test_td3 as td3
import torch

FACTORIES = {"edac": edac._make_agent, "idql": idql._make_agent, "td3": td3._make_agent}


@pytest.mark.parametrize("name", FACTORIES)
@pytest.mark.parametrize("strict", [True, False])
def test_scheduler_checkpoint_and_legacy_missing_state(name, strict, tmp_path):
    def build():
        return FACTORIES[name](lr_schedule="linear_warmup", lr_warmup_steps=20)

    agent = build()
    if name == "td3":
        for step in range(8):
            td3._add_transition(agent, float(step))
    else:
        {"edac": edac, "idql": idql}[name]._fill(agent)
    agent.train(4)
    path = agent.save(tmp_path / "checkpoint.pt", include_replay_buffer=False)
    saved = torch.load(path, weights_only=False)
    states = saved["state"]["extra"]["lr_scheduler_states"]
    assert all(state["last_epoch"] > 0 for state in states)
    restored = build()
    restored.load(path, strict=strict, load_replay_buffer=False)
    assert [s.state_dict() for s in restored._lr_schedulers] == states
    for scheduler in restored._lr_schedulers:
        assert scheduler.get_last_lr() == [group["lr"] for group in scheduler.optimizer.param_groups]

    del saved["state"]["extra"]["lr_scheduler_states"]
    legacy = tmp_path / "legacy.pt"
    torch.save(saved, legacy)
    restored = build()
    fresh = copy.deepcopy([s.state_dict() for s in restored._lr_schedulers])
    restored.load(legacy, strict=strict, load_replay_buffer=False)
    assert [s.state_dict() for s in restored._lr_schedulers] == fresh
    # Missing scheduler state leaves the scheduler fresh; optimizer loading is unchanged.
    for name, optimizer in restored._optimizer_state_dicts().items():
        torch.testing.assert_close(optimizer, saved["state"]["optimizers"][name], rtol=0, atol=0)
