"""NCRL refills its unsaved online replay on every load."""
import copy
import math

import pytest
import torch
import test_ncrl_smoke as smoke
from test_resume_contract import differences

from rl_garden.algorithms.ncrl import _ACTION_FROM_ACTOR

artifacts = smoke.artifacts
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("num_envs,train_ratio", [(1, 8.0), (2, 8.0), (2, 2.0)])
@pytest.mark.parametrize("online_ratio", [0.5, 1.0])
def test_resume_refills_again_after_mid_refill_save(
    artifacts, tmp_path, monkeypatch, num_envs, train_ratio, online_ratio
):
    def build():
        return smoke._agent(artifacts, num_envs=num_envs, train_ratio=train_ratio,
                      online_ratio=online_ratio, learning_starts=18, batch_size=2,
                      batch_length=4, bc_train_steps=2, warmup_frames=1000)

    fresh = build()
    stride = fresh.steps_per_env * fresh.num_envs
    first_update = math.ceil(fresh.learning_starts / stride) * stride
    fresh.learn(first_update + 2 * stride)
    assert fresh._global_step == first_update + 2 * stride
    assert fresh._global_update == 3 * fresh.grad_steps_per_iteration
    path = fresh.save(tmp_path / "initial.pt")
    start, updates = fresh._global_step, fresh._global_update

    def track_policy(agent):
        predict = agent.policy.predict
        rollout = agent._rollout_action
        predicted = []
        executed = []

        def recorded_predict(*args, **kwargs):
            action, state = predict(*args, **kwargs)
            predicted.append(action.float().clone())
            return action, state

        def recorded_rollout(*args, **kwargs):
            action, env_action, context = rollout(*args, **kwargs)
            assert torch.equal(action, predicted[-1])
            assert (context["action_source"] == _ACTION_FROM_ACTOR).all()
            executed.append(env_action.clone())
            return action, env_action, context

        monkeypatch.setattr(agent.policy, "predict", recorded_predict)
        monkeypatch.setattr(agent, "_rollout_action", recorded_rollout)
        return executed

    resumed = build()
    resumed.load(path, load_replay_buffer=False)
    actions = track_policy(resumed)
    before = copy.deepcopy(resumed.state_dict())
    partial = first_update - stride
    resumed.learn(start + partial)
    assert resumed._global_step == start + partial
    assert resumed._global_update == updates
    assert len(actions) == partial // num_envs
    for section in ("policy", "optimizers", "extra"):
        assert not differences(resumed.state_dict()[section], before[section], section)
    midway = resumed.save(tmp_path / "midway.pt")

    again = build()
    again.load(midway, load_replay_buffer=False)
    actions = track_policy(again)
    second_start = again._global_step
    again.learn(second_start + partial)
    assert again._global_step == second_start + partial
    assert again._global_update == updates
    assert len(actions) == partial // num_envs
    for section in ("policy", "optimizers", "extra"):
        assert not differences(again.state_dict()[section], before[section], section)

    # Keep checking actor actions through the threshold-crossing rollout.
    again.learn(second_start + first_update)
    assert again._global_step == second_start + first_update
    assert again._global_update == updates + again.grad_steps_per_iteration
    monkeypatch.undo()
    again.learn(second_start + first_update + 2 * stride)
    assert again._global_update == updates + 3 * again.grad_steps_per_iteration
