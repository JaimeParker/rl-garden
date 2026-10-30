"""CPU regression tests for online SAC bootstrap configuration and boundaries."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tyro
from gymnasium import spaces
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.sac_core import SACCore
from rl_garden.buffers.residual_h5 import _residual_done
from rl_garden.common.resolved_config import persist_resolved_config
from rl_garden.training.online.residual_sac import (
    ResidualSACArgs,
    build_residual_sac,
)
from rl_garden.training.online.sac import SACArgs, build_sac

MODES = ("always", "never", "truncated")
TERMINATED = [False, True, False, True]
TRUNCATED = [False, False, True, True]
STOP = {
    "always": [False, False, False, False],
    "never": [False, True, True, True],
    "truncated": [False, True, False, True],
}
FINAL = {
    "always": [False, True, True, True],
    "never": [False, True, True, True],
    "truncated": [False, False, True, False],
}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("args_type", [SACArgs, ResidualSACArgs])
def test_default_is_truncated_and_invalid_cli_rejected(args_type):
    assert tyro.cli(args_type, args=[]).bootstrap_at_done == "truncated"
    with pytest.raises(SystemExit):
        tyro.cli(args_type, args=["--bootstrap-at-done", "invalid"])


class StateEnv:
    num_envs = 4
    single_observation_space = spaces.Dict(
        {"state": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)}
    )
    single_action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def test_algorithm_and_h5_defaults_match_cli():
    import inspect
    from rl_garden.algorithms.sac import SAC
    from rl_garden.algorithms.ddpg import DDPG
    from rl_garden.algorithms.cql import _CQLRolloutTrainingShell
    from rl_garden.algorithms.wsrl import WSRL
    from rl_garden.buffers.residual_h5 import load_residual_h5_to_replay_buffer
    for entry in (OffPolicyAlgorithm, SAC, DDPG, _CQLRolloutTrainingShell, WSRL, load_residual_h5_to_replay_buffer):
        assert inspect.signature(entry).parameters['bootstrap_at_done'].default == 'truncated'


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "args_type,builder", [(SACArgs, build_sac), (ResidualSACArgs, build_residual_sac)]
)
def test_cli_builder_agent_and_saved_config(mode, args_type, builder, tmp_path):
    args = tyro.cli(args_type, args=["--bootstrap-at-done", mode])
    args.obs_mode = "state"
    args.buffer_device = "cpu"
    args.buffer_size = 32
    args.batch_size = 4
    args.hidden_dim = 8
    args.actor_hidden_layers = 1
    args.critic_hidden_layers = 1
    args.eval_freq = 0
    args.std_log = False
    if isinstance(args, ResidualSACArgs):
        args.base_policy = "zero"
    agent = builder(args, StateEnv(), None, None, None)
    assert agent.device.type == "cpu"
    assert agent.bootstrap_at_done == mode
    assert agent._checkpoint_metadata()["bootstrap_at_done"] == mode
    config = persist_resolved_config(
        args,
        training_phase="online",
        algorithm="sac",
        run_name="test",
        log_dir=str(tmp_path),
    )
    saved = json.loads((tmp_path / "test" / "config.json").read_text())
    assert (
        config["args"]["bootstrap_at_done"]
        == saved["args"]["bootstrap_at_done"]
        == mode
    )
    _, stop = agent._compute_done_masks(
        torch.tensor(TERMINATED), torch.tensor(TRUNCATED)
    )
    obs = {"state": torch.zeros(4, 3)}
    actions = torch.zeros(4, 2)
    extras = {}
    if isinstance(args, ResidualSACArgs):
        extras = {"base_actions": actions, "next_base_actions": actions}
    agent.replay_buffer.add(obs, obs, actions, torch.ones(4), stop.float(), **extras)
    metrics = agent.train(1, compute_info=True)
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))
    checkpoint_path = tmp_path / "model.pt"
    agent.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["metadata"]["hyperparameters"]["bootstrap_at_done"] == mode


@pytest.mark.parametrize("mode", MODES)
def test_boundary_masks_and_actual_bellman_target(mode):
    owner = SimpleNamespace(bootstrap_at_done=mode)
    final, stop = OffPolicyAlgorithm._compute_done_masks(
        owner, torch.tensor(TERMINATED), torch.tensor(TRUNCATED)
    )
    assert final.tolist() == FINAL[mode]
    assert stop.tolist() == STOP[mode]
    # Exercise production SACCore rather than reimplementing the target formula.
    owner.gamma = 0.99
    owner._current_alpha = lambda: torch.tensor(2.0)
    owner._target_action_log_prob = lambda data: (
        torch.zeros(4, 2),
        torch.full((4, 1), -0.5),
        torch.zeros(4, 3),
    )
    owner._target_critic_subsample_size = lambda: None
    owner._backup_entropy_enabled = lambda: True
    owner.policy = SimpleNamespace(
        min_q_value=lambda *a, **kw: torch.full((4, 1), 10.0)
    )
    owner._target_discounts = lambda data: SACCore._target_discounts(owner, data)
    data = SimpleNamespace(rewards=torch.ones(4), dones=stop.float())
    target = SACCore._target_q(owner, data).flatten()
    expected = torch.tensor([1.0 if ended else 11.89 for ended in STOP[mode]])
    torch.testing.assert_close(target, expected)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dict_obs", [False, True])
def test_partial_vector_final_obs_does_not_replace_continuing_rows(mode, dict_obs):
    owner = SimpleNamespace(bootstrap_at_done=mode)
    mask, _ = OffPolicyAlgorithm._compute_done_masks(
        owner, torch.tensor(TERMINATED), torch.tensor(TRUNCATED)
    )
    next_obs = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    # Non-ended final slots may be invalid on other vector backends.
    final_obs = torch.tensor([[float("nan")], [21.0], [31.0], [41.0]])
    expected = next_obs.clone()
    expected[torch.tensor(FINAL[mode])] = final_obs[torch.tensor(FINAL[mode])]
    actual = {"state": next_obs.clone()} if dict_obs else next_obs.clone()
    info = {"final_observation": {"state": final_obs} if dict_obs else final_obs}
    OffPolicyAlgorithm._write_final_obs(actual, info, mask)
    torch.testing.assert_close(actual["state"] if dict_obs else actual, expected)
    assert torch.isfinite(actual["state"] if dict_obs else actual).all()


@pytest.mark.parametrize("mode", MODES)
def test_explicit_offline_flags_match_online_stop_mask(mode):
    traj = {"terminated": TERMINATED, "truncated": TRUNCATED}
    done = _residual_done(traj, 4, torch.device("cpu"), bootstrap_at_done=mode)
    assert done.tolist() == [float(x) for x in STOP[mode]]


@pytest.mark.parametrize("mode", MODES)
def test_legacy_offline_done_only_fallback_is_not_reinterpreted(mode):
    # Existing conservative fallback lacks terminated/truncated distinction.
    # Exposing the online option must not silently relabel old offline data.
    done = _residual_done(
        {"done": [False, True]}, 2, torch.device("cpu"), bootstrap_at_done=mode
    )
    assert done.tolist() == [0.0, 1.0]
