from __future__ import annotations

import warnings

import pytest
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.common.alpha_tuning import AlphaTuner, parse_auto_alpha_init, softplus_inverse
from rl_garden.common.checkpoint import load_checkpoint_file, save_checkpoint_file


class DummyVecEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _agent(**kwargs) -> SAC:
    params = {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": 8,
        "learning_starts": 0,
        "training_freq": 1,
        "eval_freq": 0,
        "net_arch": {"pi": [16], "qf": [16]},
    }
    params.update(kwargs)
    return SAC(env=DummyVecEnv(), **params)


def _fill(agent: SAC, steps: int = 8) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_parse_auto_alpha_init():
    assert parse_auto_alpha_init("auto") == (True, 1.0)
    assert parse_auto_alpha_init("auto_0.25") == (True, 0.25)
    assert parse_auto_alpha_init(0.5) == (False, 0.5)


def test_softplus_inverse_large_init_does_not_overflow():
    # Plain log(expm1(x)) overflows to inf for x >= ~89 in float32.
    assert np.isfinite(softplus_inverse(100.0))
    assert np.isfinite(softplus_inverse(1000.0))

    tuner = AlphaTuner("lagrange_softplus", init_value=100.0)
    assert torch.isfinite(tuner.raw_alpha).all()
    torch.testing.assert_close(tuner.current_alpha(), torch.tensor([100.0]))


def test_alpha_tuning_losses_match_formulas():
    log_prob = torch.full((4, 1), -1.0)
    target_entropy = -2.0

    legacy = AlphaTuner("legacy_exp", init_value=2.0)
    loss = legacy.loss(log_prob, target_entropy)
    assert torch.allclose(loss, torch.tensor(6.0))
    loss.backward()
    assert torch.allclose(legacy.log_alpha.grad, torch.tensor([6.0]))

    sb3 = AlphaTuner("log_alpha", init_value=2.0)
    loss = sb3.loss(log_prob, target_entropy)
    assert torch.allclose(loss, torch.log(torch.tensor(2.0)) * 3.0)
    loss.backward()
    assert torch.allclose(sb3.log_alpha.grad, torch.tensor([3.0]))

    wsrl = AlphaTuner("lagrange_softplus", init_value=2.0)
    loss = wsrl.loss(log_prob, target_entropy)
    assert torch.allclose(loss, torch.tensor(6.0), atol=1e-6)
    loss.backward()
    assert wsrl.raw_alpha.grad is not None
    assert 0.0 < float(wsrl.raw_alpha.grad.item()) < 3.0


@pytest.mark.parametrize("mode", ["legacy_exp", "log_alpha", "lagrange_softplus"])
def test_sac_alpha_tuning_modes_train_one_step(mode: str):
    agent = _agent(alpha_tuning=mode)
    _fill(agent)

    info = agent.train(gradient_steps=1, compute_info=True)

    assert torch.isfinite(torch.tensor(info["alpha"]))
    assert torch.isfinite(torch.tensor(info["alpha_loss"]))


def test_fixed_ent_coef_does_not_create_alpha_optimizer():
    agent = _agent(ent_coef=0.2, alpha_tuning="log_alpha")

    assert agent.autotune is False
    assert agent.alpha_optimizer is None
    assert torch.allclose(agent._current_alpha(), torch.tensor(0.2))


def test_invalid_alpha_tuning_rejected_with_fixed_ent_coef():
    with pytest.raises(ValueError, match="Unknown alpha_tuning"):
        _agent(ent_coef=0.2, alpha_tuning="bad")


def test_checkpoint_alpha_tuning_mismatch_warns_and_reinitializes(tmp_path):
    source = _agent(alpha_tuning="legacy_exp", ent_coef="auto_0.5")
    with torch.no_grad():
        source.log_alpha.fill_(torch.log(torch.tensor(0.25)))
    ckpt = tmp_path / "sac.pt"
    source.save(ckpt)

    target = _agent(alpha_tuning="log_alpha", ent_coef="auto_0.5")
    with pytest.warns(RuntimeWarning, match="reinitializing alpha"):
        target.load(ckpt, load_replay_buffer=False, load_optimizers=True)

    assert torch.allclose(target._current_alpha(), torch.tensor([0.5]))


def test_legacy_log_alpha_only_checkpoint_restores_under_legacy_exp(tmp_path):
    """Pre-AlphaTuner checkpoints stored only ``log_alpha`` (no ``alpha_tuner``/
    ``alpha_tuning`` keys). Loading such a checkpoint under the default
    ``alpha_tuning="legacy_exp"`` must restore alpha and the alpha optimizer
    without warnings.
    """
    source = _agent(alpha_tuning="legacy_exp", ent_coef="auto_0.5")
    with torch.no_grad():
        source.log_alpha.fill_(torch.log(torch.tensor(0.25)))
    ckpt = tmp_path / "sac.pt"
    source.save(ckpt)

    raw = load_checkpoint_file(ckpt, map_location="cpu")
    raw["state"]["extra"].pop("alpha_tuner")
    raw["state"]["extra"].pop("alpha_tuning")
    save_checkpoint_file(ckpt, raw)

    target = _agent(alpha_tuning="legacy_exp", ent_coef="auto_0.5")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        target.load(ckpt, load_replay_buffer=False, load_optimizers=True)

    assert target._skip_alpha_optimizer_load is False
    assert torch.allclose(target._current_alpha(), source._current_alpha())
    assert torch.allclose(
        torch.tensor(target.alpha_optimizer.state_dict()["param_groups"][0]["lr"]),
        torch.tensor(source.alpha_optimizer.state_dict()["param_groups"][0]["lr"]),
    )
