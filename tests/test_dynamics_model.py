from __future__ import annotations

import torch

from rl_garden.models.dynamics import EnsembleDynamicsModel, get_termination_fn, rollout_q_mean, train_ensemble
from rl_garden.models.dynamics.network import EnsembleLinear
from rl_garden.models.dynamics.termination_fns import (
    termination_fn_halfcheetah,
    termination_fn_hopper,
    termination_fn_walker2d,
)


def test_ensemble_linear_matches_manual_per_member_forward():
    torch.manual_seed(0)
    num_ensemble, in_dim, out_dim = 4, 3, 5
    layer = EnsembleLinear(in_dim, out_dim, num_ensemble)
    x = torch.randn(num_ensemble, 7, in_dim)

    out = layer(x)
    for m in range(num_ensemble):
        expected = x[m] @ layer.weight[m] + layer.bias[m]
        assert torch.allclose(out[m], expected, atol=1e-6)


def test_ensemble_linear_2d_input_broadcasts_to_all_members():
    torch.manual_seed(0)
    layer = EnsembleLinear(3, 5, num_ensemble=4)
    x = torch.randn(7, 3)
    out = layer(x)
    assert out.shape == (4, 7, 5)
    for m in range(4):
        expected = x @ layer.weight[m] + layer.bias[m]
        assert torch.allclose(out[m], expected, atol=1e-6)


def test_ensemble_linear_load_save_snapshot_rollback():
    layer = EnsembleLinear(3, 3, num_ensemble=2)
    original = layer.weight.data.clone()
    with torch.no_grad():
        layer.weight.add_(1.0)
    layer.update_save([0])  # only member 0's change is snapshotted
    with torch.no_grad():
        layer.weight.add_(1.0)  # both members drift further
    layer.load_save()  # rolls back to the snapshot: member 0 at +1, member 1 at original
    assert torch.allclose(layer.weight[0], original[0] + 1.0)
    assert torch.allclose(layer.weight[1], original[1])


def test_soft_clamp_bounds_logvar():
    from rl_garden.models.dynamics.network import soft_clamp

    x = torch.tensor([-100.0, 0.0, 100.0])
    lo, hi = torch.tensor(-10.0), torch.tensor(0.5)
    out = soft_clamp(x, lo, hi)
    assert (out > lo - 1e-3).all()
    assert (out < hi + 1e-3).all()


def test_ensemble_dynamics_model_forward_shape():
    model = EnsembleDynamicsModel(4, 2, [8, 8], num_ensemble=3, num_elites=2)
    x = torch.randn(10, 4 + 2)
    mean, logvar = model(x)
    assert mean.shape == (3, 10, 4 + 1)  # obs_dim + reward
    assert logvar.shape == mean.shape


def test_set_elites_and_random_elite_member():
    model = EnsembleDynamicsModel(4, 2, [8], num_ensemble=5, num_elites=2)
    model.set_elites([1, 3])
    idx = model.random_elite_member(1000)
    assert set(idx.tolist()) <= {1, 3}


# --- termination_fns ---


def test_termination_fn_halfcheetah_matches_reference_bounds():
    obs = torch.zeros(2, 4)
    action = torch.zeros(2, 2)
    in_bounds = torch.full((2, 4), 5.0)
    out_of_bounds = torch.tensor([[200.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert not termination_fn_halfcheetah(obs, action, in_bounds).any()
    assert termination_fn_halfcheetah(obs, action, out_of_bounds)[0].item()
    assert not termination_fn_halfcheetah(obs, action, out_of_bounds)[1].item()


def test_termination_fn_hopper_healthy_vs_unhealthy():
    obs = torch.zeros(2, 4)
    action = torch.zeros(2, 2)
    healthy = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    unhealthy_height = torch.tensor([[0.5, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    unhealthy_angle = torch.tensor([[1.0, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert not termination_fn_hopper(obs, action, healthy).any()
    assert termination_fn_hopper(obs, action, unhealthy_height)[0].item()
    assert termination_fn_hopper(obs, action, unhealthy_angle)[0].item()


def test_termination_fn_walker2d_healthy_vs_unhealthy():
    obs = torch.zeros(2, 4)
    action = torch.zeros(2, 2)
    healthy = torch.tensor([[1.2, 0.0, 0.0, 0.0], [1.2, 0.0, 0.0, 0.0]])
    unhealthy_height = torch.tensor([[0.5, 0.0, 0.0, 0.0], [1.2, 0.0, 0.0, 0.0]])
    unhealthy_angle = torch.tensor([[1.2, 1.5, 0.0, 0.0], [1.2, 0.0, 0.0, 0.0]])
    assert not termination_fn_walker2d(obs, action, healthy).any()
    assert termination_fn_walker2d(obs, action, unhealthy_height)[0].item()
    assert termination_fn_walker2d(obs, action, unhealthy_angle)[0].item()


def test_get_termination_fn_dispatches_by_substring():
    assert get_termination_fn("halfcheetah-medium-v2") is termination_fn_halfcheetah
    assert get_termination_fn("hopper-medium-replay-v2") is termination_fn_hopper
    assert get_termination_fn("walker2d-random-v2") is termination_fn_walker2d


def test_get_termination_fn_rejects_unknown_task():
    import pytest

    with pytest.raises(ValueError, match="No termination_fn"):
        get_termination_fn("antmaze-medium-diverse-v2")


# --- trainer ---


def _synthetic_transitions(n: int = 400, obs_dim: int = 4, action_dim: int = 2):
    torch.manual_seed(0)
    obs = torch.randn(n, obs_dim)
    actions = torch.rand(n, action_dim) * 2 - 1
    next_obs = obs + 0.1 * actions.sum(-1, keepdim=True)  # trivially fittable
    rewards = actions.sum(-1)
    return obs, actions, next_obs, rewards


def test_train_ensemble_early_stops_within_budget_and_fits_trivial_data():
    obs, actions, next_obs, rewards = _synthetic_transitions()
    model = EnsembleDynamicsModel(4, 2, [16, 16], num_ensemble=2, num_elites=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    metrics, input_mean, input_std = train_ensemble(
        model,
        optimizer,
        obs,
        actions,
        next_obs,
        rewards,
        max_epochs_since_update=3,
        max_epochs=50,
        batch_size=64,
    )
    assert metrics["dynamics_epochs"] <= 50
    assert input_mean.shape == (1, 4 + 2)
    assert input_std.shape == (1, 4 + 2)
    assert model.elites.numel() == 1


def test_rollout_q_mean_early_breaks_when_all_rows_terminate():
    obs, actions, next_obs, rewards = _synthetic_transitions()
    model = EnsembleDynamicsModel(4, 2, [8, 8], num_ensemble=2, num_elites=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    _, input_mean, input_std = train_ensemble(
        model, optimizer, obs, actions, next_obs, rewards, max_epochs=2
    )

    class _DeterministicPolicy:
        def predict(self, obs, deterministic=True):
            return torch.zeros(obs.shape[0], 2)

    class _ZeroQ:
        def __call__(self, obs, action):
            return torch.ones(obs.shape[0], 1)

    def _always_terminate(obs, action, next_obs):
        return torch.ones(obs.shape[0], 1, dtype=torch.bool)

    # rollout_length=1000 requested, but every row terminates on step 1 --
    # must return after collecting exactly one step's Q values (all == 1.0),
    # not hang or loop 1000 times.
    score = rollout_q_mean(
        _DeterministicPolicy(),
        _ZeroQ(),
        model,
        _always_terminate,
        obs[:16],
        1000,
        input_mean=input_mean,
        input_std=input_std,
    )
    assert score == 1.0
