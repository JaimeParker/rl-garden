"""``WorldModel`` contract additions: ``reward_value`` (real and symlog
space), ``SequenceBatch.reward_valid`` filtering of the reward loss, and
``BoundTaskWorldModel`` driving ``imagine()``/``MPPIPlanner.plan()``."""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.dreamer_conv import DreamerConvEncoder
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.networks.symlog import symexp, symlog
from rl_garden.networks.twohot import twohot_mean
from rl_garden.observations import ObservationSchema
from rl_garden.planners.mppi import MPPIPlanner, PlannerConfig
from rl_garden.world_models import (
    RSSM,
    BoundTaskWorldModel,
    LatentConsistencyModel,
    MultitaskWorldModel,
    SequenceBatch,
    imagine,
)
from rl_garden.world_models.rssm import RSSMSize

_OBS_DIM, _ACTION_DIM, _LATENT = 4, 2, 8


def _lcm() -> LatentConsistencyModel:
    # Default nn.Linear init (no apply_init): the reward head is not
    # zero-initialized, so decoded rewards are non-trivial.
    encoder = FlattenExtractor(spaces.Box(low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32))
    return LatentConsistencyModel(encoder=encoder, action_dim=_ACTION_DIM, latent_dim=_LATENT, mlp_dim=8, num_bins=11)


def _mtwm() -> MultitaskWorldModel:
    return MultitaskWorldModel(
        num_tasks=2, obs_dims=[4, 3], action_dims=[2, 1], task_dim=4, latent_dim=_LATENT, enc_dim=8, mlp_dim=8,
        num_bins=11,
    )


def _rssm(action_dim: int = 3) -> RSSM:
    obs_space = spaces.Dict({"state": spaces.Box(-1, 1, shape=(6,), dtype=np.float32)})
    schema = ObservationSchema.from_space(obs_space)
    size = RSSMSize(deter=32, hidden=16, discrete=4, units=16, cnn_depth=4)
    encoder = DreamerConvEncoder(obs_space, schema, units=size.units, state_layers=1)
    rssm = RSSM(encoder, schema, action_dim=action_dim, size=size, stoch=4, decoder_layers=1)
    # reward_head's last layer starts at zero (outscale=0); perturb it so the
    # decoded reward is not identically zero.
    with torch.no_grad():
        rssm.reward_head.last.weight.normal_(std=0.5)
    return rssm


# ---------------------------------------------------------------------------
# reward_value
# ---------------------------------------------------------------------------


def test_lcm_reward_value_shapes_and_symlog_relation():
    torch.manual_seed(0)
    model = _lcm()
    for lead in [(5,), (3, 5)]:
        state = {"z": torch.randn(*lead, _LATENT)}
        action = torch.rand(*lead, _ACTION_DIM) * 2 - 1
        real = model.reward_value(state, action)
        in_symlog = model.reward_value(state, action, symlog_space=True)
        assert real.shape == (*lead, 1)
        assert in_symlog.shape == (*lead, 1)
        torch.testing.assert_close(symexp(in_symlog), real)


def test_mtwm_reward_value_matches_bound_task_model():
    torch.manual_seed(0)
    model = _mtwm()
    bound = BoundTaskWorldModel(model, task_idx=1)
    state = {"z": torch.randn(5, _LATENT)}
    action = torch.rand(5, _ACTION_DIM) * 2 - 1
    task = torch.full((5,), 1)
    for symlog_space in (False, True):
        torch.testing.assert_close(
            bound.reward_value(state, action, symlog_space=symlog_space),
            model.reward_value(state, action, symlog_space=symlog_space, task=task),
        )


def test_rssm_reward_value_shapes_and_symlog_space():
    torch.manual_seed(0)
    rssm = _rssm()
    state = rssm.initial_state(5, torch.device("cpu"))
    state["deter"] = torch.randn_like(state["deter"])
    real = rssm.reward_value(state)
    in_symlog = rssm.reward_value(state, symlog_space=True)
    assert real.shape == (5, 1)
    assert in_symlog.shape == (5, 1)
    # Real space is exactly the reward head's TwoHotDist mode.
    torch.testing.assert_close(real, rssm.reward_head(rssm.features(state)).mode, rtol=0, atol=0)
    # Symlog space averages the same probabilities over symlog(bins).
    logits = rssm.reward(state, None)
    expected = twohot_mean(logits, symlog(rssm.reward_head._bins))
    torch.testing.assert_close(in_symlog, expected)


# ---------------------------------------------------------------------------
# reward_valid
# ---------------------------------------------------------------------------


def _scramble_invalid_columns(reward: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    scrambled = reward.clone()
    scrambled[:, ~valid] = torch.randn_like(scrambled[:, ~valid]) * 5
    return scrambled


def test_lcm_reward_loss_only_depends_on_valid_columns():
    torch.manual_seed(0)
    model = _lcm()
    horizon, batch = 3, 4
    obs = torch.randn(horizon + 1, batch, _OBS_DIM)
    action = torch.rand(horizon, batch, _ACTION_DIM) * 2 - 1
    reward = torch.randn(horizon, batch, 1)
    valid = torch.tensor([True, False, True, False])

    losses, _ = model.model_loss(SequenceBatch(obs=obs, action=action, reward=reward, reward_valid=valid))
    losses_scrambled, _ = model.model_loss(
        SequenceBatch(obs=obs, action=action, reward=_scramble_invalid_columns(reward, valid), reward_valid=valid)
    )
    losses_all, _ = model.model_loss(SequenceBatch(obs=obs, action=action, reward=reward))

    torch.testing.assert_close(losses["reward_loss"], losses_scrambled["reward_loss"])
    torch.testing.assert_close(losses["consistency_loss"], losses_all["consistency_loss"])
    assert not torch.allclose(losses["reward_loss"], losses_all["reward_loss"])


def test_lcm_model_loss_computes_next_z_when_missing():
    torch.manual_seed(0)
    model = _lcm()
    horizon, batch = 3, 4
    obs = torch.randn(horizon + 1, batch, _OBS_DIM)
    action = torch.rand(horizon, batch, _ACTION_DIM) * 2 - 1
    reward = torch.randn(horizon, batch, 1)
    with torch.no_grad():
        next_z = model.encode(obs[1:].reshape(-1, _OBS_DIM)).reshape(horizon, batch, -1)

    implicit, _ = model.model_loss(SequenceBatch(obs=obs, action=action, reward=reward))
    explicit, _ = model.model_loss(SequenceBatch(obs=obs, action=action, reward=reward, next_z=next_z))
    for key in implicit:
        torch.testing.assert_close(implicit[key], explicit[key])


def test_mtwm_reward_loss_only_depends_on_valid_columns():
    torch.manual_seed(0)
    model = _mtwm()
    horizon, batch = 2, 4
    obs = torch.randn(horizon + 1, batch, model.obs_dim)
    action = torch.rand(horizon, batch, model.action_dim) * 2 - 1
    reward = torch.randn(horizon, batch, 1)
    task = torch.tensor([0, 1, 0, 1])
    valid = torch.tensor([False, True, True, False])

    losses, _ = model.model_loss(SequenceBatch(obs=obs, action=action, reward=reward, task=task, reward_valid=valid))
    losses_scrambled, _ = model.model_loss(
        SequenceBatch(
            obs=obs, action=action, reward=_scramble_invalid_columns(reward, valid), task=task, reward_valid=valid
        )
    )
    torch.testing.assert_close(losses["reward_loss"], losses_scrambled["reward_loss"])


def test_rssm_reward_loss_only_depends_on_valid_columns():
    action_dim, horizon, batch = 3, 3, 4
    torch.manual_seed(0)
    rssm = _rssm(action_dim)
    obs = {"state": torch.randn(horizon + 1, batch, 6)}
    action = torch.rand(horizon, batch, action_dim) * 2 - 1
    reward = torch.randn(horizon, batch)
    valid = torch.tensor([True, True, False, False])
    common = {
        "obs": obs,
        "action": action,
        "is_first": torch.zeros(horizon, batch, dtype=torch.bool),
        "terminated": torch.zeros(horizon, batch, dtype=torch.bool),
        "carry": {"deter": torch.zeros(batch, 32), "stoch": torch.zeros(batch, rssm.flat_stoch)},
        "reward_valid": valid,
    }

    scrambled = _scramble_invalid_columns(reward, valid)
    torch.manual_seed(1)  # observe() samples the stochastic state
    losses, _ = rssm.model_loss(SequenceBatch(reward=reward, **common))
    torch.manual_seed(1)
    losses_scrambled, _ = rssm.model_loss(SequenceBatch(reward=scrambled, **common))
    torch.testing.assert_close(losses["rew"], losses_scrambled["rew"])
    torch.testing.assert_close(losses["dyn"], losses_scrambled["dyn"])


# ---------------------------------------------------------------------------
# BoundTaskWorldModel
# ---------------------------------------------------------------------------


def test_bound_task_model_forwards_task_to_multitask_model():
    torch.manual_seed(0)
    model = _mtwm()
    bound = BoundTaskWorldModel(model, task_idx=0)
    assert bound.latent_dim == model.latent_dim == bound.features_dim
    obs = torch.randn(5, model.obs_dim)
    task = torch.zeros(5, dtype=torch.long)
    torch.testing.assert_close(bound.encode(obs), model.encode(obs, task))
    window = torch.randn(3, 5, model.obs_dim)
    torch.testing.assert_close(bound.encode(window), model.encode(window, task))

    state = {"z": bound.encode(obs)}
    action = torch.rand(5, model.action_dim) * 2 - 1
    torch.testing.assert_close(bound.step(state, action)["z"], model.step(state, action, task=task)["z"])
    torch.testing.assert_close(bound.reward(state, action), model.reward(state, action, task=task))

    horizon = 2
    batch = SequenceBatch(
        obs=torch.randn(horizon + 1, 5, model.obs_dim),
        action=torch.rand(horizon, 5, model.action_dim) * 2 - 1,
        reward=torch.randn(horizon, 5, 1),
    )
    bound_losses, _ = bound.model_loss(batch)
    direct_losses, _ = model.model_loss(SequenceBatch(**{**batch.__dict__, "task": task}))
    for key in direct_losses:
        torch.testing.assert_close(bound_losses[key], direct_losses[key])
    assert batch.task is None  # the caller's batch is not mutated


def test_bound_task_model_runs_imagine_with_gradient():
    torch.manual_seed(0)
    model = _mtwm()
    bound = BoundTaskWorldModel(model, task_idx=1)
    start = {"z": bound.encode(torch.randn(6, model.obs_dim))}

    def policy_fn(state):
        return torch.tanh(state["z"][..., : model.action_dim])

    traj = imagine(bound, policy_fn, start, horizon=3, grad=True)
    assert traj.states["z"].shape == (4, 6, _LATENT)
    assert traj.actions.shape == (3, 6, model.action_dim)
    rewards = bound.reward_value({"z": traj.states["z"][:-1]}, traj.actions)
    assert rewards.shape == (3, 6, 1)
    rewards.sum().backward()
    assert model._dynamics[0].weight.grad is not None


def test_bound_task_model_runs_mppi_planner():
    torch.manual_seed(0)
    model = _mtwm()
    bound = BoundTaskWorldModel(model, task_idx=0)
    cfg = PlannerConfig(
        action_dim=model.action_dim, discount=0.9, horizon=3, num_samples=16, num_elites=4, num_pi_trajs=2,
        iterations=2,
    )
    with torch.no_grad():
        embed = bound.encode(torch.randn(1, model.obs_dim))
    state = bound.observe(None, None, embed, torch.tensor([True]))

    def policy_prior(state):
        return torch.tanh(state["z"][..., : model.action_dim])

    def value_fn(state, action):
        return state["z"].sum(-1, keepdim=True) + action.sum(-1, keepdim=True)

    action, prev_mean = MPPIPlanner(cfg).plan(
        bound, state, policy_prior=policy_prior, value_fn=value_fn, prev_mean=None, t0=True
    )
    assert action.shape == (model.action_dim,)
    assert torch.all(action.abs() <= 1.0)
    assert prev_mean.shape == (cfg.horizon, model.action_dim)
