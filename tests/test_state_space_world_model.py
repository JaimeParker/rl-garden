"""``StateSpaceWorldModel``, ``RunningNorm`` and the PWM return normalizer."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from _supermimic_fakes import PointMassCodec, PointMassMDP

from rl_garden.algorithms.pwm import PWMPolicy
from rl_garden.networks.running_norm import RunningNorm
from rl_garden.world_models import (
    ResidualStateCodec,
    SequenceBatch,
    StateSpaceWorldModel,
)

_S, _A, _C, _G, _K, _B = 4, 2, 3, 5, 2, 3


def _model(**overrides) -> StateSpaceWorldModel:
    kwargs = {
        "state_dim": _S,
        "action_dim": _A,
        "obs_dim": _S,
        "ctx_dim": _C,
        "goal_dim": _G,
        "hidden_dims": (16, 16),
        "head_hidden_dims": (8,),
        "observation_head": "learned",
    }
    kwargs.update(overrides)
    return StateSpaceWorldModel(**kwargs)


def _randomize(model: StateSpaceWorldModel) -> None:
    """Non-trivial weights and normalizer statistics (the init is all-zero)."""
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.3)
        for norm in (model.in_norm, model.out_norm, model.goal_norm):
            norm.mean.normal_()
            norm.var.uniform_(0.5, 2.0)


def _window(extra: bool = True) -> SequenceBatch:
    obs = {
        "state": torch.randn(_K + 1, _B, _S),
        "state_ctx": torch.randn(_K + 1, _B, _C),
        "state_goal": torch.randn(_K + 1, _B, _G),
    }
    return SequenceBatch(
        obs=obs,
        action=torch.randn(_K, _B, _A),
        reward=torch.randn(_K, _B),
        terminated=(torch.rand(_K, _B) > 0.5).float(),
    )


def test_construction_validation():
    with pytest.raises(ValueError, match="analytic"):
        _model(reward_head="analytic")
    with pytest.raises(ValueError, match="analytic"):
        _model(termination_head="analytic")
    with pytest.raises(ValueError, match="identity"):
        _model(observation_head="identity", obs_dim=_S + 1)


def test_state_from_obs_defaults():
    model = _model()
    obs = {"state": torch.randn(_B, _S)}
    state = model.state_from_obs(obs)
    torch.testing.assert_close(state["s"], obs["state"])
    assert state["ctx"].shape == (_B, 0) and state["goal"].shape == (_B, 0)
    assert state["t"].dtype == torch.long and torch.equal(state["t"], torch.zeros(_B, dtype=torch.long))
    torch.testing.assert_close(model.encode(obs), obs["state"])
    obs["state_sim"] = torch.randn(_B, _S)
    obs["state_time"] = torch.tensor([[2.0], [3.0], [4.0]])
    state = model.state_from_obs(obs)
    torch.testing.assert_close(state["s"], obs["state_sim"])
    assert torch.equal(state["t"], torch.tensor([2, 3, 4]))


def test_step_is_identity_at_init_and_advances_time():
    model = _model()
    state = {"s": torch.randn(_B, _S), "ctx": torch.randn(_B, _C), "goal": torch.randn(_B, _G), "t": torch.arange(_B)}
    nxt = model.step(state, torch.randn(_B, _A))
    torch.testing.assert_close(nxt["s"], state["s"])
    assert torch.equal(nxt["t"], state["t"] + 1)
    torch.testing.assert_close(nxt["goal"], state["goal"])
    torch.testing.assert_close(nxt["ctx"], state["ctx"])


def test_model_loss_matches_hand_computation():
    torch.manual_seed(0)
    model = _model(codec=ResidualStateCodec(_S))
    _randomize(model)
    batch = _window()
    losses, posterior = model.model_loss(batch)

    s_real, ctx, goal = batch.obs["state"], batch.obs["state_ctx"][0], batch.obs["state_goal"]
    scale = model.out_norm.std().clamp_min(model.out_norm_eps)

    def head_in(s, a, g, with_action):
        normed = model.in_norm(torch.cat([s, a, ctx], -1))
        feat, act, c = normed[:, :_S], normed[:, _S : _S + _A], normed[:, _S + _A :]
        return torch.cat([feat] + ([act] if with_action else []) + [c, model.goal_norm(g)], -1)

    s_hat = s_real[0]
    state_term = reward_term = obs_term = term_term = 0.0
    for j in range(1, _K + 1):
        a = batch.action[j - 1]
        dyn_in = model.in_norm(torch.cat([s_hat, a, ctx], -1))
        s_prev, s_hat = s_hat, s_hat + model.dynamics_mlp(dyn_in) * torch.sqrt(model.out_norm.var + model.out_norm.eps**2) + model.out_norm.mean
        state_term = state_term + ((s_hat - s_real[j]) / scale).square().mean(-1)
        reward_pred = model.reward_mlp(head_in(s_prev, a, goal[j - 1], True)).squeeze(-1)
        reward_term = reward_term + (reward_pred - batch.reward[j - 1]).square()
        obs_pred = model.observation_mlp(head_in(s_hat, a, goal[j], False))
        obs_term = obs_term + (obs_pred - s_real[j]).square().mean(-1)
        logit = model.termination_mlp(head_in(s_hat, a, goal[j], False)).squeeze(-1)
        term_term = term_term + F.binary_cross_entropy_with_logits(logit, batch.terminated[j - 1], reduction="none")
    expected = {"state": state_term, "reward": reward_term, "observation": obs_term, "termination": term_term}
    for name, value in expected.items():
        torch.testing.assert_close(losses[name], value.mean(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(losses["total"], sum(v.mean() for v in expected.values()), rtol=1e-4, atol=1e-5)
    assert torch.equal(posterior["s"], s_real[0])
    assert posterior["t"].shape == (_B,)


def test_prev_action_shifts_by_one_for_analytic_reward_target():
    model = _model(reward_head="analytic", mdp=PointMassMDP(), state_dim=4, obs_dim=4, goal_dim=4, ctx_dim=0)
    seen = []
    original = model.mdp.reward
    model.mdp.reward = lambda s, a, s1, prev, ctx, t: (seen.append(prev), original(s, a, s1, prev, ctx, t))[1]
    state = model.initial_state(2, torch.device("cpu"))
    action = torch.ones(2, _A)
    model.transition_reward(state, action, model.step(state, action), action * 2)
    assert torch.equal(seen[0], action * 2)


def test_gradient_flows_through_all_steps():
    torch.manual_seed(1)
    model = _model()
    _randomize(model)
    losses, _ = model.model_loss(_window())
    losses["total"].backward()
    assert model.dynamics_mlp[-1].bias.grad.abs().sum() > 0

    batch = _window()
    first_action = batch.action[0].clone().requires_grad_(True)
    state = {"s": batch.obs["state"][0], "ctx": batch.obs["state_ctx"][0], "goal": batch.obs["state_goal"][0],
             "t": torch.zeros(_B, dtype=torch.long)}
    first = model.step(state, first_action)
    last = model.step(first, batch.action[1])
    error = model.codec.increment(batch.obs["state"][2], last["s"]).square().sum()
    (grad,) = torch.autograd.grad(error, first["s"], retain_graph=True)
    assert grad.abs().sum() > 0
    (grad_first,) = torch.autograd.grad(error, first_action)
    assert grad_first.abs().sum() > 0


def test_analytic_heads_with_point_mass():
    torch.manual_seed(2)
    mdp = PointMassMDP()
    model = StateSpaceWorldModel(
        state_dim=4, action_dim=2, obs_dim=4, goal_dim=4, codec=PointMassCodec(), mdp=mdp,
        reward_head="analytic", observation_head="analytic", termination_head="analytic",
        hidden_dims=(16,),
    )
    assert list(model.parameter_groups()["heads"]) == []
    t = torch.tensor([0, 5, 62])
    state = {"s": torch.randn(3, 4), "ctx": torch.zeros(3, 0), "goal": mdp.goal(t), "t": t}
    action = torch.randn(3, 2)
    nxt = model.step(state, action)
    assert torch.equal(nxt["t"], t + 1)
    torch.testing.assert_close(nxt["goal"], mdp.goal(t + 1))
    reward = model.transition_reward(state, action, nxt, torch.zeros(3, 2))
    torch.testing.assert_close(reward, mdp.reward(state["s"], action, nxt["s"], torch.zeros(3, 2), state["ctx"], t + 1).unsqueeze(-1))
    torch.testing.assert_close(model.termination_prob(state), mdp.termination_prob(state["s"], t).unsqueeze(-1))
    torch.testing.assert_close(model.continue_(state), 1 - model.termination_prob(state))
    torch.testing.assert_close(model.observation(state, action), state["s"])
    with pytest.raises(NotImplementedError):
        model.reward(state, action)
    with pytest.raises(ValueError):
        _model().reward_value(state, action, symlog_space=True)

    window = {"state": torch.randn(_K + 1, _B, 4), "state_time": torch.arange(_K + 1).repeat(_B, 1).T.float()}
    batch = SequenceBatch(obs=window, action=torch.randn(_K, _B, 2), reward=torch.zeros(_K, _B))
    losses, posterior = model.model_loss(batch)
    for name in ("reward", "observation", "termination"):
        assert losses[name] == 0
    assert losses["total"] == losses["state"]
    assert torch.equal(posterior["t"], torch.zeros(_B, dtype=torch.long))


def test_running_norm_merge_and_roundtrip():
    torch.manual_seed(3)
    x = torch.randn(40, 5) * 3 + 1
    merged, whole = RunningNorm((5,)), RunningNorm((5,))
    merged.update(x[:13])
    merged.update(x[13:])
    whole.update(x)
    torch.testing.assert_close(merged.mean, whole.mean, atol=1e-6, rtol=0)
    torch.testing.assert_close(merged.var, whole.var, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(merged.count, whole.count)
    torch.testing.assert_close(whole.denormalize(whole(x)), x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(whole.std(), whole.var.sqrt())
    leading = RunningNorm((5,))
    leading.update(x.reshape(4, 10, 5))
    torch.testing.assert_close(leading.mean, whole.mean)


def test_running_norm_scalar_matches_previous_pwm_formula():
    class OldReturnRunningMeanStd(torch.nn.Module):
        def __init__(self, epsilon: float = 1e-4) -> None:
            super().__init__()
            self.register_buffer("mean", torch.zeros(()))
            self.register_buffer("var", torch.ones(()))
            self.register_buffer("count", torch.tensor(float(epsilon)))

        def update(self, x):
            batch_mean = x.mean()
            batch_var = x.var(unbiased=False)
            batch_count = x.numel()
            delta = batch_mean - self.mean
            total = self.count + batch_count
            new_mean = self.mean + delta * batch_count / total
            m2 = self.var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
            self.mean.copy_(new_mean)
            self.var.copy_(m2 / total)
            self.count.copy_(total)

    torch.manual_seed(4)
    old, new = OldReturnRunningMeanStd(), RunningNorm(shape=(), epsilon=1e-4)
    for n in (7, 1, 32):
        x = torch.randn(n) * 5
        old.update(x)
        new.update(x)
        for name in ("mean", "var", "count"):
            assert torch.equal(getattr(old, name), getattr(new, name))
    assert set(new.state_dict()) == set(old.state_dict()) == {"mean", "var", "count"}
    policy = PWMPolicy(6, 2, actor_units=(8,), critic_units=(8,), num_critics=2, init_logstd=0.0, min_logstd=-5.0)
    assert {k for k in policy.state_dict() if k.startswith("ret_rms")} == {"ret_rms.mean", "ret_rms.var", "ret_rms.count"}


def test_parameter_groups_partition():
    model = _model()
    groups = model.parameter_groups()
    assert set(groups) == {"encoder", "model", "heads"}
    encoder, dynamics, heads = ({id(p) for p in groups[k]} for k in ("encoder", "model", "heads"))
    assert not encoder
    assert dynamics == {id(p) for p in model.dynamics_mlp.parameters()}
    assert heads == {
        id(p) for m in (model.reward_mlp, model.observation_mlp, model.termination_mlp) for p in m.parameters()
    }
    assert not dynamics & heads
    assert dynamics | heads == {id(p) for p in model.parameters()}


def test_update_normalizers_increments_counts():
    model = _model()
    obs = {"state": torch.randn(_B, _S), "state_ctx": torch.randn(_B, _C), "state_goal": torch.randn(_B, _G)}
    next_obs = {**obs, "state": obs["state"] + 1.0}
    before = model.in_norm.count.clone()
    model.update_normalizers(obs, torch.randn(_B, _A), next_obs)
    assert model.in_norm.count == before + _B
    assert model.goal_norm.count > 1e-4 + 1
    torch.testing.assert_close(model.out_norm.mean, torch.ones(_S), atol=1e-3, rtol=0)
