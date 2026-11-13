"""``NominalErrorStateSpaceWorldModel``: velocity-level predictions scored in
the codec's loss blocks with fixed nominal errors, on a point-mass codec."""
from __future__ import annotations

import math

import pytest
import torch
from _supermimic_fakes import PointMassTrackingEnv, PointMassVelocityCodec

from rl_garden.algorithms.supermimic import SuperMimic
from rl_garden.world_models import (
    NominalErrorStateSpaceWorldModel,
    ResidualStateCodec,
    SequenceBatch,
    StateSpaceWorldModel,
)

_S, _A, _C, _K, _B = 4, 2, 1, 3, 5
_NOMINAL = {"pos": 0.01, "vel": 1.0}
_CHANNELS = {"pos": slice(0, 2), "vel": slice(2, 4)}

NUM_ENVS = 4
SMALL = {
    "wm_hidden_dims": (32, 32),
    "head_hidden_dims": (16,),
    "goal_encoder_dims": (32,),
    "decoder_dims": (32,),
    "critic_dims": (16,),
    "horizon": 4,
    "wm_window": 2,
    "wm_batches": 1,
    "wm_batch_size": 8,
    "imagination_batch": 8,
    "warmup_updates": 1,
    "learning_starts": 4 * NUM_ENVS,
    "replay_ratio": None,
    "sim_steps_per_iteration": 2,
    "updates_per_iteration": 1,
    "buffer_size": 1024,
    "buffer_device": "cpu",
    "device": "cpu",
    "std_log": False,
    "seed": 0,
}


def _model(**overrides) -> NominalErrorStateSpaceWorldModel:
    kwargs = {
        "state_dim": _S,
        "action_dim": _A,
        "obs_dim": _S,
        "ctx_dim": _C,
        "codec": PointMassVelocityCodec(),
        "hidden_dims": (16, 16),
        "head_hidden_dims": (8,),
    }
    kwargs.update(overrides)
    return NominalErrorStateSpaceWorldModel(**kwargs)


def _randomize(model: StateSpaceWorldModel) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.3)
        model.in_norm.mean.normal_()
        model.in_norm.var.uniform_(0.5, 2.0)


def _window() -> SequenceBatch:
    return SequenceBatch(
        obs={"state": torch.randn(_K + 1, _B, _S), "state_ctx": torch.randn(_K + 1, _B, _C)},
        action=torch.randn(_K, _B, _A),
        reward=torch.randn(_K, _B),
        terminated=(torch.rand(_K, _B) > 0.5).float(),
    )


def _start_state(batch: SequenceBatch, s0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    return {
        "s": batch.obs["state"][0] if s0 is None else s0,
        "ctx": batch.obs["state_ctx"][0],
        "goal": torch.zeros(_B, 0),
        "t": torch.zeros(_B, dtype=torch.long),
    }


def _hand_rollout(model: StateSpaceWorldModel, batch: SequenceBatch) -> torch.Tensor:
    state = _start_state(batch)
    predicted = []
    for j in range(_K):
        state = model.step(state, batch.action[j])
        predicted.append(state["s"])
    return torch.stack(predicted)  # (k, B, S): predictions for steps 1..k


def test_state_loss_matches_closed_form():
    torch.manual_seed(0)
    model = _model()
    _randomize(model)
    batch = _window()
    losses, posterior = model.model_loss(batch)

    s_hat, s_real = _hand_rollout(model, batch), batch.obs["state"][1:]
    total_width = sum(sl.stop - sl.start for sl in _CHANNELS.values())
    expected = expected_last = 0.0
    for name, sl in _CHANNELS.items():
        squared = (s_hat[..., sl] - s_real[..., sl]).square()
        weight, width = 1.0 / _NOMINAL[name] ** 2, sl.stop - sl.start
        expected = expected + squared.mean() * weight * width / total_width
        expected_last = expected_last + squared[-1].mean(-1) * weight * width / total_width
        torch.testing.assert_close(losses[f"state_block/{name}"], squared.mean() * weight, rtol=1e-4, atol=1e-6)
        assert not losses[f"state_block/{name}"].requires_grad
    torch.testing.assert_close(losses["state"], expected, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(losses["state_err_last"], expected_last.mean(), rtol=1e-4, atol=1e-6)
    assert losses["observation"] == 0  # identity observation head
    torch.testing.assert_close(losses["total"], losses["state"] + losses["reward"] + losses["termination"])
    assert torch.equal(posterior["s"], batch.obs["state"][0])


def test_channels_outside_the_blocks_do_not_leak():
    torch.manual_seed(1)
    model = _model(codec=PointMassVelocityCodec(blocks=("pos",)))
    _randomize(model)
    batch = _window()
    base, _ = model.model_loss(batch)
    assert set(base) >= {"state_block/pos"} and "state_block/vel" not in base

    batch.obs["state"][1:, :, 2:] = 100.0 * torch.randn(_K, _B, 2)
    moved_velocity, _ = model.model_loss(batch)
    assert torch.equal(moved_velocity["state"], base["state"])
    assert torch.equal(moved_velocity["state_err_last"], base["state_err_last"])

    batch.obs["state"][1:, :, :2] += 0.1
    moved_position, _ = model.model_loss(batch)
    assert not torch.equal(moved_position["state"], base["state"])


def test_last_step_error_backprops_through_the_rollout():
    """Scoring only the last step must reach the first prediction, the start
    state and the dynamics parameters (full BPTT through the integrator)."""
    torch.manual_seed(2)
    model = _model()
    _randomize(model)
    batch = _window()
    s0 = batch.obs["state"][0].clone().requires_grad_(True)
    state = _start_state(batch, s0)
    predicted = [s0]
    for j in range(_K):
        state = model.step(state, batch.action[j])
        predicted.append(state["s"])
    first = predicted[1]
    s_hat = torch.stack(predicted)
    target = s_hat.detach().clone()
    target[-1] += 0.1  # error on the last step only
    loss, per_step_error, _ = model._state_loss(target, s_hat)
    assert torch.equal(per_step_error[:-1], torch.zeros(_K - 1, _B))
    params = list(model.dynamics_mlp.parameters())
    grad_first, grad_start, *grad_params = torch.autograd.grad(loss, [first, s0, *params])
    assert float(grad_first.abs().sum()) > 0
    assert float(grad_start.abs().sum()) > 0
    assert sum(float(g.abs().sum()) for g in grad_params) > 0


def test_output_is_not_normalized():
    torch.manual_seed(3)
    model = _model()
    _randomize(model)
    assert model.normalize_output is False and StateSpaceWorldModel.normalize_output is True
    batch = _window()
    state, action = _start_state(batch), batch.action[0]
    before = model.step(state, action)["s"]
    raw = model.dynamics_mlp(model.in_norm(torch.cat([state["s"], action, state["ctx"]], -1)))
    torch.testing.assert_close(before, model.codec.integrate(state["s"], raw))
    with torch.no_grad():
        model.out_norm.mean.normal_()
        model.out_norm.var.uniform_(2.0, 3.0)
    assert torch.equal(model.step(state, action)["s"], before)

    out_stats = {name: buf.clone() for name, buf in model.out_norm.named_buffers()}
    in_count = model.in_norm.count.clone()
    obs = {"state": torch.randn(_B, _S), "state_ctx": torch.randn(_B, _C)}
    model.update_normalizers(obs, torch.randn(_B, _A), {**obs, "state": obs["state"] + 1.0})
    for name, buf in model.out_norm.named_buffers():
        assert torch.equal(buf, out_stats[name]), name
    assert model.in_norm.count == in_count + _B


def test_construction_validation():
    with pytest.raises(ValueError, match="LossBlockStateCodec"):
        _model(codec=ResidualStateCodec(_S))
    with pytest.raises(ValueError, match="LossBlockStateCodec"):
        _model(codec=None)
    mismatched = PointMassVelocityCodec()
    mismatched.loss_blocks = lambda s: {"pos": s[..., :2]}
    with pytest.raises(ValueError, match="must equal"):
        _model(codec=mismatched)
    for bad in (0.0, -1.0, float("inf")):
        codec = PointMassVelocityCodec()
        codec.nominal_errors = {"pos": bad, "vel": 1.0}
        with pytest.raises(ValueError, match="finite and > 0"):
            _model(codec=codec)


def test_prediction_is_velocity_level_but_loss_covers_all_channels():
    torch.manual_seed(4)
    model = _model()
    _randomize(model)
    assert model.dynamics_mlp[-1].out_features == 2
    assert model.out_norm.mean.shape == (2,)
    assert model.block_widths == {"pos": 2, "vel": 2}
    batch = _window()
    base, _ = model.model_loss(batch)
    for channel in range(_S):
        moved = SequenceBatch(
            obs={**batch.obs, "state": batch.obs["state"].clone()},
            action=batch.action,
            reward=batch.reward,
            terminated=batch.terminated,
        )
        moved.obs["state"][1:, :, channel] += 0.5
        losses, _ = model.model_loss(moved)
        assert not torch.equal(losses["state"], base["state"]), channel


def _make_agent(heads: str, **overrides) -> SuperMimic:
    env = PointMassTrackingEnv(NUM_ENVS)
    kwargs = {"codec": PointMassVelocityCodec(), "world_model_cls": NominalErrorStateSpaceWorldModel}
    return SuperMimic(env, heads=heads, **{**SMALL, **kwargs, **overrides})


def test_supermimic_builds_the_class_attribute_by_default():
    assert SuperMimic.world_model_cls is StateSpaceWorldModel
    agent = SuperMimic(PointMassTrackingEnv(NUM_ENVS), heads="learned", **SMALL)
    assert type(agent.world_model) is StateSpaceWorldModel
    assert agent._checkpoint_metadata()["world_model_cls"] == (
        "rl_garden.world_models.state_space.StateSpaceWorldModel"
    )


@pytest.mark.parametrize("heads", ["learned", "analytic"])
def test_supermimic_learns_and_round_trips_with_nominal_error_world_model(monkeypatch, tmp_path, heads):
    agent = _make_agent(heads)
    assert type(agent.world_model) is NominalErrorStateSpaceWorldModel
    recorded = []
    gradient_step = agent._gradient_step

    def recording_gradient_step():
        metrics = gradient_step()
        recorded.append(metrics)
        return metrics

    monkeypatch.setattr(agent, "_gradient_step", recording_gradient_step)
    agent.learn(total_timesteps=200)
    assert recorded and any("actor/loss" in metrics for metrics in recorded)
    for metrics in recorded:
        assert {"wm/loss", "wm/state_block/pos", "wm/state_block/vel"} <= set(metrics)
        for key, value in metrics.items():
            assert math.isfinite(value), (key, value)

    path = agent.save(tmp_path / "supermimic.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["hyperparameters"]["world_model_cls"] == (
        "rl_garden.world_models.nominal_error_state_space.NominalErrorStateSpaceWorldModel"
    )
    restored = _make_agent(heads, seed=1)
    restored.load(path, load_replay_buffer=False)
    restored_policy = restored.policy.state_dict()
    for key, value in agent.policy.state_dict().items():
        torch.testing.assert_close(restored_policy[key], value)
