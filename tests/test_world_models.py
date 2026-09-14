"""Tests for ``rl_garden.world_models``: the abstract ``WorldModel``/``State``
contract (``base.py``), the shared ``imagine()`` rollout primitive
(``imagine.py``), and TD-MPC2's ``LatentConsistencyModel``
(``latent_consistency.py``). No simulator/hardware -- fake CPU tensors only.
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.world_models.base import State, WorldModel
from rl_garden.world_models.imagine import imagine
from rl_garden.world_models.latent_consistency import LatentConsistencyModel


class _FakeWorldModel(WorldModel):
    """Minimal concrete ``WorldModel``: state = {"h": Tensor}, a linear
    scalar-sum dynamics, used only to exercise ``imagine()``/the base
    interface without any real network."""

    def __init__(self, dim: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.latent_dim = dim
        self.linear = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.linear.weight)
        self.encoder = FlattenExtractor(spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder.extract(obs)

    def initial_state(self, batch_size: int, device: torch.device) -> State:
        return {"h": torch.zeros(batch_size, self.dim, device=device)}

    def observe(self, state, action, embed, is_first) -> State:
        del state, action, is_first
        return {"h": embed}

    def step(self, state: State, action: torch.Tensor, *, sample: bool = True) -> State:
        del sample
        return {"h": self.linear(state["h"]) + action}

    def reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        return state["h"].sum(-1, keepdim=True) + action.sum(-1, keepdim=True)

    def continue_(self, state: State):
        return torch.full((state["h"].shape[0], 1), 0.9)

    def features(self, state: State) -> torch.Tensor:
        return state["h"]

    def parameter_groups(self):
        return {"encoder": self.encoder.parameters(), "model": self.linear.parameters()}

    def model_loss(self, batch):
        raise NotImplementedError("not exercised by these tests")


def _policy_fn(state: State) -> torch.Tensor:
    return torch.ones_like(state["h"])


# ---------------------------------------------------------------------------
# base.py: WorldModel / State contract
# ---------------------------------------------------------------------------


def test_continue_and_decode_default_to_none():
    model = _FakeWorldModel()
    # continue_ is overridden above (non-None); decode is not overridden
    # anywhere -- confirms the base class's stated default.
    assert model.decode({"h": torch.zeros(1, 3)}) is None


def test_world_model_duck_types_actor_extractor_surface():
    model = _FakeWorldModel(dim=4)
    assert model.features_dim == model.latent_dim == 4

    obs = torch.randn(2, 4)
    out = model.extract(obs)
    torch.testing.assert_close(out, model.encode(obs))
    assert model.extract(obs, stop_gradient=True).requires_grad is False

    # Delegates to self.encoder; a no-op default should not raise.
    model.update_normalizer(obs)


# ---------------------------------------------------------------------------
# imagine.py
# ---------------------------------------------------------------------------


def test_imagine_produces_expected_shapes_and_no_grad_by_default():
    model = _FakeWorldModel(dim=3)
    start = model.initial_state(batch_size=5, device=torch.device("cpu"))

    traj = imagine(model, _policy_fn, start, horizon=4, grad=False)

    assert traj.states["h"].shape == (5, 5, 3)  # horizon + 1, B, dim
    assert traj.actions.shape == (4, 5, 3)
    assert traj.rewards.shape == (4, 5, 1)
    assert traj.continues.shape == (4, 5, 1)
    assert traj.done_mask.shape == (4, 5, 1)
    assert not traj.rewards.requires_grad


def test_imagine_grad_true_keeps_autograd_enabled():
    model = _FakeWorldModel(dim=2)
    model.linear.weight.requires_grad_(True)
    start = model.initial_state(batch_size=2, device=torch.device("cpu"))

    traj = imagine(model, _policy_fn, start, horizon=3, grad=True)
    loss = traj.rewards.sum()
    loss.backward()

    assert model.linear.weight.grad is not None


def test_imagine_never_terminating_model_has_all_zero_done_mask():
    model = _FakeWorldModel(dim=2)
    start = model.initial_state(batch_size=2, device=torch.device("cpu"))
    # continue_ always returns 0.9 (>= 0.5) -> never terminates.
    traj = imagine(model, _policy_fn, start, horizon=3, grad=False)
    assert torch.all(traj.done_mask == 0.0)


def test_imagine_defaults_to_never_terminating_when_no_continue_head():
    class _NoContinueModel(_FakeWorldModel):
        def continue_(self, state):
            return None

    model = _NoContinueModel(dim=2)
    start = model.initial_state(batch_size=2, device=torch.device("cpu"))
    traj = imagine(model, _policy_fn, start, horizon=3, grad=False)
    assert torch.all(traj.continues == 1.0)
    assert torch.all(traj.done_mask == 0.0)


# ---------------------------------------------------------------------------
# latent_consistency.py
# ---------------------------------------------------------------------------


def _make_model(latent_dim=8, mlp_dim=8, num_bins=11, episodic=False) -> LatentConsistencyModel:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    encoder = FlattenExtractor(obs_space)
    model = LatentConsistencyModel(
        encoder=encoder, action_dim=2, latent_dim=latent_dim, mlp_dim=mlp_dim, num_bins=num_bins, episodic=episodic
    )
    model.apply_init()
    return model


def test_encode_observe_step_reward_shapes():
    model = _make_model()
    obs = torch.zeros(3, 4)
    embed = model.encode(obs)
    assert embed.shape == (3, model.latent_dim)

    state = model.observe(None, None, embed, torch.zeros(3, dtype=torch.bool))
    assert set(state.keys()) == {"z"}
    torch.testing.assert_close(state["z"], embed)

    action = torch.zeros(3, 2)
    next_state = model.step(state, action)
    assert next_state["z"].shape == (3, model.latent_dim)

    reward_logits = model.reward(state, action)
    assert reward_logits.shape == (3, model.num_bins)

    assert torch.equal(model.features(state), state["z"])


def test_continue_is_none_when_not_episodic_and_a_probability_when_episodic():
    model = _make_model(episodic=False)
    state = {"z": torch.zeros(2, model.latent_dim)}
    assert model.continue_(state) is None

    episodic_model = _make_model(episodic=True)
    state = {"z": torch.zeros(2, episodic_model.latent_dim)}
    continue_prob = episodic_model.continue_(state)
    assert continue_prob.shape == (2, 1)
    assert torch.all((continue_prob >= 0.0) & (continue_prob <= 1.0))


def test_model_loss_zs_is_deterministic_across_repeated_calls():
    """``model_loss()``'s internal dynamics rollout (formerly the standalone
    ``rollout()`` method, deleted per model-based-base plan item 0 -- the
    live posterior it returns now replaces that second call site) has no
    RNG/dropout of its own, so two calls on identical inputs must produce
    bit-identical ``zs``."""
    model = _make_model()
    horizon, batch = 3, 4
    obs = torch.randn(horizon + 1, batch, 4)
    action = torch.randn(horizon, batch, 2).clamp(-1, 1)
    reward = torch.randn(horizon, batch, 1)
    with torch.no_grad():
        next_z = model.encode(obs[1:].reshape(-1, 4)).reshape(horizon, batch, -1)
    batch_dict = {"obs": obs, "action": action, "reward": reward, "next_z": next_z}

    _, posterior1 = model.model_loss(batch_dict)
    _, posterior2 = model.model_loss(batch_dict)

    assert posterior1["z"].shape == (horizon + 1, batch, model.latent_dim)
    torch.testing.assert_close(posterior1["z"].detach(), posterior2["z"].detach())


def test_model_loss_returns_named_losses_and_live_posterior():
    model = _make_model()
    horizon, batch = 3, 4
    obs = torch.randn(horizon + 1, batch, 4)
    action = torch.randn(horizon, batch, 2).clamp(-1, 1)
    reward = torch.randn(horizon, batch, 1)
    with torch.no_grad():
        next_z = model.encode(obs[1:].reshape(-1, 4)).reshape(horizon, batch, -1)
    batch_dict = {"obs": obs, "action": action, "reward": reward, "next_z": next_z}

    losses, posterior = model.model_loss(batch_dict)

    assert set(losses.keys()) == {"consistency_loss", "reward_loss", "termination_loss"}
    for value in losses.values():
        assert torch.isfinite(value)
    assert posterior["z"].shape == (horizon + 1, batch, model.latent_dim)
    # Live (graph-attached), not detached -- plan item 0: the caller's
    # critic value loss must backprop into this model's own parameters.
    assert posterior["z"].requires_grad is True
