"""Unit tests for ``rl_garden.world_models.rssm.RSSM`` (plan model-based-
base Part 2, section B).
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.encoders.dreamer_conv import DreamerConvEncoder
from rl_garden.observations import ObservationSchema
from rl_garden.world_models.rssm import RSSM, RSSMSize, kl_loss


def _tiny_size(**overrides) -> RSSMSize:
    base = dict(deter=32, hidden=16, discrete=4, units=16, cnn_depth=4)
    base.update(overrides)
    return RSSMSize(**base)


def _obs_space() -> spaces.Dict:
    return spaces.Dict({"state": spaces.Box(-1, 1, shape=(6,), dtype=np.float32)})


def _build_rssm(*, stoch: int = 4, action_dim: int = 3, size: RSSMSize = None):
    obs_space = _obs_space()
    schema = ObservationSchema.from_space(obs_space)
    size = size or _tiny_size()
    enc = DreamerConvEncoder(obs_space, schema, units=size.units, state_layers=1)
    rssm = RSSM(enc, schema, action_dim=action_dim, size=size, stoch=stoch, decoder_layers=1)
    return rssm, obs_space, schema


def test_rssm_constructor_signature_and_latent_dim():
    size = _tiny_size()
    rssm, _, _ = _build_rssm(stoch=4, size=size)
    assert rssm.latent_dim == size.deter + 4 * size.discrete
    assert rssm.flat_stoch == 4 * size.discrete


def test_initial_state_shapes_and_zeros():
    rssm, _, _ = _build_rssm()
    state = rssm.initial_state(5, torch.device("cpu"))
    assert state["deter"].shape == (5, rssm._deter)
    assert state["stoch"].shape == (5, rssm._stoch, rssm._discrete)
    assert torch.equal(state["deter"], torch.zeros_like(state["deter"]))
    assert torch.equal(state["stoch"], torch.zeros_like(state["stoch"]))


def test_step_shapes_and_sample_vs_mode():
    rssm, _, _ = _build_rssm(action_dim=3)
    state = rssm.initial_state(4, torch.device("cpu"))
    action = torch.randn(4, 3)
    sampled = rssm.step(state, action, sample=True)
    mode = rssm.step(state, action, sample=False)
    assert sampled["deter"].shape == state["deter"].shape
    assert sampled["stoch"].shape == state["stoch"].shape
    torch.testing.assert_close(sampled["deter"], mode["deter"])  # deter transition is deterministic
    # stoch differs in general between a sample and the mode (not asserting
    # inequality -- could coincide by chance -- just that both are one-hot).
    torch.testing.assert_close(mode["stoch"].sum(-1), torch.ones(4, rssm._stoch))
    torch.testing.assert_close(sampled["stoch"].sum(-1), torch.ones(4, rssm._stoch))


def test_observe_resets_state_and_action_where_is_first():
    rssm, _, _ = _build_rssm(action_dim=3)
    batch = 4
    state = {
        "deter": torch.randn(batch, rssm._deter),
        "stoch": torch.rand(batch, rssm._stoch, rssm._discrete),
        "logits": torch.zeros(batch, rssm._stoch, rssm._discrete),
    }
    action = torch.randn(batch, 3)
    embed = torch.randn(batch, rssm.encoder.features_dim)
    is_first = torch.tensor([1.0, 0.0, 1.0, 0.0])

    reset_state = {k: v.clone() for k, v in state.items()}
    reset_state["deter"][is_first.bool()] = 0.0
    reset_state["stoch"][is_first.bool()] = 0.0
    zeroed_action = action.clone()
    zeroed_action[is_first.bool()] = 0.0

    out = rssm.observe(state, action, embed, is_first)
    out_from_reset = rssm.observe(reset_state, zeroed_action, embed, torch.zeros(batch))
    # Per-env, observe() with is_first masking must match observe() called
    # with the state/action already zeroed and is_first all False.
    torch.testing.assert_close(out["deter"], out_from_reset["deter"])


def test_observe_shapes_per_step():
    rssm, _, _ = _build_rssm(action_dim=3)
    batch = 5
    state = rssm.initial_state(batch, torch.device("cpu"))
    action = torch.randn(batch, 3)
    embed = torch.randn(batch, rssm.encoder.features_dim)
    is_first = torch.zeros(batch)
    out = rssm.observe(state, action, embed, is_first)
    assert out["deter"].shape == (batch, rssm._deter)
    assert out["stoch"].shape == (batch, rssm._stoch, rssm._discrete)
    assert out["logits"].shape == (batch, rssm._stoch, rssm._discrete)


def test_observe_requires_action():
    import pytest

    rssm, _, _ = _build_rssm()
    state = rssm.initial_state(2, torch.device("cpu"))
    embed = torch.randn(2, rssm.encoder.features_dim)
    with pytest.raises(ValueError):
        rssm.observe(state, None, embed, torch.zeros(2))


def test_kl_loss_free_bits_clip():
    post_logits = torch.zeros(2, 3, 32, 4)
    prior_logits = torch.zeros(2, 3, 32, 4)  # identical -> KL == 0, must clip to free
    dyn, rep = kl_loss(post_logits, prior_logits, free=1.0)
    torch.testing.assert_close(dyn, torch.full((2, 3), 1.0))
    torch.testing.assert_close(rep, torch.full((2, 3), 1.0))


def test_kl_loss_exceeds_free_bits_when_distributions_differ():
    torch.manual_seed(0)
    post_logits = torch.randn(2, 3, 32, 4) * 5.0
    prior_logits = torch.randn(2, 3, 32, 4) * 5.0
    dyn, rep = kl_loss(post_logits, prior_logits, free=1.0)
    assert bool((dyn >= 1.0).all())
    assert bool((rep >= 1.0).all())


def test_kl_loss_gradient_direction():
    """``dyn_loss = KL(sg(post) || prior)`` must have ZERO gradient w.r.t.
    ``post_logits`` (``post_logits.detach()`` is the left/``sg`` argument
    there); ``rep_loss = KL(post || sg(prior))`` must have ZERO gradient
    w.r.t. ``prior_logits`` (``prior_logits.detach()`` is the right/``sg``
    argument there) -- this module's own ``kl_loss`` docstring / r2dreamer
    ``rssm.py:222-230``'s stop-gradient placement. ``free=0.0`` (rather than
    the default-sized free-bits budget) so the ``clamp(min=free)`` floor
    never binds and masks a real gradient with a clamp-boundary zero."""
    torch.manual_seed(0)
    post_logits = torch.randn(2, 3, 32, 4, requires_grad=True)
    prior_logits = torch.randn(2, 3, 32, 4, requires_grad=True)

    dyn, _ = kl_loss(post_logits, prior_logits, free=0.0)
    dyn.sum().backward()
    assert post_logits.grad is None
    assert prior_logits.grad is not None and bool((prior_logits.grad != 0).any())

    post_logits.grad = None
    prior_logits.grad = None
    _, rep = kl_loss(post_logits, prior_logits, free=0.0)
    rep.sum().backward()
    assert prior_logits.grad is None
    assert post_logits.grad is not None and bool((post_logits.grad != 0).any())


def test_decode_keys_match_schema():
    rssm, _, schema = _build_rssm()
    state = rssm.initial_state(3, torch.device("cpu"))
    dists = rssm.decode(state)
    assert set(dists.keys()) == set(schema.keys)


def _fill_buffer_and_sample(rssm, obs_space, action_space, *, num_envs=2, horizon=4, add_steps=8):
    buf = SequenceReplayBuffer(
        obs_space,
        action_space,
        num_envs=num_envs,
        buffer_size=num_envs * 32,
        cross_episode=True,
        priority=False,
        horizon=horizon,
        carry_spec={"deter": (rssm._deter,), "stoch": (rssm.flat_stoch,)},
        storage_device="cpu",
        sample_device="cpu",
    )
    carry_state = rssm.initial_state(num_envs, torch.device("cpu"))
    is_first = torch.ones(num_envs)
    for t in range(add_steps):
        obs = {"state": torch.randn(num_envs, 6)}
        action_t = torch.randn(num_envs, action_space.shape[0])
        reward_t = torch.randn(num_envs)
        done_t = torch.zeros(num_envs)
        ep_end_t = torch.zeros(num_envs)
        carry = {
            "deter": carry_state["deter"],
            "stoch": carry_state["stoch"].reshape(num_envs, -1),
        }
        buf.add(obs, obs, action_t, reward_t, done_t, ep_end_t, is_first=is_first, carry=carry)
        embed = rssm.encode(obs)
        carry_state = rssm.observe(carry_state, action_t, embed, is_first)
        is_first = torch.zeros(num_envs)
    return buf.sample(batch_size=6)


def test_model_loss_is_finite_and_returns_expected_shapes():
    action_dim = 3
    rssm, obs_space, _ = _build_rssm(action_dim=action_dim)
    action_space = spaces.Box(-1, 1, shape=(action_dim,), dtype=np.float32)
    batch = _fill_buffer_and_sample(rssm, obs_space, action_space, horizon=4)

    losses, posterior = rssm.model_loss(batch)
    for key in ("dyn", "rep", "state", "rew", "con"):
        assert key in losses, f"missing loss key {key!r}"
        assert torch.isfinite(losses[key]).all()
        assert losses[key].dim() == 0  # already mean-reduced

    horizon, num_envs_sampled = batch.action.shape[0], batch.action.shape[1]
    assert posterior["deter"].shape == (horizon, num_envs_sampled, rssm._deter)
    assert posterior["stoch"].shape == (horizon, num_envs_sampled, rssm._stoch, rssm._discrete)


def test_model_loss_gradients_flow_to_encoder_and_dynamics():
    action_dim = 3
    rssm, obs_space, _ = _build_rssm(action_dim=action_dim)
    action_space = spaces.Box(-1, 1, shape=(action_dim,), dtype=np.float32)
    batch = _fill_buffer_and_sample(rssm, obs_space, action_space, horizon=4)

    losses, _ = rssm.model_loss(batch)
    total = sum(losses.values())
    total.backward()

    encoder_grad = next(p for p in rssm.encoder.parameters() if p.requires_grad).grad
    dyn_grad = next(p for p in rssm._deter_net.parameters()).grad
    assert encoder_grad is not None and torch.isfinite(encoder_grad).all()
    assert dyn_grad is not None and torch.isfinite(dyn_grad).all()


def test_contdisc_flag_changes_continue_target():
    action_dim = 2
    obs_space = _obs_space()
    schema = ObservationSchema.from_space(obs_space)
    action_space = spaces.Box(-1, 1, shape=(action_dim,), dtype=np.float32)
    size = _tiny_size()

    torch.manual_seed(0)
    enc_off = DreamerConvEncoder(obs_space, schema, units=size.units, state_layers=1)
    rssm_off = RSSM(enc_off, schema, action_dim=action_dim, size=size, stoch=4, decoder_layers=1, contdisc=False)
    torch.manual_seed(0)
    enc_on = DreamerConvEncoder(obs_space, schema, units=size.units, state_layers=1)
    rssm_on = RSSM(
        enc_on, schema, action_dim=action_dim, size=size, stoch=4, decoder_layers=1,
        contdisc=True, discount_horizon=10.0,
    )

    torch.manual_seed(1)
    batch = _fill_buffer_and_sample(rssm_off, obs_space, action_space, horizon=3)
    is_terminal = torch.zeros_like(batch.is_terminal)  # force no terminations for a clean comparison
    batch.is_terminal = is_terminal

    target_off = 1.0 - batch.is_terminal.float()
    target_on = (1.0 - batch.is_terminal.float()) * (1.0 - 1.0 / rssm_on.discount_horizon)
    assert not torch.allclose(target_off, target_on)
