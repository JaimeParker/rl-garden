"""``RSSM`` NCRL options: ``heads``,
``reward_detached``, ``reward_valid`` filtering, ``kl_free_mode`` (with
unimix), reward-head size and two-hot space, straight-through estimator, and
that the defaults reproduce the pre-option behaviour."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.encoders.dreamer_conv import DreamerConvEncoder
from rl_garden.observations import ObservationSchema
from rl_garden.policies.action_mask import action_mask, masked_bounded_normal
from rl_garden.world_models.batch import SequenceBatch
from rl_garden.world_models.rssm import (
    RSSM,
    RSSMSize,
    _categorical_kl,
    kl_loss_mean_free,
)

_SIZE = RSSMSize(deter=32, hidden=16, discrete=4, units=16, cnn_depth=4)
T, B, A = 5, 3, 3


def _obs_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "state": spaces.Box(-1, 1, (6,), np.float32),
            "rgb_cam": spaces.Box(0, 255, (16, 16, 3), np.uint8),
        }
    )


def _rssm(seed: int = 0, **kwargs) -> RSSM:
    torch.manual_seed(seed)
    obs_space = _obs_space()
    schema = ObservationSchema.from_space(obs_space)
    encoder = DreamerConvEncoder(obs_space, schema, depth=4, units=16, state_layers=1)
    return RSSM(encoder, schema, action_dim=A, size=_SIZE, stoch=4, decoder_layers=1, **kwargs)


def _batch(**overrides) -> SequenceBatch:
    g = torch.Generator().manual_seed(1)
    fields = {
        "obs": {
            "state": torch.randn(T + 1, B, 6, generator=g),
            "rgb_cam": torch.randint(0, 256, (T + 1, B, 16, 16, 3), generator=g, dtype=torch.uint8),
        },
        "action": torch.randn(T, B, A, generator=g),
        "reward": torch.randn(T, B, generator=g),
        "is_first": torch.zeros(T, B, dtype=torch.bool),
        "terminated": torch.zeros(T, B, dtype=torch.bool),
        "carry": {"deter": torch.randn(B, 32, generator=g), "stoch": torch.rand(B, 16, generator=g)},
    }
    fields.update(overrides)
    return SequenceBatch(**fields)


def _losses(rssm: RSSM, batch: SequenceBatch, seed: int = 2):
    torch.manual_seed(seed)
    return rssm.model_loss(batch)


# Captured on 6017 (torch 2.6.0, CPU fp32) from the unmodified RSSM with this
# file's exact _rssm(seed=0)/_batch()/model_loss seed-2 setup, before the
# heads/reward_detached/kl_free_mode options existed.
_PRE_OPTION_LOSSES = {
    "dyn": 1.0201935768127441,
    "rep": 1.0201935768127441,
    "rgb_cam": 72.0066146850586,
    "state": 5.018107891082764,
    "rew": 5.541263103485107,
    "con": 0.7609280347824097,
}


def test_defaults_reproduce_pre_option_losses():
    losses, _ = _losses(_rssm(), _batch())
    assert set(losses) == set(_PRE_OPTION_LOSSES)
    for key, value in _PRE_OPTION_LOSSES.items():
        assert float(losses[key]) == pytest.approx(value, rel=1e-6), key


def test_explicit_default_options_are_bit_identical():
    default = _rssm()
    explicit = _rssm(heads=("reward", "continue"), reward_detached=False, kl_free_mode="per_element")
    for (name_a, a), (name_b, b) in zip(default.state_dict().items(), explicit.state_dict().items()):
        assert name_a == name_b and torch.equal(a, b)
    losses_a, _ = _losses(default, _batch())
    losses_b, _ = _losses(explicit, _batch())
    assert {k: float(v) for k, v in losses_a.items()} == {k: float(v) for k, v in losses_b.items()}


def test_no_heads_drops_losses_and_parameters():
    rssm = _rssm(heads=())
    assert rssm.reward_head is None and rssm.cont_head is None
    names = [name for name, _ in rssm.named_parameters()]
    assert not any(name.startswith(("reward_head.", "cont_head.")) for name in names)
    grouped = sum(p.numel() for group in rssm.parameter_groups().values() for p in group)
    assert grouped == sum(p.numel() for p in rssm.parameters())
    losses, _ = _losses(rssm, _batch(terminated=None))
    assert set(losses) == {"dyn", "rep", "rgb_cam", "state"}
    state = rssm.initial_state(B, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="reward head"):
        rssm.reward(state, None)
    with pytest.raises(RuntimeError, match="reward head"):
        rssm.reward_value(state)
    assert rssm.continue_(state) is None


def test_reward_only_head():
    losses, _ = _losses(_rssm(heads=("reward",)), _batch(terminated=None))
    assert "rew" in losses and "con" not in losses


def test_unknown_head_rejected():
    with pytest.raises(ValueError, match="Unknown RSSM heads"):
        _rssm(heads=("reward", "value"))


def _encoder_grad_norm(rssm: RSSM) -> float:
    return sum(float(p.grad.abs().sum()) for p in rssm.encoder.parameters() if p.grad is not None)


@pytest.mark.parametrize("detached", [False, True])
def test_reward_detached_blocks_representation_gradient(detached):
    rssm = _rssm(heads=("reward",), reward_detached=detached)
    with torch.no_grad():  # the zero-initialised output layer would hide the path
        rssm.reward_head.last.weight.normal_(std=0.5)
    losses, _ = _losses(rssm, _batch(terminated=None))
    losses["rew"].backward()
    head_grad = sum(float(p.grad.abs().sum()) for p in rssm.reward_head.parameters())
    assert head_grad > 0
    if detached:
        assert _encoder_grad_norm(rssm) == 0.0
        assert all(p.grad is None for p in rssm._deter_net.parameters())
    else:
        assert _encoder_grad_norm(rssm) > 0


def test_reward_valid_filters_columns():
    rssm = _rssm()
    valid = torch.tensor([True, False, True])
    batch = _batch(reward_valid=valid)
    masked, _ = _losses(rssm, batch)
    corrupted_reward = batch.reward.clone()
    corrupted_reward[:, 1] = 1e3
    masked_corrupted, _ = _losses(rssm, _batch(reward=corrupted_reward, reward_valid=valid))
    unmasked, _ = _losses(rssm, _batch())
    assert float(masked["rew"]) == float(masked_corrupted["rew"])
    assert float(masked["rew"]) != float(unmasked["rew"])


def test_kl_loss_mean_free_matches_hand_computation():
    g = torch.Generator().manual_seed(3)
    post = torch.randn(T, B, 4, 5, generator=g)
    prior = torch.randn(T, B, 4, 5, generator=g)
    kl = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=post), torch.distributions.Categorical(logits=prior)
    ).sum(-1)
    for free in (0.0, float(kl.mean()) + 1.0):
        dyn, rep = kl_loss_mean_free(post, prior, free)
        expected = max(float(kl.mean()), free)
        assert float(dyn) == pytest.approx(expected, rel=1e-5)
        assert float(rep) == pytest.approx(expected, rel=1e-5)
    # Differs from per-element clipping when only some elements are below free.
    free = float(kl.median())
    per_element = torch.clamp(_categorical_kl(post, prior).sum(-1), min=free).mean()
    assert float(kl_loss_mean_free(post, prior, free)[0]) != pytest.approx(float(per_element))


def test_kl_free_mode_mean_in_model_loss():
    high = 1e3
    losses, _ = _losses(_rssm(kl_free=high, kl_free_mode="mean"), _batch())
    assert float(losses["dyn"]) == high and float(losses["rep"]) == high
    # The mean path mixes unimix into the KL and per_element does not, so the
    # two agree at kl_free=0 only without unimix.
    zero_mean, _ = _losses(_rssm(kl_free=0.0, kl_free_mode="mean", unimix=0.0), _batch())
    zero_elem, _ = _losses(_rssm(kl_free=0.0, unimix=0.0), _batch())
    torch.testing.assert_close(zero_mean["dyn"], zero_elem["dyn"])
    with pytest.raises(ValueError, match="kl_free_mode"):
        _rssm(kl_free_mode="sum")


def test_padded_action_dims_stay_zero_in_dynamics():
    rssm = _rssm(heads=())
    state = rssm.initial_state(B, torch.device("cpu"))
    padded = torch.zeros(B, A)
    padded[:, :2] = torch.randn(B, 2)
    # The RSSM sees only zero padding: a padded action with garbage in the
    # last dim would change deter, so callers must keep it at zero.
    garbage = padded.clone()
    garbage[:, 2] = 5.0
    assert not torch.equal(rssm.step(state, padded)["deter"], rssm.step(state, garbage)["deter"])
    mask = action_mask(2, A)
    dist = masked_bounded_normal(torch.tanh(torch.randn(B, A)), torch.full((B, A), 0.5), mask)
    for sample in (dist.rsample(), dist.sample(), dist.mean, dist.mode):
        assert torch.all(sample[:, 2] == 0)
    reference = torch.distributions.Normal(dist._normal.mean[:, :2], dist._normal.stddev[:, :2])
    torch.testing.assert_close(dist.entropy(), reference.entropy().sum(-1))
    value = torch.randn(B, A)
    torch.testing.assert_close(dist.log_prob(value), reference.log_prob(value[:, :2]).sum(-1))


def _mixed_kl(post: torch.Tensor, prior: torch.Tensor, unimix: float) -> torch.Tensor:
    """Hand-computed batch-mean KL with ``unimix`` mixed into both sides."""
    def mix(logits):
        probs = torch.softmax(logits, -1)
        return probs * (1.0 - unimix) + unimix / probs.shape[-1]

    p, q = mix(post), mix(prior)
    return (p * (p.log() - q.log())).sum(-1).sum(-1).mean()


def test_kl_loss_mean_free_unimix():
    g = torch.Generator().manual_seed(4)
    post = torch.randn(T, B, 4, 5, generator=g) * 3.0
    prior = torch.randn(T, B, 4, 5, generator=g) * 3.0
    for dyn, rep in (kl_loss_mean_free(post, prior, 0.0), kl_loss_mean_free(post, prior, 0.0, unimix=0.0)):
        torch.testing.assert_close(dyn, _mixed_kl(post, prior, 0.0))
        torch.testing.assert_close(rep, _mixed_kl(post, prior, 0.0))
    dyn, rep = kl_loss_mean_free(post, prior, 0.0, unimix=0.01)
    torch.testing.assert_close(dyn, _mixed_kl(post, prior, 0.01))
    torch.testing.assert_close(rep, _mixed_kl(post, prior, 0.01))
    assert float(dyn) < float(_mixed_kl(post, prior, 0.0))


def test_mean_mode_model_loss_applies_unimix():
    mixed, _ = _losses(_rssm(kl_free=0.0, kl_free_mode="mean"), _batch())
    plain, _ = _losses(_rssm(kl_free=0.0, kl_free_mode="mean", unimix=0.0), _batch())
    assert torch.isfinite(mixed["dyn"]) and float(mixed["dyn"]) != float(plain["dyn"])


def test_reward_head_layers_units_and_symlog_space():
    rssm = _rssm(reward_layers=2, reward_units=24, reward_twohot_space="symlog")
    linears = [m for m in rssm.reward_head.mlp.layers if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(rssm.latent_dim, 24), (24, 24)]
    assert rssm.reward_head.twohot_space == "symlog"
    default = _rssm()
    assert len([m for m in default.reward_head.mlp.layers if isinstance(m, torch.nn.Linear)]) == 1
    assert default.reward_head.mlp.out_dim == _SIZE.units

    state = rssm.initial_state(B, torch.device("cpu"))
    state["stoch"] = torch.nn.functional.one_hot(torch.randint(0, 4, (B, 4)), 4).float()
    with torch.no_grad():
        rssm.reward_head.last.weight.normal_()
        value = rssm.reward_value(state)
        symlog_value = rssm.reward_value(state, symlog_space=True)
        torch.testing.assert_close(value, rssm.reward_head(rssm.features(state)).mode)
    torch.testing.assert_close(symlog_value, torch.sign(value) * torch.log1p(value.abs()), rtol=1e-4, atol=1e-5)


def test_straight_through_option_reaches_observe_and_step():
    rssm = _rssm(straight_through="probs")
    state = rssm.initial_state(B, torch.device("cpu"))
    action = torch.randn(B, A)
    stepped = rssm.step(state, action)
    stoch = stepped["stoch"].detach()
    torch.testing.assert_close(stoch, stoch.round())  # one-hot up to float rounding of probs - probs
    assert stepped["stoch"].requires_grad
    with pytest.raises(ValueError, match="straight_through"):
        _rssm(straight_through="soft")
