"""NCRL backbone (``rl_garden.networks.ncrl_nets``, ``RSSM(backbone="ncrl")``):
module shapes at 64x64, LayerNorm and default init, the block-GRU without
action normalization, ``model_loss`` gradients, ``RSSMPretrain`` and ``NCRL``
smoke runs with the backbone, checkpoint backbone checks, and the masked
TruncatedNormal actor."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from _ncrl_online_synthetic import (
    PixelVecEnv,
    make_pretrain_artifacts,
    tiny_ncrl_kwargs,
)
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "conversion"))

from _ncrl_synthetic import write_benchmark
from convert_ncrl_dataset import convert_dataset

from rl_garden.algorithms.ncrl import (
    NCRL,
    MaskedTruncatedNormalHead,
    load_rssm_pretrain_checkpoint,
    rssm_from_pretrain_checkpoint,
)
from rl_garden.algorithms.rssm_pretrain import RSSMPretrain, sequence_dataset_env_spec
from rl_garden.networks.dreamer_nets import RMSNorm
from rl_garden.networks.ncrl_nets import (
    NCRLBlockGRU,
    NCRLBlockLinear,
    NCRLConvDecoder,
    NCRLConvEncoder,
    NCRLHead,
)
from rl_garden.observations import ObservationSchema
from rl_garden.observations.schema import ObservationContractError
from rl_garden.policies.action_mask import MaskedTruncatedNormal, action_mask
from rl_garden.world_models.batch import SequenceBatch
from rl_garden.world_models.rssm import RSSM, RSSMSize, build_rssm_encoder

SIZE = RSSMSize(deter=32, hidden=16, discrete=4, units=16, cnn_depth=4)
STOCH, A, T, B = 4, 3, 4, 2
FLAT_STOCH = STOCH * SIZE.discrete


def _obs_space(size: int = 64) -> spaces.Dict:
    return spaces.Dict({"rgb_default": spaces.Box(0, 255, (size, size, 3), np.uint8)})


def _rssm(seed: int = 0, **kwargs) -> RSSM:
    torch.manual_seed(seed)
    obs_space = _obs_space()
    schema = ObservationSchema.from_space(obs_space)
    encoder = build_rssm_encoder("ncrl", obs_space, schema, SIZE)
    kwargs.setdefault("heads", ("reward",))
    return RSSM(encoder, schema, A, SIZE, stoch=STOCH, backbone="ncrl", **kwargs)


def _batch() -> SequenceBatch:
    g = torch.Generator().manual_seed(1)
    return SequenceBatch(
        obs={"rgb_default": torch.randint(0, 256, (T + 1, B, 64, 64, 3), generator=g, dtype=torch.uint8)},
        action=torch.rand(T, B, A, generator=g) * 2 - 1,
        reward=torch.randn(T, B, generator=g),
        is_first=torch.zeros(T, B, dtype=torch.bool),
        terminated=torch.zeros(T, B, dtype=torch.bool),
        carry={"deter": torch.zeros(B, SIZE.deter), "stoch": torch.zeros(B, FLAT_STOCH)},
    )


# ----------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------


def test_encoder_and_decoder_shapes_at_64x64():
    obs_space = _obs_space()
    schema = ObservationSchema.from_space(obs_space)
    encoder = NCRLConvEncoder(obs_space, schema, depth=4)
    assert encoder.features_dim == 32 * 4  # 64 -> 31 -> 14 -> 6 -> 2, 8 * depth channels
    obs = {"rgb_default": torch.randint(0, 256, (T, B, 64, 64, 3), dtype=torch.uint8)}
    assert encoder(obs).shape == (T, B, 32 * 4)

    decoder = NCRLConvDecoder(schema, SIZE.deter, FLAT_STOCH, depth=4, units=16)
    dists = decoder(torch.randn(T, B, SIZE.deter), torch.randn(T, B, FLAT_STOCH))
    assert set(dists) == {"rgb_default"}
    assert dists["rgb_default"].mode.shape == (T, B, 64, 64, 3)
    # MSEDist(output + 0.5) against obs / 255 equals the official MSE against obs / 255 - 0.5.
    target = torch.rand(T, B, 64, 64, 3)
    raw = dists["rgb_default"].mode - 0.5
    torch.testing.assert_close(
        dists["rgb_default"].log_prob(target), -((raw - (target - 0.5)) ** 2).sum((2, 3, 4))
    )


def test_image_size_and_state_keys_are_rejected():
    small = _obs_space(32)
    with pytest.raises(ObservationContractError, match="32"):
        NCRLConvDecoder(ObservationSchema.from_space(small), SIZE.deter, FLAT_STOCH, depth=4, units=16)
    with_state = spaces.Dict({**_obs_space().spaces, "state": spaces.Box(-1, 1, (3,), np.float32)})
    with pytest.raises(ObservationContractError, match="image-only"):
        NCRLConvEncoder(with_state, ObservationSchema.from_space(with_state), depth=4)


def test_block_linear_matches_the_official_layout_and_init():
    layer = NCRLBlockLinear(32, 48, blocks=8)
    assert layer.weight.shape == (8, 4, 6)
    bound = 1.0 / math.sqrt(4 * 6)  # official fan-in of the (blocks, in/G, out/G) weight
    assert layer.weight.abs().max() <= bound and layer.bias.abs().max() <= bound
    x = torch.randn(5, 32)
    expected = torch.cat([x[:, 4 * g : 4 * g + 4] @ layer.weight[g] for g in range(8)], dim=-1) + layer.bias
    torch.testing.assert_close(layer(x), expected)


def test_block_gru_does_not_normalize_actions():
    gru = NCRLBlockGRU(SIZE.deter, FLAT_STOCH, A, SIZE.hidden, blocks=8)
    deter, stoch = torch.randn(B, SIZE.deter), torch.randn(B, FLAT_STOCH)
    out = gru(stoch, deter, torch.ones(B, A))
    assert out.shape == (B, SIZE.deter)
    # The DreamerV3 core maps |a| > 1 to a / |a|; this one sees the raw action.
    assert not torch.allclose(out, gru(stoch, deter, torch.full((B, A), 3.0)))


def test_head_splits_deter_and_stoch_and_keeps_default_init():
    torch.manual_seed(0)
    head = NCRLHead(SIZE.deter, FLAT_STOCH, 255, output="symexp_twohot", layers=2, units=16, twohot_space="symlog")
    assert head.last.weight.abs().sum() > 0  # no zero outscale
    assert isinstance(head.mlp.fc_deter[0], NCRLBlockLinear)
    feat = torch.randn(T, B, SIZE.deter + FLAT_STOCH)
    value = head(feat).mode
    assert value.shape == (T, B, 1) and torch.isfinite(value).all()
    changed = feat.clone()
    changed[..., SIZE.deter :] += 1.0
    assert not torch.allclose(head(changed).mode, value)


# ----------------------------------------------------------------------
# RSSM with backbone="ncrl"
# ----------------------------------------------------------------------


def test_ncrl_rssm_uses_layernorm_only_and_trains():
    rssm = _rssm(reward_layers=2, reward_units=16, reward_twohot_space="symlog", straight_through="probs")
    assert rssm.backbone == "ncrl"
    assert not any(isinstance(m, (RMSNorm, torch.nn.RMSNorm)) for m in rssm.modules())
    assert any(isinstance(m, torch.nn.LayerNorm) for m in rssm.modules())
    assert isinstance(rssm.reward_head, NCRLHead) and isinstance(rssm.decoder, NCRLConvDecoder)
    losses, posterior = rssm.model_loss(_batch())
    assert set(losses) == {"dyn", "rep", "rgb_default", "rew"}
    assert all(torch.isfinite(v) for v in losses.values())
    assert posterior["deter"].shape == (T, B, SIZE.deter)
    sum(losses.values()).backward()
    for name, parameter in rssm.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    state = rssm.initial_state(B, torch.device("cpu"))
    step = rssm.step(state, torch.rand(B, A))
    assert rssm.reward_value(step).shape == (B, 1)


# ----------------------------------------------------------------------
# Algorithms
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("ncrl_nets")
    write_benchmark(root / "src", {"walker": (6, (14, 16, 12))}, image_size=64)
    convert_dataset(root / "src", root / "dst")
    return root / "dst"


def test_rssm_pretrain_with_ncrl_backbone_reduces_reconstruction(dataset_dir, tmp_path):
    torch.manual_seed(0)
    agent = RSSMPretrain(
        sequence_dataset_env_spec(dataset_dir),
        dataset_dir=dataset_dir,
        backbone="ncrl",
        rssm_size=SIZE,
        stoch=STOCH,
        batch_size=4,
        batch_length=6,
        lr=1e-3,
        compute_dtype="float32",
        device="cpu",
        buffer_device="cpu",
        log_freq=0,
        std_log=False,
    )
    assert isinstance(agent.world_model.encoder, NCRLConvEncoder)
    history = [agent.train(1)["recon_loss"] for _ in range(20)]
    assert all(math.isfinite(v) for v in history)
    assert np.mean(history[-3:]) < np.mean(history[:3])
    path = agent.save(tmp_path / "pretrain.pt")
    checkpoint = load_rssm_pretrain_checkpoint(path)
    assert checkpoint["metadata"]["hyperparameters"]["backbone"] == "ncrl"
    rebuilt = rssm_from_pretrain_checkpoint(checkpoint)
    assert rebuilt.backbone == "ncrl"
    with pytest.raises(ValueError, match="--backbone ncrl"):
        rssm_from_pretrain_checkpoint(checkpoint, backbone="dreamer_v3")


@pytest.fixture(scope="module")
def ncrl_artifacts(tmp_path_factory):
    return make_pretrain_artifacts(tmp_path_factory.mktemp("ncrl_backbone"), backbone="ncrl")


def test_ncrl_with_ncrl_backbone_smoke(ncrl_artifacts):
    torch.manual_seed(0)
    agent = NCRL(PixelVecEnv(), **tiny_ncrl_kwargs(ncrl_artifacts, backbone="ncrl"))
    assert agent.world_model.backbone == "ncrl"
    assert isinstance(agent.policy.actor, MaskedTruncatedNormalHead)
    assert isinstance(agent.policy.critic, NCRLHead) and agent.policy.critic.twohot_space == "symlog"
    assert isinstance(agent.policy.slow_critic, NCRLHead)
    assert not any(p.requires_grad for p in agent.policy.slow_critic.parameters())
    agent.learn(total_timesteps=80)
    assert agent._global_update > 0
    metrics = agent.train(2)
    for key, value in metrics.items():
        assert math.isfinite(value), f"{key}={value}"
    assert agent._checkpoint_metadata()["backbone"] == "ncrl"


def test_ncrl_backbone_must_match_the_checkpoint(ncrl_artifacts, tmp_path):
    with pytest.raises(ValueError, match="pass --backbone ncrl"):
        NCRL(PixelVecEnv(), **tiny_ncrl_kwargs(ncrl_artifacts, backbone="dreamer_v3"))
    # Checkpoints written before the option existed count as dreamer_v3.
    legacy = make_pretrain_artifacts(tmp_path / "legacy")
    checkpoint = torch.load(legacy["checkpoint"], map_location="cpu", weights_only=False)
    del checkpoint["metadata"]["hyperparameters"]["backbone"]
    torch.save(checkpoint, legacy["checkpoint"])
    with pytest.raises(ValueError, match="pass --backbone dreamer_v3"):
        NCRL(PixelVecEnv(), **tiny_ncrl_kwargs(legacy, backbone="ncrl"))
    agent = NCRL(PixelVecEnv(), **tiny_ncrl_kwargs(legacy, backbone="dreamer_v3"))
    assert agent.world_model.backbone == "dreamer_v3"


# ----------------------------------------------------------------------
# TruncatedNormal actor
# ----------------------------------------------------------------------


def test_masked_truncated_normal_sample_range_entropy_and_gradient():
    mask = action_mask(2, 3)
    mean = torch.tensor([[0.99, -0.5, 0.3]] * 64, requires_grad=True)
    std = torch.full((64, 3), 1.0, requires_grad=True)
    dist = MaskedTruncatedNormal(mean, std, mask)
    sample = dist.rsample()
    assert sample.abs().max() <= 1.0 - 1e-6
    assert torch.all(sample[:, 2] == 0) and torch.all(dist.mean[:, 2] == 0)
    assert (sample[:, 0].detach() == 1.0 - 1e-6).any()  # some samples hit the clamp
    entropy = dist.entropy()
    assert entropy.shape == (64,) and torch.isfinite(entropy).all()
    sample.sum().backward()
    # Straight-through clamp: d sample / d mean is the mask, clamped or not.
    torch.testing.assert_close(mean.grad, mask.expand(64, 3))
