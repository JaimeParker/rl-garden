from __future__ import annotations

from dataclasses import MISSING, fields
import inspect
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.cli_args import vit_sac_kwargs_from_args
from rl_garden.encoders import RandomShiftsAug, ViTTokenAndPropExtractor, ViTImageEncoder
from rl_garden.encoders.base import TokenAndPropFeatureConfig
from rl_garden.policies import ResidualSACPolicy, SACPolicy


def test_random_shifts_aug_shape_and_dtype():
    aug = RandomShiftsAug(padding=4)
    x = torch.rand(5, 3, 32, 32)
    y = aug(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_vit_token_and_prop_extractor_per_key_concats_patch_dimension():
    obs_space = spaces.Dict(
        {
            "rgb_base": spaces.Box(0, 255, (32, 32, 3), np.uint8),
            "rgb_wrist": spaces.Box(0, 255, (32, 32, 3), np.uint8),
            "state": spaces.Box(-1.0, 1.0, (4,), np.float32),
        }
    )
    extractor = ViTTokenAndPropExtractor(
        obs_space,
        image_keys=("rgb_base", "rgb_wrist"),
        embed_dim=16,
        num_heads=4,
        augmentation="none",
    )
    assert extractor.fusion_mode == "per_key"
    assert set(extractor.image_encoders.keys()) == {"rgb_base", "rgb_wrist"}
    assert extractor.image_encoder is None
    assert extractor.patch_dim == 16
    assert extractor.num_patches == 18
    assert extractor.prop_dim == 4

    obs = {
        "rgb_base": torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8),
        "rgb_wrist": torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8),
        "state": torch.randn(2, 4),
    }
    features = extractor(obs)
    assert features.shape == (2, 18 * 16 + 4)


def test_vit_token_and_prop_extractor_stack_channels_is_explicit():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (32, 32, 3), np.uint8),
            "depth": spaces.Box(0.0, 1.0, (32, 32, 1), np.float32),
        }
    )
    extractor = ViTTokenAndPropExtractor(
        obs_space,
        image_keys=("rgb", "depth"),
        fusion_mode="stack_channels",
        embed_dim=16,
        num_heads=4,
        augmentation="none",
    )
    assert extractor.image_encoder is not None
    assert len(extractor.image_encoders) == 0
    assert extractor.num_patches == 9
    obs = {
        "rgb": torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8),
        "depth": torch.rand(2, 32, 32, 1),
    }
    assert extractor(obs).shape == (2, 9 * 16)


def test_vit_token_and_prop_extractor_structured_config():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (32, 32, 3), np.uint8),
            "state": spaces.Box(-1.0, 1.0, (4,), np.float32),
        }
    )
    extractor = ViTTokenAndPropExtractor(
        obs_space,
        image_keys=("rgb",),
        embed_dim=16,
        num_heads=4,
        augmentation="none",
    )
    sc = extractor.structured_feature_config()
    assert sc is not None
    assert sc["layout"] == "token_and_prop"
    assert sc["num_patches"] == extractor.num_patches
    assert sc["patch_dim"] == extractor.patch_dim
    assert sc["prop_dim"] == extractor.prop_dim
    assert sc["num_patches"] * sc["patch_dim"] + sc["prop_dim"] == extractor.features_dim


def test_vit_token_and_prop_extractor_prepare_batch_caches_features():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (32, 32, 3), np.uint8),
            "state": spaces.Box(-1.0, 1.0, (4,), np.float32),
        }
    )
    extractor = ViTTokenAndPropExtractor(
        obs_space,
        image_keys=("rgb",),
        embed_dim=16,
        num_heads=4,
        augmentation="none",
    )
    obs = {
        "rgb": torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8),
        "state": torch.randn(2, 4),
    }
    next_obs = {
        "rgb": torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8),
        "state": torch.randn(2, 4),
    }
    extractor.prepare_batch(obs, next_obs)
    assert "_vit_features" in obs
    assert "_vit_features" in next_obs
    # forward() returns the cached value
    features = extractor(obs)
    assert features.shape == (2, extractor.features_dim)


def test_vit_image_encoder_returns_vector_features():
    img_space = spaces.Box(0.0, 1.0, (3, 32, 32), np.float32)
    encoder = ViTImageEncoder(
        img_space,
        features_dim=11,
        embed_dim=16,
        num_heads=4,
        augmentation="none",
    )
    y = encoder(torch.rand(2, 3, 32, 32))
    assert y.shape == (2, 11)


def test_sac_policy_with_vit_builds_spatial_critic():
    """SACPolicy auto-selects SpatialEmbQEnsemble when extractor declares layout."""
    from rl_garden.networks import SpatialEmbQEnsemble

    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (32, 32, 3), np.uint8),
            "state": spaces.Box(-1.0, 1.0, (4,), np.float32),
        }
    )
    action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
    extractor = ViTTokenAndPropExtractor(
        obs_space,
        image_keys=("rgb",),
        embed_dim=16,
        num_heads=4,
        augmentation="none",
    )
    policy = SACPolicy(
        obs_space,
        action_space,
        extractor,
        net_arch={"pi": [32], "qf": [32]},
        n_critics=2,
        actor_feature_dim=8,
        critic_spatial_emb_dim=32,
    )
    assert isinstance(policy.critic, SpatialEmbQEnsemble)
    assert isinstance(policy.critic_target, SpatialEmbQEnsemble)
    assert policy._actor_adapter is not None

    obs = {
        "rgb": torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8),
        "state": torch.randn(2, 4),
    }
    extractor.prepare_batch(obs)
    action, log_prob, features = policy.actor_action_log_prob(obs, stop_gradient=True)
    assert action.shape == (2, 3)
    assert log_prob.shape == (2, 1)
    assert policy.q_values_all(features, action).shape == (2, 2, 1)


def test_sac_policy_flat_path_unaffected():
    """SACPolicy uses EnsembleQCritic (flat path) for non-structured extractors."""
    from rl_garden.networks import EnsembleQCritic

    obs_space = spaces.Box(-1.0, 1.0, (8,), np.float32)
    action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
    from rl_garden.encoders import FlattenExtractor

    extractor = FlattenExtractor(obs_space)
    policy = SACPolicy(obs_space, action_space, extractor, n_critics=2)
    assert isinstance(policy.critic, EnsembleQCritic)
    assert policy._actor_adapter is None
    obs = torch.randn(2, 8)
    action, log_prob, features = policy.actor_action_log_prob(obs)
    assert action.shape == (2, 3)


def test_sac_policy_actor_feature_dim_without_structured_config_raises():
    """actor_feature_dim with a flat extractor must raise ValueError."""
    obs_space = spaces.Box(-1.0, 1.0, (8,), np.float32)
    action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
    from rl_garden.encoders import FlattenExtractor

    extractor = FlattenExtractor(obs_space)
    try:
        SACPolicy(obs_space, action_space, extractor, n_critics=2, actor_feature_dim=64)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_residual_sac_policy_with_vit_uses_adapter():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (32, 32, 3), np.uint8),
            "state": spaces.Box(-1.0, 1.0, (4,), np.float32),
        }
    )
    action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
    extractor = ViTTokenAndPropExtractor(
        obs_space,
        image_keys=("rgb",),
        embed_dim=16,
        num_heads=4,
        augmentation="none",
    )
    policy = ResidualSACPolicy(
        obs_space,
        action_space,
        extractor,
        net_arch={"pi": [32], "qf": [32]},
        n_critics=2,
        actor_feature_dim=8,
        critic_spatial_emb_dim=32,
    )
    obs = {
        "rgb": torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8),
        "state": torch.randn(2, 4),
    }
    base_actions = torch.zeros(2, 3)
    extractor.prepare_batch(obs)
    action, log_prob, features = policy.actor_action_log_prob(obs, base_actions)
    assert action.shape == (2, 3)
    assert log_prob.shape == (2, 1)
    assert policy.q_values_all(features, action).shape == (2, 2, 1)


def test_vit_sac_kwargs_from_args_defaults_to_per_key():
    class Args:
        encoder = "vit"
        include_state = True
        vit_fusion_mode = "per_key"
        vit_embed_dim = 128
        vit_depth = 1
        vit_num_heads = 4
        vit_embed_norm = False
        vit_augmentation = "random_shift"
        vit_random_shift_pad = 4
        vit_actor_feature_dim = 128
        vit_critic_spatial_emb_dim = 1024

    kwargs = vit_sac_kwargs_from_args(Args(), ("rgb_base", "rgb_wrist"))
    policy_kwargs = kwargs["policy_kwargs"]
    assert policy_kwargs["features_extractor_class"] is ViTTokenAndPropExtractor
    ext_kwargs = policy_kwargs["features_extractor_kwargs"]
    assert ext_kwargs["fusion_mode"] == "per_key"
    assert ext_kwargs["image_keys"] == ("rgb_base", "rgb_wrist")
    # head hyperparams live at the top level, not in the extractor kwargs
    assert kwargs["actor_feature_dim"] == 128
    assert kwargs["critic_spatial_emb_dim"] == 1024
    assert "actor_feature_dim" not in ext_kwargs
    assert "critic_spatial_emb_dim" not in ext_kwargs


def test_vit_token_and_prop_extractor_has_no_legacy_alias():
    import rl_garden.encoders as encoders

    assert hasattr(encoders, "ViTTokenAndPropExtractor")
    assert not hasattr(encoders, "ViT" + "CombinedExtractor")


def test_sac_policy_sample_actor_actions_uses_adapter():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (32, 32, 3), np.uint8),
            "state": spaces.Box(-1.0, 1.0, (4,), np.float32),
        }
    )
    action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
    extractor = ViTTokenAndPropExtractor(
        obs_space,
        image_keys=("rgb",),
        embed_dim=16,
        num_heads=4,
        augmentation="none",
    )
    policy = SACPolicy(
        obs_space,
        action_space,
        extractor,
        net_arch={"pi": [32], "qf": [32]},
        n_critics=2,
        actor_feature_dim=8,
        critic_spatial_emb_dim=32,
    )
    obs = {
        "rgb": torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8),
        "state": torch.randn(2, 4),
    }
    features = extractor(obs)
    actions, log_probs = policy._sample_actor_actions(features, n=5)
    assert actions.shape == (2, 5, 3)
    assert log_probs.shape == (2, 5)


def test_cql_sample_n_actions_delegates_to_policy_helper():
    from rl_garden.algorithms.cql import CQLCore

    features = torch.randn(2, 7)
    expected_actions = torch.randn(2, 4, 3)
    expected_log_probs = torch.randn(2, 4)

    class Policy:
        def _sample_actor_actions(self, got_features, n):
            assert got_features is features
            assert n == 4
            return expected_actions, expected_log_probs

    self = SimpleNamespace(policy=Policy())
    actions, log_probs = CQLCore._sample_n_actions_with_log_probs(
        self, obs=None, n=4, features=features
    )
    assert actions is expected_actions
    assert log_probs is expected_log_probs


def test_cql_and_wsrl_accept_vit_policy_head_params():
    from rl_garden.algorithms import CQL, WSRL

    for cls in (CQL, WSRL):
        sig = inspect.signature(cls.__init__)
        assert "actor_feature_dim" in sig.parameters
        assert "critic_spatial_emb_dim" in sig.parameters


def test_generate_wsrl_dataset_vit_actor_feature_dim_defaults_to_none():
    from examples.generate_wsrl_dataset import Args

    field = next(f for f in fields(Args) if f.name == "vit_actor_feature_dim")
    assert field.default is None
    assert field.default is not MISSING
