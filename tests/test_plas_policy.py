from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.plas_policy import PLASPolicy

OBS_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


def _make_policy(**kwargs) -> PLASPolicy:
    fe = FlattenExtractor(observation_space=OBS_SPACE)
    defaults = dict(net_arch=[16, 16], vae_hidden_dim=16)
    defaults.update(kwargs)
    return PLASPolicy(OBS_SPACE, ACT_SPACE, fe, **defaults)


def test_vae_is_a_real_submodule_in_state_dict():
    policy = _make_policy()
    keys = policy.state_dict().keys()
    assert any(k.startswith("vae.") for k in keys)


def test_latent_actor_output_dim_matches_vae_latent_dim():
    policy = _make_policy()
    features = torch.randn(4, policy.features_extractor.features_dim)
    latent = policy.latent_actor(features)
    assert latent.shape == (4, policy.vae.latent_dim)


def test_latent_actor_output_bounded_by_max_latent_action():
    policy = _make_policy(max_latent_action=0.3)
    features = torch.randn(16, policy.features_extractor.features_dim)
    latent = policy.latent_actor(features)
    assert torch.all(latent.abs() <= 0.3 + 1e-6)


def test_vae_parameters_are_distinct_from_actor_critic_parameters():
    policy = _make_policy()
    vae_ids = {id(p) for p in policy.vae_parameters()}
    actor_ids = {id(p) for p in policy.actor_parameters()}
    critic_ids = {id(p) for p in policy.critic_and_encoder_parameters()}
    assert vae_ids.isdisjoint(actor_ids)
    assert vae_ids.isdisjoint(critic_ids)
    assert len(vae_ids) > 0


def test_train_eval_keeps_vae_in_eval_mode():
    policy = _make_policy()
    policy.train(True)
    assert policy.vae.training is False
    assert policy.latent_actor.training is True

    policy.eval()
    assert policy.vae.training is False
    assert policy.latent_actor.training is False


def test_predict_in_bounds():
    policy = _make_policy()
    obs = torch.randn(4, 6)
    action = policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_predict_is_deterministic_no_sampling():
    """Unlike BCQ, PLAS's eval action is a single deterministic pass -- two
    calls with the same obs and no intervening training must match exactly."""
    policy = _make_policy()
    obs = torch.randn(4, 6)
    action_a = policy.predict(obs)
    action_b = policy.predict(obs)
    assert torch.equal(action_a, action_b)


def test_without_perturbation_action_is_plain_vae_decode():
    policy = _make_policy(use_perturbation=False)
    assert policy.perturbation is None
    features = torch.randn(4, policy.features_extractor.features_dim)
    latent = policy.latent_actor(features)
    expected = policy.vae.decode(features, z=latent)
    actual = policy.action_from_latent(features, latent, target=False)
    assert torch.equal(actual, expected)


def test_with_perturbation_action_differs_from_plain_decode():
    policy = _make_policy(use_perturbation=True, phi=0.5)
    features = torch.randn(4, policy.features_extractor.features_dim)
    latent = policy.latent_actor(features)
    decoded = policy.vae.decode(features, z=latent)
    perturbed = policy.action_from_latent(features, latent, target=False)
    assert not torch.equal(perturbed, decoded)
    assert torch.all(perturbed >= -1.0) and torch.all(perturbed <= 1.0)


def test_perturbation_target_is_a_frozen_separate_copy():
    policy = _make_policy(use_perturbation=True)
    assert policy.perturbation_target is not None
    for p in policy.perturbation_target.parameters():
        assert p.requires_grad is False
    live_ids = {id(p) for p in policy.perturbation.parameters()}
    target_ids = {id(p) for p in policy.perturbation_target.parameters()}
    assert live_ids.isdisjoint(target_ids)
