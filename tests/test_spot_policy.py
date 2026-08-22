from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.spot_policy import SPOTPolicy

OBS_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


def _make_policy(**kwargs) -> SPOTPolicy:
    fe = FlattenExtractor(observation_space=OBS_SPACE)
    defaults = dict(net_arch=[16, 16], vae_hidden_dim=16)
    defaults.update(kwargs)
    return SPOTPolicy(OBS_SPACE, ACT_SPACE, fe, **defaults)


def test_vae_is_a_real_submodule_in_state_dict():
    policy = _make_policy()
    keys = policy.state_dict().keys()
    assert any(k.startswith("vae.") for k in keys)


def test_vae_default_latent_dim_matches_action_dim():
    policy = _make_policy()
    assert policy.vae.latent_dim == 2 * 3


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
    assert policy.actor.training is True

    policy.eval()
    assert policy.vae.training is False
    assert policy.actor.training is False


def test_reuses_td3bc_actor_critic_predict():
    policy = _make_policy()
    obs = torch.randn(4, 6)
    action = policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
