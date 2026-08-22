from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.bcq_policy import BCQPolicy

OBS_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)


def _make_policy(**kwargs) -> BCQPolicy:
    fe = FlattenExtractor(observation_space=OBS_SPACE)
    defaults = dict(net_arch=[16, 16], vae_hidden_dim=16)
    defaults.update(kwargs)
    return BCQPolicy(OBS_SPACE, ACT_SPACE, fe, **defaults)


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


def test_predict_in_bounds():
    policy = _make_policy()
    obs = torch.randn(4, 6)
    action = policy.predict(obs, num_candidates=8)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_predict_picks_the_max_q_candidate_per_row():
    """This grouping/argmax-gather logic has no upstream reference (BCQ's
    own select_action is single-state only) -- it's rl-garden-authored code
    that must be tested directly rather than trusted by inspection. Stub the
    critic's q1 output with a distinct, known-argmax value per (row,
    candidate) pair and assert predict() picks exactly that candidate for
    every row."""
    policy = _make_policy()
    batch_size, num_candidates = 3, 5

    def fake_q_values(features, actions, target=False):
        del target
        # features carries a per-row index baked into extract_features's
        # normalized output; recover which (row, candidate) pair each of the
        # batch*num_candidates flattened rows is, then hand back a Q value
        # that is highest at a distinct, deliberately-chosen candidate index
        # per row -- e.g. row r's max is at candidate index r.
        n = features.shape[0]
        row_idx = torch.arange(n) // num_candidates
        candidate_idx = torch.arange(n) % num_candidates
        best_candidate_for_row = row_idx % num_candidates
        q1 = -((candidate_idx - best_candidate_for_row).float().abs()).unsqueeze(-1)
        q2 = q1.clone()
        return q1, q2

    policy.q_values = fake_q_values

    obs = torch.randn(batch_size, 6)

    # vae.decode(z=None) samples from the prior, so the manual recomputation
    # below must consume the RNG in exactly the same order as predict() --
    # reset the seed before each of the two identical call sequences.
    torch.manual_seed(42)
    action = policy.predict(obs, num_candidates=num_candidates)
    assert action.shape == (batch_size, 3)

    # Recompute what each row's actual sampled/perturbed candidates were, to
    # verify the returned action for row r really is candidate index (r %
    # num_candidates), not some other row's or some other candidate's action.
    torch.manual_seed(42)
    features = policy.extract_features(obs)
    tiled_features = features.repeat_interleave(num_candidates, dim=0)
    sampled_actions = policy.vae.decode(tiled_features, clip=0.5)
    perturbed_actions = policy.actor(tiled_features, sampled_actions)
    perturbed_actions = perturbed_actions.reshape(batch_size, num_candidates, 3)

    for row in range(batch_size):
        expected_candidate = row % num_candidates
        assert torch.equal(action[row], perturbed_actions[row, expected_candidate])


def test_train_keeps_vae_trainable_not_forced_eval():
    """Regression guard vs. SPOTPolicy's pattern: BCQPolicy must NOT force
    self.vae.eval() in train() -- the VAE is a live, jointly trained network."""
    policy = _make_policy()
    policy.train(True)
    assert policy.vae.training is True
    assert policy.actor.training is True
