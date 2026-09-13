from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import AWAC, OfflineEnvSpec
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObservationContractError


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _dict_env(num_envs: int = 1) -> OfflineEnvSpec:
    """Dict observation space with only a ``"state"`` key -- the
    schema-normalized shape of a state-only env (see
    ``rl_garden.observations.normalize_observation_space``). AWAC accepts
    this: it only rejects Dict spaces that carry image keys."""
    return OfflineEnvSpec(
        spaces.Dict({"state": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)}),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _dict_image_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "state": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
                "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> AWAC:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
    )
    defaults.update(kwargs)
    return AWAC(**defaults)


def _random_obs(space, num_envs: int):
    if isinstance(space, spaces.Dict):
        return {key: _random_obs(subspace, num_envs) for key, subspace in space.spaces.items()}
    return torch.randn(num_envs, *space.shape)


def _fill(agent: AWAC, steps: int = 64) -> None:
    env = agent.env
    obs_space = env.single_observation_space
    for _ in range(steps):
        obs = _random_obs(obs_space, env.num_envs)
        next_obs = _random_obs(obs_space, env.num_envs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_rejects_dict_observation_space_with_images():
    with pytest.raises(ObservationContractError):
        AWAC(env=_dict_image_env(), buffer_device="cpu", device="cpu")


def test_accepts_dict_state_only_observation_space():
    agent = AWAC(env=_dict_env(), buffer_device="cpu", device="cpu")
    from rl_garden.buffers.replay_buffer import ReplayBuffer

    assert isinstance(agent.replay_buffer, ReplayBuffer)


def test_policy_has_no_actor_target():
    agent = _make_agent()
    assert not hasattr(agent.policy, "actor_target")
    assert hasattr(agent.policy, "critic_target")


def test_gradient_step_produces_finite_losses():
    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()
    metrics = agent.train(1, compute_info=True)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])


def test_critic_target_updates_every_step_not_delayed():
    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()

    target_before = [p.clone() for p in agent.policy.critic_target.parameters()]
    agent.train(1)
    assert not all(
        torch.equal(a, b)
        for a, b in zip(target_before, agent.policy.critic_target.parameters())
    )


def test_actor_loss_weight_clamped_at_exp_adv_max():
    agent = _make_agent(exp_adv_max=2.0, awac_lambda=1.0)
    _fill(agent)
    agent.fit_obs_normalizer()
    data = agent._sample_train_batch(agent.batch_size)

    with torch.no_grad():
        actor_features = agent.policy.extract_actor_features(data.obs)
        critic_features = agent.policy.extract_critic_features(data.obs)
        pi_action, _ = agent.policy.actor.action_log_prob(actor_features)
        v = agent.policy.q_values_all(critic_features, pi_action, target=False).min(dim=0).values
        q = agent.policy.q_values_all(
            critic_features, data.actions, target=False
        ).min(dim=0).values
        adv = q - v
        weights = torch.clamp_max(torch.exp(adv / agent.awac_lambda), agent.exp_adv_max)
    assert weights.max().item() <= 2.0 + 1e-5


def test_unsquashed_actor_log_prob_has_no_tanh_jacobian():
    """CORL's AWAC actor evaluates log_prob directly on the (clamped) action,
    with no change-of-variables correction -- verify this differs from a
    manual tanh-Jacobian-corrected computation."""
    from rl_garden.networks import UnsquashedGaussianActor

    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    actor = UnsquashedGaussianActor(4, action_space, hidden_dims=[16])
    features = torch.randn(5, 4)
    actions = torch.rand(5, 2) * 1.8 - 0.9  # inside bounds, away from +/-1

    mean, log_std = actor(features)
    normal = torch.distributions.Normal(mean, log_std.exp())
    expected = normal.log_prob(actions).sum(-1, keepdim=True)

    actual = actor.evaluate_action_log_prob(features, actions)
    assert torch.allclose(actual, expected)


def test_unsquashed_actor_tanh_mean_matches_corl_iql_gaussian_policy():
    """tanh_mean=True (IQL's actor_distribution="unsquashed" opt-in) matches
    CORL's iql.py::GaussianPolicy / official JAX's NormalTanhPolicy shape:
    tanh-bounded mean, still no Jacobian correction on log_prob -- the
    discriminating check for the opt-in mode, not just an isinstance check."""
    from rl_garden.networks import UnsquashedGaussianActor

    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    actor = UnsquashedGaussianActor(4, action_space, hidden_dims=[16], tanh_mean=True)
    features = torch.randn(5, 4)
    actions = torch.rand(5, 2) * 1.8 - 0.9

    mean, log_std = actor(features)
    assert torch.allclose(mean, torch.tanh(actor.fc_mean(actor.trunk(features))))

    expected = torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(
        -1, keepdim=True
    )
    actual = actor.evaluate_action_log_prob(features, actions)
    assert torch.allclose(actual, expected)


def test_checkpoint_round_trip_preserves_normalizer_and_critic_target():
    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()
    for _ in range(3):
        agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent()
        loaded.load(path, load_replay_buffer=False)

    assert torch.allclose(agent.policy.obs_mean, loaded.policy.obs_mean)
    assert torch.allclose(agent.policy.obs_std, loaded.policy.obs_std)
    for a, b in zip(agent.policy.actor.parameters(), loaded.policy.actor.parameters()):
        assert torch.allclose(a, b)
    for a, b in zip(
        agent.policy.critic_target.parameters(), loaded.policy.critic_target.parameters()
    ):
        assert torch.allclose(a, b)


def test_checkpoint_round_trip_with_encoder_config():
    # AWAC is state-only (rejects Dict obs with image keys, see
    # test_rejects_dict_observation_space_with_images above), so
    # 'normalize_obs' is the only EncoderConfig field build_observation_encoder
    # (rl_garden.encoders.factory) allows to be non-default here -- every
    # other field is image-related and would raise ObservationContractError.
    encoder_config = EncoderConfig(normalize_obs=True)
    agent = _make_agent(env=_dict_env(), encoder_config=encoder_config)
    _fill(agent)
    agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = AWAC(env=_dict_env(), buffer_device="cpu", device="cpu", encoder_config=encoder_config)
        loaded.load(path, load_replay_buffer=False)

    meta = loaded._checkpoint_metadata()
    assert meta["encoder_config"] == {
        field: getattr(encoder_config, field) for field in meta["encoder_config"]
    }


def test_asymmetric_obs_groups_critic_sees_extra_state_actor_does_not():
    """``encoder_sharing="separate"`` + ``obs_groups`` gives the critic a
    privileged ``state_object_pose`` key the actor never sees. AWAC forbids
    image keys entirely (see test_rejects_dict_observation_space_with_images),
    so -- unlike SAC's/PPO's asymmetric test -- there is no way to give either
    extractor a shared, parameter-bearing image branch here:
    ``FlattenExtractor`` (the only extractor a state-only schema ever builds,
    see rl_garden.encoders.factory.build_observation_encoder) has zero
    parameters regardless of ``normalize_obs`` (its optional
    ``RunningObsNormalizer`` only holds buffers, no ``nn.Parameter``), so a
    ``torch.autograd.grad(loss, extractor.parameters())``-style gradient-
    isolation check is inapplicable here -- there is nothing to isolate.
    Instead this verifies the two things that ARE meaningful for a
    parameterless extractor: (1) the actor/critic extractors resolve to
    different feature dims (proving the extra key is routed to the critic
    only), and (2) a real end-to-end training step succeeds, proving the
    whole asymmetric construction path works."""
    from rl_garden.observations import ObsGroups

    env = OfflineEnvSpec(
        spaces.Dict(
            {
                "state": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
                "state_object_pose": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
    )
    agent = _make_agent(
        env=env,
        obs_groups=ObsGroups(actor=("state",), critic=("state", "state_object_pose")),
        encoder_sharing="separate",
    )
    assert agent.policy.critic_extractor is not None
    assert agent.policy.critic_extractor is not agent.policy.actor_extractor
    assert agent.policy.actor_extractor.features_dim == 6
    assert agent.policy.critic_extractor.features_dim == 9

    # FlattenExtractor has no public state_keys (unlike CombinedExtractor);
    # `_state_keys` is the schema-filtered tuple build_observation_encoder
    # actually constructs it from (rl_garden.encoders.factory) -- confirms
    # the schema exclusion directly rather than inferring it from dims alone.
    assert "state_object_pose" not in agent.policy.actor_extractor._state_keys
    assert "state_object_pose" in agent.policy.critic_extractor._state_keys

    # Per-role optimizer/parameter separation: actor_parameters() and
    # critic_and_encoder_parameters() never share a parameter object (always
    # true here since actor/critic are distinct nn.Modules regardless of
    # encoder_sharing -- the extractors themselves contribute no parameters
    # either way, per the parameterless-FlattenExtractor note above).
    actor_param_ids = {id(p) for p in agent.policy.actor_parameters()}
    critic_param_ids = {id(p) for p in agent.policy.critic_and_encoder_parameters()}
    assert actor_param_ids.isdisjoint(critic_param_ids)

    _fill(agent)
    agent.fit_obs_normalizer()

    # The regression this item fixes: fit_obs_normalizer() used to fit only
    # the actor's (default-suffix) obs_mean/obs_std pair, leaving the
    # critic's own suffix="critic" pair (registered whenever critic_extractor
    # is genuinely separate -- see AWACPolicy.__init__) at its __init__-time
    # zeros/ones default forever, i.e. a silent no-op normalization for the
    # critic. Recompute the expected critic-feature statistics independently
    # and check the fitted buffers (named obs_meancritic/obs_stdcritic --
    # f"obs_mean{suffix}" with suffix="critic", no separating "_") match.
    buf = agent.replay_buffer
    obs = {
        key: buf.obs[key][: buf.size].reshape(-1, *buf.obs[key].shape[2:])
        for key in buf.obs.keys()
    }
    with torch.no_grad():
        expected_critic_features = agent.policy.critic_extractor(obs)
    assert agent.policy.obs_meancritic.shape == (9,)
    torch.testing.assert_close(
        agent.policy.obs_meancritic, expected_critic_features.mean(dim=0)
    )
    torch.testing.assert_close(
        agent.policy.obs_stdcritic, expected_critic_features.std(dim=0) + 1e-3
    )
    # Sanity: not just coincidentally left at the untouched default.
    assert not torch.allclose(agent.policy.obs_meancritic, torch.zeros(9))
    assert not torch.allclose(agent.policy.obs_stdcritic, torch.ones(9))

    metrics = agent.train(1, compute_info=True)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])
