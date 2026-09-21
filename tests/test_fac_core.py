from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import FAC, OfflineEnvSpec
from rl_garden.encoders.config import EncoderConfig

_TEST_IMAGE_SIZE = 16
_test_encoder_config = EncoderConfig(features_dim=16, plain_conv_pooling="gap")

_BC_METRIC_KEYS = ("bc_flow_loss",)
_AC_METRIC_KEYS = (
    "critic_loss",
    "td_loss",
    "critic_penalty",
    "est_logpi_mean",
    "est_logpi_max",
    "est_logpi_min",
    "est_logbeta_mean",
    "est_logbeta_max",
    "est_logbeta_min",
    "penalty_weight_mean",
    "actor_loss",
    "distill_loss",
    "q_loss",
)


def _state_env(num_envs: int = 1, action_dim: int = 3) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        num_envs=num_envs,
    )


def _vision_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "rgb_cam": spaces.Box(
                    low=0, high=255, shape=(_TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3), dtype=np.uint8
                ),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> FAC:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=8,
        device="cpu",
        net_arch=[16, 16],
        flow_steps=2,
        bc_pretrain_steps=3,
        bc_batch_size=8,
    )
    defaults.update(kwargs)
    return FAC(**defaults)


def _fill(agent: FAC, steps: int = 64) -> None:
    env = agent.env
    state_shape = env.single_observation_space["state"].shape
    for _ in range(steps):
        obs = {"state": torch.randn(env.num_envs, *state_shape)}
        next_obs = {"state": torch.randn_like(obs["state"])}
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _fill_vision(agent: FAC, steps: int = 64) -> None:
    env = agent.env
    obs_space = env.single_observation_space
    img_shape = (_TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3)
    for _ in range(steps):
        obs = {
            "rgb_cam": torch.randint(0, 256, (env.num_envs, *img_shape), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, *obs_space["state"].shape),
        }
        next_obs = {
            "rgb_cam": torch.randint(0, 256, (env.num_envs, *img_shape), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, *obs_space["state"].shape),
        }
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    ids: set[int] = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            ids.add(id(param))
    return ids


# --- construction / validation ---


def test_rejects_unsupported_observation_space():
    unsupported = OfflineEnvSpec(
        spaces.MultiDiscrete([3, 3]),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=1,
    )
    with pytest.raises(ValueError, match="Box or Dict"):
        FAC(env=unsupported, buffer_device="cpu", device="cpu")


def test_invalid_fac_threshold_raises():
    with pytest.raises(ValueError, match="fac_threshold"):
        _make_agent(fac_threshold="bogus")


def test_invalid_logp_method_raises():
    with pytest.raises(ValueError, match="logp_method"):
        _make_agent(logp_method="bogus")


def test_invalid_weight_type_raises():
    with pytest.raises(ValueError, match="weight_type"):
        _make_agent(weight_type="bogus")


def test_without_replace_and_batch_adaptive_raises_at_construction():
    with pytest.raises(ValueError, match="offline_sampling"):
        _make_agent(offline_sampling="without_replace", fac_threshold="batch_adaptive")


def test_without_replace_and_batch_wide_constant_raises_at_construction():
    with pytest.raises(ValueError, match="offline_sampling"):
        _make_agent(offline_sampling="without_replace", fac_threshold="batch_wide_constant")


def test_without_replace_and_dataset_wide_constant_trains_one_call():
    # dataset_wide_constant needs no per-sample cached-logp gather, so it is
    # the one fac_threshold allowed under offline_sampling="without_replace"
    # (see FACCore's module docstring, "Deviations"). FQLCore._sample_train_batch's
    # sample_without_repeat lookup (rl_garden/buffers/_sampling.py) is now
    # fixed upstream, so this drives a full train() call end to end instead
    # of only checking construction-time validation.
    agent = _make_agent(
        offline_sampling="without_replace",
        fac_threshold="dataset_wide_constant",
        bc_pretrain_steps=1,
    )
    assert agent.fac_threshold == "dataset_wide_constant"
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in _AC_METRIC_KEYS:
        assert np.isfinite(metrics[key]), (key, metrics[key])


# --- encoder_sharing default ---


def test_encoder_sharing_defaults_to_separate_state_env():
    agent = _make_agent()
    assert agent.encoder_sharing == "separate"


def test_encoder_sharing_defaults_to_separate_vision_env():
    agent = _make_agent(env=_vision_env(), encoder_config=_test_encoder_config)
    assert agent.encoder_sharing == "separate"


def test_default_separate_encoder_sharing_trains_both_phases_state_env():
    agent = _make_agent(bc_pretrain_steps=1)
    assert agent.encoder_sharing == "separate"
    _fill(agent)
    # One train(1) call now runs the whole BC prelude, builds the transition
    # logp table, then does exactly one actor-critic update.
    metrics = agent.train(1, compute_info=True)
    assert agent.phase == "actor_critic"
    for key in _AC_METRIC_KEYS:
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_explicit_shared_critic_grad_state_env_constructs_and_trains():
    agent = _make_agent(encoder_sharing="shared_critic_grad", bc_pretrain_steps=0)
    assert agent.encoder_sharing == "shared_critic_grad"
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in _AC_METRIC_KEYS:
        assert np.isfinite(metrics[key]), (key, metrics[key])


# --- phase gate ---


def test_phase_a_updates_only_bc_params():
    # train() now always runs the WHOLE remaining BC prelude before any AC
    # step (see FACCore.train()'s docstring), so a single BC-only update is
    # isolated by calling _bc_update directly rather than through train().
    agent = _make_agent(bc_pretrain_steps=5)
    _fill(agent)
    assert agent.phase == "bc_pretrain"

    bc_before = [p.clone() for p in agent.policy.actor_bc_flow.parameters()]
    onestep_before = [p.clone() for p in agent.policy.actor_onestep_flow.parameters()]
    critic_before = [p.clone() for p in agent.policy.critic.parameters()]

    agent.policy.train()
    data = agent._sample_train_batch(agent._resolve_bc_batch_size())
    metrics = agent._bc_update(data)

    for key in _BC_METRIC_KEYS:
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert not all(
        torch.equal(a, b) for a, b in zip(bc_before, agent.policy.actor_bc_flow.parameters())
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(onestep_before, agent.policy.actor_onestep_flow.parameters())
    )
    assert all(torch.equal(a, b) for a, b in zip(critic_before, agent.policy.critic.parameters()))


def test_train_one_call_completes_prelude_and_builds_table():
    agent = _make_agent(bc_pretrain_steps=5, buffer_size=200)
    _fill(agent, steps=50)

    agent.train(1)  # prelude (5 BC steps) + transition (builds the table) + 1 AC step
    assert agent._phase_step == 5
    assert agent._logp_table is not None
    assert agent.global_update == 1


def test_second_train_call_does_not_rerun_prelude():
    agent = _make_agent(bc_pretrain_steps=2)
    _fill(agent)
    agent.train(1)  # prelude (2 BC steps) + transition + 1 AC step
    assert agent._phase_step == 2
    assert agent.global_update == 1

    def _must_not_run_bc(*args, **kwargs):
        raise AssertionError("_bc_update must not run again once the prelude has completed")

    agent._bc_update = _must_not_run_bc
    agent.train(3)  # pure AC: prelude is already done, so _bc_update above must stay unused
    assert agent.global_update == 4
    assert agent._phase_step == 2  # the BC-progress counter is frozen once the prelude finishes


def test_transition_builds_finite_logp_table():
    agent = _make_agent(bc_pretrain_steps=1, buffer_size=200)
    _fill(agent, steps=50)

    assert agent._logp_table is None
    agent.train(1)  # prelude (1 BC step) + transition (builds the table) + 1 AC step
    table = agent._logp_table
    assert table is not None
    assert table.shape == (200, 1)  # per_env_buffer_size, num_envs

    finite_mask = torch.isfinite(table)
    assert bool(finite_mask[:50, 0].all())
    assert not bool(finite_mask[50:, 0].any())
    assert math.isfinite(agent._logp_min)


def test_phase_b_updates_critic_and_onestep_not_bc():
    agent = _make_agent(bc_pretrain_steps=1)
    _fill(agent)
    agent.train(1)  # prelude + transition + first AC step, all in this one call
    assert agent.phase == "actor_critic"

    bc_before = [p.clone() for p in agent.policy.actor_bc_flow.parameters()]
    onestep_before = [p.clone() for p in agent.policy.actor_onestep_flow.parameters()]
    critic_before = [p.clone() for p in agent.policy.critic.parameters()]

    metrics = agent.train(1, compute_info=True)  # prelude already done: pure AC step

    assert agent.phase == "actor_critic"
    for key in _AC_METRIC_KEYS:
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])
    assert all(
        torch.equal(a, b) for a, b in zip(bc_before, agent.policy.actor_bc_flow.parameters())
    )
    assert not all(
        torch.equal(a, b)
        for a, b in zip(onestep_before, agent.policy.actor_onestep_flow.parameters())
    )
    assert not all(
        torch.equal(a, b) for a, b in zip(critic_before, agent.policy.critic.parameters())
    )


def test_phase_property_values():
    agent = _make_agent(bc_pretrain_steps=2)
    assert agent.phase == "bc_pretrain"
    _fill(agent)
    agent.train(2)
    assert agent.phase == "actor_critic"


# --- _logp_beta gather against a hand-injected table ---


def test_logp_beta_gathers_against_known_table():
    agent = _make_agent(bc_pretrain_steps=1, buffer_size=200)
    _fill(agent, steps=50)
    agent.train(1)  # prelude + transition (builds a real table, about to be overwritten) + 1 AC step

    per_env_buffer_size, num_envs = agent._logp_table.shape
    known_table = torch.arange(per_env_buffer_size * num_envs, dtype=torch.float32).view(
        per_env_buffer_size, num_envs
    )
    agent._logp_table = known_table.to(agent.device)
    agent._logp_min = float(known_table.min().item())

    batch_inds = torch.tensor([0, 5, 10, 49])
    env_inds = torch.tensor([0, 0, 0, 0])
    batch_size = batch_inds.shape[0]
    expected_gathered = known_table[batch_inds, env_inds].to(agent.device)

    agent.fac_threshold = "batch_adaptive"
    result = agent._logp_beta(batch_inds, env_inds, batch_size)
    assert result.device == agent.device
    assert result.dtype == torch.float32
    assert torch.equal(result, expected_gathered)

    agent.fac_threshold = "batch_wide_constant"
    result = agent._logp_beta(batch_inds, env_inds, batch_size)
    expected_min = expected_gathered.min().expand(batch_size)
    assert result.device == agent.device
    assert result.dtype == torch.float32
    assert torch.equal(result, expected_min)

    agent.fac_threshold = "dataset_wide_constant"
    result = agent._logp_beta(batch_inds, env_inds, batch_size)
    expected_constant = torch.full((batch_size,), agent._logp_min, dtype=torch.float32)
    assert result.device == agent.device
    assert result.dtype == torch.float32
    assert torch.equal(result, expected_constant)


# --- threshold / weight-type combinations ---


@pytest.mark.parametrize(
    "fac_threshold", ["batch_adaptive", "batch_wide_constant", "dataset_wide_constant"]
)
@pytest.mark.parametrize("weight_type", ["linear", "logarithmic", "convex", "concave"])
def test_threshold_and_weight_combinations_produce_finite_bounded_penalty(
    fac_threshold, weight_type
):
    agent = _make_agent(bc_pretrain_steps=0, fac_threshold=fac_threshold, weight_type=weight_type)
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in _AC_METRIC_KEYS:
        assert np.isfinite(metrics[key]), (key, metrics[key])
    assert 0.0 <= metrics["penalty_weight_mean"] <= 1.0


def test_fac_alpha_zero_critic_loss_equals_td_loss():
    agent = _make_agent(bc_pretrain_steps=0, fac_alpha=0.0)
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    assert metrics["critic_loss"] == pytest.approx(metrics["td_loss"])


def _expected_linear_weight(d: float) -> float:
    return min(max(1.0 - math.exp(d), 0.0), 1.0)


def _expected_temp_weight(d: float, temp: float) -> float:
    return min(max(1.0 - math.log1p(math.exp(d * temp)) / math.log(2.0), 0.0), 1.0)


def test_penalty_weight_numeric_pins():
    agent = _make_agent()
    diff_values = [-2.0, -0.5, 0.0, 0.5]
    diff = torch.tensor(diff_values)

    agent.weight_type = "linear"
    expected = torch.tensor([_expected_linear_weight(d) for d in diff_values])
    assert torch.allclose(agent._penalty_weight(diff), expected, atol=1e-6)

    temps = {"logarithmic": 1.0, "convex": 0.5, "concave": 2.0}
    for weight_type, temp in temps.items():
        agent.weight_type = weight_type
        expected = torch.tensor([_expected_temp_weight(d, temp) for d in diff_values])
        assert torch.allclose(agent._penalty_weight(diff), expected, atol=1e-6), weight_type

    # A swapped convex/concave temperature must be caught: the two must
    # differ at d=-1.0 (clip(1-exp(d),0,1) territory where neither is
    # saturated to 0 or 1).
    d_single = torch.tensor([-1.0])
    agent.weight_type = "convex"
    w_convex = agent._penalty_weight(d_single).item()
    agent.weight_type = "concave"
    w_concave = agent._penalty_weight(d_single).item()
    assert w_convex != pytest.approx(w_concave)


def test_fac_alpha_penalizes_q_at_own_penalty_actions():
    """End-to-end penalty-direction check: two agents with identical seeds,
    identical replay data, and identical training-time random draws (only
    ``fac_alpha`` differs) must end up with the high-alpha agent assigning a
    lower mean Q to its own one-step actor's ("penalty") actions."""
    seed = 0

    def _build(fac_alpha: float):
        agent = _make_agent(fac_alpha=fac_alpha, bc_pretrain_steps=1, seed=seed)
        torch.manual_seed(seed + 1)
        _fill(agent, steps=32)
        # Force weight ~= 1 (logp_pi - logp_beta pinned very large) so the
        # penalty term's magnitude is driven by fac_alpha alone, not by
        # whatever the (identical, since Phase A is fac_alpha-independent)
        # cached logp values happen to be.
        agent._logp_beta = lambda batch_inds, env_inds, batch_size, agent=agent: torch.full(
            (batch_size,), -1e6, device=agent.device, dtype=torch.float32
        )
        return agent

    agent_zero = _build(0.0)
    agent_five = _build(5.0)

    torch.manual_seed(seed + 2)
    agent_zero.train(31)  # prelude: 1 BC step (bc_pretrain_steps=1); then 31 AC steps
    torch.manual_seed(seed + 2)
    agent_five.train(31)

    torch.manual_seed(seed + 3)
    eval_obs = {"state": torch.randn(64, 6)}

    def _mean_q_at_own_penalty_action(agent) -> float:
        with torch.no_grad():
            features = agent.policy.extract_critic_features(eval_obs)
            noise = agent.policy.sample_noise(64, device=features.device, dtype=features.dtype)
            a_pen = agent.policy.actor_onestep_flow(features, noise).clamp(
                agent.policy.action_low, agent.policy.action_high
            )
            q_all = agent.policy.q_values_all(features, a_pen, target=False)
            return float(q_all.mean().item())

    q_zero = _mean_q_at_own_penalty_action(agent_zero)
    q_five = _mean_q_at_own_penalty_action(agent_five)
    assert q_five < q_zero


# --- bc_pretrain_steps / bc_batch_size resolution ---


def test_bc_pretrain_steps_none_resolves_from_epochs_and_buffer_size():
    agent = _make_agent(bc_pretrain_steps=None, bc_pretrain_epochs=2, bc_batch_size=8)
    _fill(agent, steps=64)
    resolved = agent._resolve_bc_pretrain_steps()
    expected = 2 * math.ceil(64 / 8)
    assert resolved == expected
    assert agent._bc_pretrain_steps_resolved == expected


def test_bc_batch_size_none_resolves_from_reference_heuristic():
    agent = _make_agent(bc_batch_size=None)
    _fill(agent, steps=64)  # epoch_size = 64 < 100_000 -> multiplier 1
    assert agent._resolve_bc_batch_size() == agent.batch_size


# --- optimizer parameter disjointness ---


def test_optimizer_params_pairwise_disjoint_state_env():
    agent = _make_agent()
    critic_ids = _param_ids(agent.critic_optimizer)
    actor_ids = _param_ids(agent.actor_optimizer)
    bc_ids = _param_ids(agent.bc_optimizer)
    assert critic_ids.isdisjoint(actor_ids)
    assert critic_ids.isdisjoint(bc_ids)
    assert actor_ids.isdisjoint(bc_ids)


def test_optimizer_params_pairwise_disjoint_vision_separate_encoder():
    agent = _make_agent(
        env=_vision_env(), encoder_sharing="separate", encoder_config=_test_encoder_config
    )
    critic_ids = _param_ids(agent.critic_optimizer)
    actor_ids = _param_ids(agent.actor_optimizer)
    bc_ids = _param_ids(agent.bc_optimizer)
    assert critic_ids.isdisjoint(actor_ids)
    assert critic_ids.isdisjoint(bc_ids)
    assert actor_ids.isdisjoint(bc_ids)

    _fill_vision(agent, steps=32)
    # bc_pretrain_steps defaults to 3 (see _make_agent), so this single
    # train(1) call runs the whole BC prelude first and then exactly one AC
    # step; the returned (averaged) metrics therefore only cover that AC step.
    metrics = agent.train(1, compute_info=True)
    for key in _AC_METRIC_KEYS:
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_vision_shared_encoder_smoke():
    agent = _make_agent(
        env=_vision_env(),
        encoder_sharing="shared_critic_grad",
        encoder_config=_test_encoder_config,
        bc_pretrain_steps=0,
    )
    _fill_vision(agent, steps=32)
    metrics = agent.train(1, compute_info=True)
    for key in _AC_METRIC_KEYS:
        assert np.isfinite(metrics[key]), (key, metrics[key])
    assert not hasattr(agent.policy, "actor_bc_flow_encoder")


# --- online fine-tuning (_online_finetuning, set directly here since plain
# FAC has no Off2OnReplayMixin/switch_to_online_mode -- Off2OnFAC's shell
# flips this same FACCore-owned flag through _apply_online_regularizer_override,
# exercised end to end in tests/test_off2on_fac_smoke.py) ---


def test_online_finetuning_critic_update_uses_logp_beta_online_not_table():
    agent = _make_agent(bc_pretrain_steps=1)
    _fill(agent)
    agent.train(1)  # offline prelude + transition + 1 AC step, builds _logp_table
    assert agent._logp_table is not None

    agent._online_finetuning = True
    calls = []
    real_online = agent._logp_beta_online

    def _spy_online(data):
        calls.append(data)
        return real_online(data)

    def _must_not_run_offline(*args, **kwargs):
        raise AssertionError("_logp_beta (offline, table-gathering) must not run online")

    agent._logp_beta_online = _spy_online
    agent._logp_beta = _must_not_run_offline

    metrics = agent.train(1, compute_info=True)
    assert len(calls) == 1
    assert "bc_flow_loss" in metrics
    for key in _AC_METRIC_KEYS:
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_online_finetuning_skips_prelude_and_table_for_untrained_agent():
    # bc_pretrain_steps huge: if train() ever ran the offline prelude here it
    # would loop until _phase_step reaches it, hanging/failing this test.
    agent = _make_agent(bc_pretrain_steps=1_000_000, bc_batch_size=8)
    _fill(agent, steps=16)
    agent._online_finetuning = True

    assert agent._logp_table is None
    metrics = agent.train(2, compute_info=True)
    assert agent._logp_table is None
    assert agent._phase_step == 0
    assert agent.global_update == 2
    for key in _AC_METRIC_KEYS:
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_logp_beta_online_batch_adaptive_keeps_per_sample_values():
    from rl_garden.networks.flow_logprob import flow_log_prob

    agent = _make_agent(bc_pretrain_steps=1, fac_threshold="batch_adaptive")
    _fill(agent)
    agent.train(1)
    agent._online_finetuning = True

    data = agent._sample_train_batch(agent.batch_size)
    with torch.no_grad():
        bc_features = agent._bc_features(data.obs)
        expected = flow_log_prob(
            agent.policy.actor_bc_flow,
            bc_features,
            data.actions,
            num_steps=agent.flow_steps,
            method=agent.logp_method,
            num_probes=agent.logp_hutch_probes,
        ).to(device=agent.device, dtype=torch.float32)

    result = agent._logp_beta_online(data)
    assert result.shape == (data.actions.shape[0],)
    assert torch.allclose(result, expected)
    # Not collapsed: more than one distinct value across the batch.
    assert result.unique().numel() > 1


def test_logp_beta_online_batch_wide_and_dataset_wide_collapse_to_batch_min():
    from rl_garden.networks.flow_logprob import flow_log_prob

    for fac_threshold in ("batch_wide_constant", "dataset_wide_constant"):
        agent = _make_agent(bc_pretrain_steps=1, fac_threshold=fac_threshold)
        _fill(agent)
        agent.train(1)
        agent._online_finetuning = True

        data = agent._sample_train_batch(agent.batch_size)
        with torch.no_grad():
            bc_features = agent._bc_features(data.obs)
            per_sample = flow_log_prob(
                agent.policy.actor_bc_flow,
                bc_features,
                data.actions,
                num_steps=agent.flow_steps,
                method=agent.logp_method,
                num_probes=agent.logp_hutch_probes,
            ).to(device=agent.device, dtype=torch.float32)
        expected = per_sample.min().expand(data.actions.shape[0])

        result = agent._logp_beta_online(data)
        assert result.shape == (data.actions.shape[0],), fac_threshold
        assert torch.equal(result, expected), fac_threshold
        assert result.unique().numel() == 1, fac_threshold


def test_online_finetuning_runs_bc_update_every_step():
    agent = _make_agent(bc_pretrain_steps=1)
    _fill(agent)
    agent.train(1)  # completes the offline prelude (bc params now "trained")
    agent._online_finetuning = True

    bc_before = [p.clone() for p in agent.policy.actor_bc_flow.parameters()]
    agent.train(1)
    assert not all(
        torch.equal(a, b) for a, b in zip(bc_before, agent.policy.actor_bc_flow.parameters())
    ), "bc flow must keep training every online step, unlike offline Phase B"

    bc_before = [p.clone() for p in agent.policy.actor_bc_flow.parameters()]
    agent.train(1)
    assert not all(
        torch.equal(a, b) for a, b in zip(bc_before, agent.policy.actor_bc_flow.parameters())
    ), "bc flow must keep training on a second consecutive online step too"


# --- checkpointing ---


def test_checkpoint_round_trip_preserves_phase_and_logp_table():
    agent = _make_agent(bc_pretrain_steps=1)
    _fill(agent)
    agent.train(1)  # prelude + transition + first AC step, all in this one call
    assert agent._logp_table is not None
    assert agent.phase == "actor_critic"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent(bc_pretrain_steps=1)
        loaded.load(path, load_replay_buffer=False)

    assert loaded._phase_step == agent._phase_step
    assert loaded._bc_pretrain_steps_resolved == agent._bc_pretrain_steps_resolved
    assert loaded._logp_min == pytest.approx(agent._logp_min)
    assert loaded._logp_table is not None
    assert torch.allclose(
        loaded._logp_table.nan_to_num(), agent._logp_table.nan_to_num(), equal_nan=True
    )
    assert loaded.phase == "actor_critic"

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key

    # The loaded agent must continue in Phase B without recomputing the
    # cached table -- patch it to fail loudly if that regresses.
    def _must_not_recompute():
        raise AssertionError("_compute_dataset_logp should not run again after a resume")

    loaded._compute_dataset_logp = _must_not_recompute
    _fill(loaded)
    loaded.train(1)
    assert loaded.phase == "actor_critic"


# --- logging ---


class _DummyLogger:
    """Minimal duck-typed logger (only add_scalar, matching what
    _compute_dataset_logp calls) -- confirms self._global_step (set on
    BaseAlgorithm, base_algorithm.py:100) is readable at that call site
    without an AttributeError, without depending on the real Logger's
    TensorBoard/W&B backends."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.calls.append((tag, value, step))


def test_dataset_logp_logging_with_dummy_logger():
    logger = _DummyLogger()
    agent = _make_agent(bc_pretrain_steps=1, logger=logger)
    _fill(agent)
    agent.train(1)  # prelude + transition (builds the table and logs its stats once) + 1 AC step
    tags = [tag for tag, _, _ in logger.calls]
    assert "logp/dataset_mean" in tags
    assert "logp/dataset_min" in tags
    for _, value, step in logger.calls:
        assert np.isfinite(value)
        assert isinstance(step, int)


def test_bc_flow_loss_logged_during_prelude():
    logger = _DummyLogger()
    agent = _make_agent(bc_pretrain_steps=3, logger=logger, log_freq=1)
    _fill(agent)
    agent.train(1)  # prelude runs 3 BC steps, logging bc_flow_loss every log_freq=1 step
    tags = [tag for tag, _, _ in logger.calls]
    assert "train/bc_flow_loss" in tags
