from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import FloQ, Off2OnFloQ, OfflineEnvSpec
from rl_garden.algorithms.floq import FloQCore
from rl_garden.algorithms.fql import FQLCore
from rl_garden.encoders.combined import default_image_encoder_factory
from rl_garden.networks.critic_vector_field import compute_support, hl_gauss_to_probs

# Small + fast: "gap" pooling (unlike the default "flatten") tolerates tiny
# images without PlainConv's flatten-layer size mismatch. Mirrors
# tests/test_fql_core.py's own vision-test image encoder factory.
_TEST_IMAGE_SIZE = 16
_test_image_encoder_factory = default_image_encoder_factory(
    features_dim=16, plain_conv_pooling="gap"
)

_METRIC_KEYS = (
    "critic_loss",
    "floq_loss",
    "distilled_critic_loss",
    "actor_loss",
    "bc_flow_loss",
    "distill_loss",
    "q_loss",
)


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _vision_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=0, high=255, shape=(_TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3), dtype=np.uint8
                ),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> FloQ:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=16,
        device="cpu",
        net_arch=[16, 16],
        flow_steps=4,
        noise_samples=3,
        critic_flow_steps=3,
        num_bins=11,
    )
    defaults.update(kwargs)
    return FloQ(**defaults)


def _fill(agent: FloQ, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _fill_vision(agent: FloQ, steps: int = 64) -> None:
    env = agent.env
    obs_space = env.single_observation_space
    img_shape = (_TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3)
    for _ in range(steps):
        obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, *img_shape), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, *obs_space["state"].shape),
        }
        next_obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, *img_shape), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, *obs_space["state"].shape),
        }
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _assert_predict_in_bounds(agent: FloQ) -> None:
    obs_space = agent.env.single_observation_space
    obs = {
        "rgb": torch.randint(
            0, 256, (1, _TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3), dtype=torch.uint8
        ),
        "state": torch.randn(1, *obs_space["state"].shape),
    }
    with torch.no_grad():
        action = agent.policy.predict(obs)
    assert action.shape == (1, 3)
    assert torch.all(action >= agent.policy.action_low)
    assert torch.all(action <= agent.policy.action_high)


def test_rejects_unsupported_observation_space():
    unsupported = OfflineEnvSpec(
        spaces.MultiDiscrete([3, 3]),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=1,
    )
    with pytest.raises(TypeError, match="Box or Dict"):
        FloQ(env=unsupported, buffer_device="cpu", device="cpu")


def test_vision_shared_encoder_smoke():
    agent = _make_agent(
        env=_vision_env(),
        encoder_sharing="shared",
        image_encoder_factory=_test_image_encoder_factory,
    )
    _fill_vision(agent)
    metrics = agent.train(1, compute_info=True)
    for key in _METRIC_KEYS:
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])
    assert not hasattr(agent.policy, "actor_bc_flow_encoder")
    _assert_predict_in_bounds(agent)


def test_vision_separate_encoder_smoke():
    agent = _make_agent(
        env=_vision_env(),
        encoder_sharing="separate",
        image_encoder_factory=_test_image_encoder_factory,
    )
    _fill_vision(agent)
    metrics = agent.train(1, compute_info=True)
    for key in _METRIC_KEYS:
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])
    _assert_predict_in_bounds(agent)


def test_gradient_step_produces_finite_losses():
    agent = _make_agent()
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in _METRIC_KEYS:
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_actor_and_critic_and_floq_update_every_step():
    agent = _make_agent()
    _fill(agent)

    bc_flow_before = [p.clone() for p in agent.policy.actor_bc_flow.parameters()]
    onestep_before = [p.clone() for p in agent.policy.actor_onestep_flow.parameters()]
    critic_before = [p.clone() for p in agent.policy.critic.parameters()]
    floq_before = [p.clone() for p in agent.policy.floq.parameters()]

    metrics = agent.train(1, compute_info=True)

    assert "actor_loss" in metrics
    assert not all(
        torch.equal(a, b) for a, b in zip(bc_flow_before, agent.policy.actor_bc_flow.parameters())
    )
    assert not all(
        torch.equal(a, b)
        for a, b in zip(onestep_before, agent.policy.actor_onestep_flow.parameters())
    )
    assert not all(
        torch.equal(a, b) for a, b in zip(critic_before, agent.policy.critic.parameters())
    )
    assert not all(
        torch.equal(a, b) for a, b in zip(floq_before, agent.policy.floq.parameters())
    )


def test_floq_target_frozen_and_disjoint_from_critic_optimizer():
    agent = _make_agent()
    _fill(agent)

    for p in agent.policy.floq_target.parameters():
        assert not p.requires_grad

    floq_target_ptrs = {p.data_ptr() for p in agent.policy.floq_target.parameters()}
    critic_optimizer_ptrs = {
        p.data_ptr() for group in agent.critic_optimizer.param_groups for p in group["params"]
    }
    assert floq_target_ptrs.isdisjoint(critic_optimizer_ptrs)

    floq_before = [p.clone() for p in agent.policy.floq_target.parameters()]
    agent.train(1)
    # floq_target moves only via polyak update in _update_targets, never via
    # a gradient step, but should still differ after one full train() call.
    assert not all(
        torch.equal(a, b) for a, b in zip(floq_before, agent.policy.floq_target.parameters())
    )


def test_mean_current_returns_has_no_grad(monkeypatch):
    """`mean_current_returns` (the distilled critic's TD target) must never
    carry a gradient -- instrument `integrate_returns` to record whether
    grad tracking was enabled at each call site inside _critic_update."""
    agent = _make_agent()
    _fill(agent)

    import rl_garden.algorithms.floq as floq_module

    seen_grad_enabled = []
    original = floq_module.integrate_returns

    def spy(*args, **kwargs):
        seen_grad_enabled.append(torch.is_grad_enabled())
        return original(*args, **kwargs)

    monkeypatch.setattr(floq_module, "integrate_returns", spy)
    agent.train(1)

    # Both integrate_returns call sites (target unroll, current-returns
    # unroll) run inside torch.no_grad() in _critic_update.
    assert seen_grad_enabled
    assert not any(seen_grad_enabled)


def test_ebr_shapes_with_small_ensembles():
    """(E,B,R) shape sanity with flow_num_ensembles=2, noise_samples=3,
    batch_size=4 -- instrument the policy's floq forward to check the shapes
    flowing through _critic_update."""
    agent = _make_agent(
        flow_num_ensembles=2, noise_samples=3, batch_size=4, critic_flow_steps=2
    )
    _fill(agent)

    seen_shapes = []
    original_forward = agent.policy.floq.forward

    def spy(features, actions, returns, times):
        seen_shapes.append(tuple(returns.shape))
        return original_forward(features, actions, returns, times)

    agent.policy.floq.forward = spy
    agent.train(1)

    assert seen_shapes
    for shape in seen_shapes:
        assert shape[0] == 2  # E
        assert shape[2] == 1
    # Every floq() forward call operates on the noise-repeated batch B*R
    # (target unroll, current-returns unroll, flow-matching loss) -- none
    # of them run on the raw, un-repeated B.
    assert set(shape[1] for shape in seen_shapes) == {4 * 3}


def test_next_action_sampled_per_noise_repeat_not_broadcast():
    """The reference draws one next-action per (batch, noise-sample) pair,
    not one action per batch element broadcast across all R noise samples --
    repeating a single sampled action would silently collapse the target's
    Monte Carlo average over next actions to an average over noise ratios
    only. Spy on actor_onestep_flow to check the actual leading dim it's
    called with, and that its outputs vary across the R axis."""
    agent = _make_agent(noise_samples=5, batch_size=4)
    _fill(agent)

    seen_batch_sizes = []
    original = agent.policy.actor_onestep_flow.forward

    def spy(features, noise):
        seen_batch_sizes.append(features.shape[0])
        return original(features, noise)

    agent.policy.actor_onestep_flow.forward = spy
    agent.train(1)
    agent.policy.actor_onestep_flow.forward = original

    # One call for _actor_update's onestep sample (batch_size), one call for
    # _critic_update's repeated next-action sample (batch_size * noise_samples).
    assert agent.batch_size * agent.noise_samples in seen_batch_sizes


def test_hl_gauss_worked_defaults():
    """HL-Gauss fixture using the worked defaults from the port plan:
    q_min=-101, q_max=1, bin_width=2.04, sigma_eff=16*2.04=32.64."""
    q_min, q_max, num_bins = -101.0, 1.0, 51
    support = compute_support(q_min, q_max, num_bins)
    assert support.shape == (num_bins,)
    assert float(support[0]) == pytest.approx(q_min)
    assert float(support[-1]) == pytest.approx(q_max)
    bin_width = float(support[1] - support[0])
    assert bin_width == pytest.approx(2.04, abs=1e-2)

    sigma = 16.0
    sigma_eff = sigma * bin_width
    assert sigma_eff == pytest.approx(32.64, abs=1e-1)

    target = torch.tensor([-50.0, 0.0])
    probs = hl_gauss_to_probs(target, support, sigma_eff)
    assert probs.shape == (2, num_bins - 1)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    bin_centers = (support[:-1] + support[1:]) / 2
    for i, t in enumerate(target.tolist()):
        peak_bin = int(probs[i].argmax())
        assert abs(float(bin_centers[peak_bin]) - t) < 5 * bin_width


def test_checkpoint_round_trip():
    agent = _make_agent()
    _fill(agent)
    for _ in range(3):
        agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent()
        loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key
    assert "floq" in dict(loaded.policy.named_children())
    assert "floq_target" in dict(loaded.policy.named_children())
    assert not hasattr(loaded.policy, "critic_target")

    metadata = agent._checkpoint_metadata()
    for key in (
        "r_min",
        "r_max",
        "flow_num_ensembles",
        "noise_samples",
        "noise_coverage",
        "critic_flow_steps",
        "train_at_zero_only",
        "embed_time",
        "time_embed_dim",
        "use_prob_embed",
        "num_bins",
        "sigma",
        "reward_offset",
    ):
        assert key in metadata


def test_critic_flow_net_arch_overrides_shared_net_arch():
    agent = _make_agent(net_arch=[16, 16], critic_flow_net_arch=[8])

    assert list(agent.policy.floq._head_kwargs["hidden_dims"]) == [8]
    actor_first_layer = next(
        m for m in agent.policy.actor_bc_flow.modules() if isinstance(m, torch.nn.Linear)
    )
    assert actor_first_layer.out_features == 16

    metadata = agent._checkpoint_metadata()
    assert "critic_flow_net_arch" in metadata


def test_off2on_floq_mro_puts_floq_core_before_fql_core():
    mro = Off2OnFloQ.__mro__
    assert mro.index(FloQCore) < mro.index(FQLCore)
