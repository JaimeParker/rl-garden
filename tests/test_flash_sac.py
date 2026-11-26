"""Unit tests for FlashSAC components (CPU, no ManiSkill dependency)."""
from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from rl_garden.algorithms.flash_sac import (
    FlashSAC,
    _build_truncated_zeta_cdf,
    _compute_categorical_td_target,
    _sample_integer_from_cdf,
    _select_min_q_log_probs,
)
from rl_garden.common.distributions import safe_tanh_log_det_jacobian
from rl_garden.observations import ObservationContractError
from rl_garden.policies.flash_sac_policy import FlashSACPolicy

OBS_DIM = 20
ACT_DIM = 7
NUM_ENVS = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def obs_space():
    return spaces.Box(-1, 1, (OBS_DIM,), np.float32)


@pytest.fixture
def act_space():
    return spaces.Box(-1, 1, (ACT_DIM,), np.float32)


@pytest.fixture
def policy(obs_space, act_space):
    return FlashSACPolicy(
        observation_space=obs_space,
        action_space=act_space,
        actor_hidden_dim=32,
        actor_num_blocks=1,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        num_bins=11,
        min_v=-2.0,
        max_v=2.0,
    )


class _FakeEnv:
    def __init__(self, num_envs: int = NUM_ENVS):
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(-1, 1, (OBS_DIM,), np.float32)
        self.single_action_space = spaces.Box(-1, 1, (ACT_DIM,), np.float32)
        self.observation_space = spaces.Box(-1, 1, (num_envs, OBS_DIM), np.float32)
        self.action_space = spaces.Box(-1, 1, (num_envs, ACT_DIM), np.float32)

    def reset(self, seed=None):
        return torch.randn(self.num_envs, OBS_DIM), {}

    def step(self, actions):
        return (
            torch.randn(self.num_envs, OBS_DIM),
            torch.randn(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )


def _agent(num_envs: int = NUM_ENVS, **overrides) -> FlashSAC:
    kwargs = dict(
        env=_FakeEnv(num_envs),
        eval_env=None,
        buffer_size=400,
        buffer_device="cpu",
        learning_starts=0,
        batch_size=8,
        gamma=0.99,
        tau=0.005,
        training_freq=10,
        utd=1.0,
        n_step=3,
        actor_hidden_dim=32,
        actor_num_blocks=1,
        critic_hidden_dim=64,
        critic_num_blocks=1,
        num_bins=11,
        min_v=-2.0,
        max_v=2.0,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        seed=1,
        device="cpu",
        std_log=False,
        log_freq=0,
        eval_freq=0,
        save_final_checkpoint=False,
    )
    return FlashSAC(**{**kwargs, **overrides})


@pytest.fixture
def agent():
    return _agent()


def test_flash_sac_rejects_dict_observation_space_with_images():
    env = _FakeEnv()
    env.single_observation_space = spaces.Dict(
        {
            "state": spaces.Box(-1, 1, (OBS_DIM,), np.float32),
            "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
        }
    )
    with pytest.raises(ObservationContractError):
        FlashSAC(
            env=env,
            buffer_size=400,
            buffer_device="cpu",
            learning_starts=0,
            batch_size=8,
            actor_hidden_dim=32,
            actor_num_blocks=1,
            critic_hidden_dim=64,
            critic_num_blocks=1,
            num_bins=11,
            min_v=-2.0,
            max_v=2.0,
            device="cpu",
            std_log=False,
            log_freq=0,
            eval_freq=0,
            save_final_checkpoint=False,
        )


def _fill_buffer(agent: FlashSAC, n_transitions: int = 50) -> None:
    # agent.replay_buffer is Dict-based now (a bare Box env is
    # boundary-normalized to Dict({"state": Box}) by BaseAlgorithm.__init__).
    obs = torch.randn(NUM_ENVS, OBS_DIM)
    for i in range(n_transitions):
        next_obs = torch.randn(NUM_ENVS, OBS_DIM)
        actions = torch.randn(NUM_ENVS, ACT_DIM)
        rewards = torch.randn(NUM_ENVS)
        done = torch.zeros(NUM_ENVS, dtype=torch.bool)
        ep_end = torch.zeros(NUM_ENVS, dtype=torch.bool)
        if i % 10 == 9:
            ep_end[:] = True
        agent.replay_buffer.add(
            {"state": obs}, {"state": next_obs}, actions, rewards, done, ep_end
        )
        obs = next_obs


# ---------------------------------------------------------------------------
# safe_tanh_log_det_jacobian
# ---------------------------------------------------------------------------


def test_safe_tanh_log_det_jacobian_shape():
    x = torch.randn(8, ACT_DIM)
    j = safe_tanh_log_det_jacobian(x)
    assert j.shape == (8, ACT_DIM)


def test_safe_tanh_log_det_jacobian_matches_formula():
    x = torch.randn(4, 3)
    expected = torch.log(1.0 - torch.tanh(x) ** 2 + 1e-6)
    result = safe_tanh_log_det_jacobian(x)
    assert torch.allclose(result, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# FlashSACPolicy
# ---------------------------------------------------------------------------


def test_policy_predict_stochastic(policy):
    obs = torch.randn(8, OBS_DIM)
    actions = policy.predict(obs, deterministic=False)
    assert actions.shape == (8, ACT_DIM)
    assert actions.abs().max() <= 1.0 + 1e-5  # tanh bound


def test_policy_predict_deterministic(policy):
    obs = torch.randn(8, OBS_DIM)
    a1 = policy.predict(obs, deterministic=True)
    a2 = policy.predict(obs, deterministic=True)
    assert torch.allclose(a1, a2)


def test_policy_critic_output_shapes(policy):
    obs = torch.randn(8, OBS_DIM)
    actions = torch.randn(8, ACT_DIM)
    qs, infos = policy.critic(obs, actions, training=True)
    assert qs.shape == (2, 8)
    assert infos["log_prob"].shape == (2, 8, 11)


def test_policy_min_q_value_shape(policy):
    obs = torch.randn(8, OBS_DIM)
    actions = torch.randn(8, ACT_DIM)
    min_q = policy.min_q_value(obs, actions, target=False)
    assert min_q.shape == (8, 1)


def test_policy_normalize_weights_called(policy):
    called = []
    # Monkey-patch one layer's normalize_parameters to track calls
    orig = policy.actor.embedder.norm.normalize_parameters
    policy.actor.embedder.norm.normalize_parameters = lambda: (called.append(1), orig())[1]
    policy.normalize_weights()
    assert len(called) == 1, "normalize_parameters was not called"


# N-step replay buffer semantics (terminal-vs-truncation discount handling,
# ring-wrap rejection, etc.) are covered by NStepReplayBuffer's own tests
# in test_replay_buffer.py -- the flat-Box n-step buffer this file used to
# test here is gone (buffers are Dict-observation only, see
# ~/.claude/plans/observation-redesign.md).


# ---------------------------------------------------------------------------
# Categorical TD helpers
# ---------------------------------------------------------------------------


def test_categorical_td_target_sum_to_one():
    B, bins = 16, 11
    log_probs = torch.randn(B, bins).log_softmax(-1)
    reward = torch.randn(B)
    discounts = torch.ones(B) * 0.97
    actor_entropy = torch.zeros(B)
    probs = _compute_categorical_td_target(log_probs, reward, discounts, actor_entropy, bins, -2.0, 2.0)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)


def test_select_min_q_log_probs_picks_min():
    B, bins = 8, 11
    qs = torch.tensor([[1.0] * B, [2.0] * B])  # critic 0 always smaller
    log_probs = torch.stack([torch.ones(B, bins), torch.zeros(B, bins)])
    selected = _select_min_q_log_probs(qs, log_probs)
    assert selected.shape == (B, bins)
    assert torch.allclose(selected, torch.ones(B, bins))  # should pick critic 0


# ---------------------------------------------------------------------------
# Zeta noise
# ---------------------------------------------------------------------------


def test_zeta_cdf_ends_at_one():
    cdf = _build_truncated_zeta_cdf(2.0, 16)
    assert abs(float(cdf[-1]) - 1.0) < 1e-5


def test_zeta_samples_in_range():
    cdf = _build_truncated_zeta_cdf(2.0, 16)
    samples = [int(_sample_integer_from_cdf(cdf).item()) for _ in range(200)]
    assert min(samples) >= 1
    assert max(samples) <= 16


# ---------------------------------------------------------------------------
# FlashSAC.train()
# ---------------------------------------------------------------------------


def test_train_produces_finite_losses(agent):
    _fill_buffer(agent)
    info = agent.train(gradient_steps=2, compute_info=True)
    assert "critic/loss" in info
    assert torch.isfinite(torch.tensor(info["critic/loss"]))
    assert "actor/loss" in info
    assert torch.isfinite(torch.tensor(info["actor/loss"]))
    assert "temperature/value" in info
    assert info["temperature/value"] > 0


def test_temperature_gradient_direction():
    """alpha * (H - H_target) has the correct gradient sign.

    H > H_target → gradient positive → alpha decreases (policy too entropic).
    H < H_target → gradient negative → alpha increases (policy too deterministic).
    """

    target_entropy = -float(ACT_DIM)

    # H > H_target: gradient should be positive → descent decreases alpha
    log_temp = nn.Parameter(torch.tensor(0.0))
    loss = log_temp.exp() * (torch.tensor(0.5) - target_entropy)  # 0.5 > -7
    loss.backward()
    assert log_temp.grad.item() > 0, "Gradient should be positive when entropy > target"

    # H < H_target: gradient should be negative → descent increases alpha
    log_temp2 = nn.Parameter(torch.tensor(0.0))
    loss2 = log_temp2.exp() * (torch.tensor(-10.0) - target_entropy)  # -10 < -7
    loss2.backward()
    assert log_temp2.grad.item() < 0, "Gradient should be negative when entropy < target"


def test_normalize_weights_called_after_train(agent):
    _fill_buffer(agent)
    called = []
    orig = agent.policy.normalize_weights
    agent.policy.normalize_weights = lambda: (called.append(1), orig())
    agent.train(gradient_steps=1, compute_info=False)
    # Called at least once for critic, once for actor
    assert len(called) >= 2


def test_target_network_updated_after_train(agent):
    _fill_buffer(agent)
    # Copy target params before update
    before = [p.clone() for p in agent.policy.critic_target.parameters()]
    agent.train(gradient_steps=1, compute_info=False)
    after = list(agent.policy.critic_target.parameters())
    # Some params should change (polyak update moves them toward critic)
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed, "Target network was not updated by polyak"


def test_zeta_noise_repeats_for_same_env(agent):
    """Same environment should produce identical actions within a noise repeat window."""
    # _rollout_action receives whatever the (boundary-normalized) env
    # produces -- always Dict({"state": Box}) now.
    obs = {"state": torch.randn(NUM_ENVS, OBS_DIM)}

    # First call: force reinit by setting count=0
    agent._zeta_count[:] = 0
    actions1, _, _ = agent._rollout_action(obs, learning_has_started=True)
    assert agent._zeta_count.min().item() == 1

    # Force _zeta_n to be large so the second call definitely does NOT reinit
    agent._zeta_n[:] = 100
    actions2, _, _ = agent._rollout_action(obs, learning_has_started=True)
    assert torch.allclose(actions1, actions2, atol=1e-5), "Noise was not repeated"


def test_flash_sac_args_carry_every_field_the_shared_env_request_reads():
    from rl_garden.common.cli_args import resolve_eval_record_dir
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online.entrypoints.flash_sac import FlashSACArgs

    args = FlashSACArgs()
    assert resolve_eval_record_dir(args, "run").endswith("run/eval_videos")
    assert make_env_request(args, "run").eval_record_dir is not None
    args.eval_output_dir = "/tmp/elsewhere"
    assert resolve_eval_record_dir(args, "run") == "/tmp/elsewhere"


# ---------------------------------------------------------------------------
# Reward-normalizer state in checkpoints (the official agent saves and loads
# G_r, G_r_max and the mean/var/count of the return statistics)
# ---------------------------------------------------------------------------

_STATE_KEYS = {"G_r", "G_r_max", "G_rms_mean", "G_rms_var", "G_rms_count"}
_learner = functools.partial(_agent, normalize_reward=True, learning_starts=10_000)


def _trained(**kwargs) -> FlashSAC:
    agent = _learner(**kwargs)
    agent.learn(total_timesteps=40)
    return agent


def _normalizer_warnings(record) -> list[str]:
    return [str(w.message) for w in record if "normalizer" in str(w.message)]


def test_checkpoint_restores_the_reward_normalizer(tmp_path):
    agent = _trained()
    saved = {k: v.clone() for k, v in agent.reward_normalizer.state_dict().items()}
    assert set(saved) == _STATE_KEYS and saved["G_r"].shape == (NUM_ENVS,)
    agent.save(tmp_path / "c.pt")
    resumed = _learner()
    resumed.load(tmp_path / "c.pt")
    for key, value in resumed.reward_normalizer.state_dict().items():
        assert torch.equal(value, saved[key]), key
    rewards = torch.linspace(-3.0, 3.0, 8)
    expected = agent.reward_normalizer.normalize_rewards(rewards)
    assert torch.equal(resumed.reward_normalizer.normalize_rewards(rewards), expected)
    assert not torch.equal(_learner().reward_normalizer.normalize_rewards(rewards), expected)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        resumed.learn(total_timesteps=60)
    assert _normalizer_warnings(record) == []
    assert resumed.reward_normalizer.G_rms.count == saved["G_rms_count"] + 20


def test_checkpoint_without_a_normalizer_state_warns_when_training_starts(tmp_path):
    agent = _trained()
    agent.save(tmp_path / "c.pt")
    checkpoint = torch.load(tmp_path / "c.pt", weights_only=False)
    checkpoint["state"]["extra"].pop("reward_normalizer")
    torch.save(checkpoint, tmp_path / "old.pt")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        resumed = _learner()
        resumed.load(tmp_path / "old.pt")
        assert _normalizer_warnings(record) == []
        assert resumed.reward_normalizer.G_rms.count == 0
        resumed.learn(total_timesteps=60)
        resumed.learn(total_timesteps=80)
    assert len(_normalizer_warnings(record)) == 1


@pytest.mark.parametrize("saved_envs, loaded_envs", [(2, 3), (3, 1), (1, 3)])
def test_other_number_of_envs_keeps_the_statistics_and_restarts_the_running_returns(
    saved_envs, loaded_envs, tmp_path
):
    agent = _trained(num_envs=saved_envs)
    agent.save(tmp_path / "c.pt")
    resumed = _learner(num_envs=loaded_envs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resumed.load(tmp_path / "c.pt")  # evaluation stops here
    step = agent._global_step
    with pytest.warns(UserWarning, match=f"running returns of {saved_envs} envs, this run has {loaded_envs}"):
        resumed._on_training_start(step)
    assert torch.equal(resumed.reward_normalizer.G_r, torch.zeros(1))
    for key in _STATE_KEYS - {"G_r"}:
        assert torch.equal(resumed.reward_normalizer.state_dict()[key], agent.reward_normalizer.state_dict()[key])
    resumed.learn(total_timesteps=step + 6)
    assert resumed.reward_normalizer.G_r.shape == (loaded_envs,)


@pytest.mark.parametrize("saved_with", [False, True])
def test_checkpoints_load_with_the_other_normalize_reward_setting(saved_with, tmp_path):
    agent = _agent(normalize_reward=saved_with, learning_starts=10_000)
    agent.learn(total_timesteps=20)
    agent.save(tmp_path / "c.pt")
    extra = torch.load(tmp_path / "c.pt", weights_only=False)["state"]["extra"]
    assert ("reward_normalizer" in extra) is saved_with
    _agent(normalize_reward=not saved_with).load(tmp_path / "c.pt")


@pytest.mark.parametrize("loaded_envs", [3, 1])
def test_all_zero_running_returns_of_another_number_of_envs_restart_too(loaded_envs, tmp_path):
    agent = _trained(num_envs=2)
    agent.reward_normalizer.G_r = torch.zeros(2)  # e.g. every env ended on the last step
    agent.save(tmp_path / "c.pt")
    resumed = _learner(num_envs=loaded_envs, learning_starts=0)
    resumed.load(tmp_path / "c.pt", load_replay_buffer=False)
    resumed.learn(total_timesteps=agent._global_step + 2 * loaded_envs)
    assert resumed.reward_normalizer.G_r.shape == (loaded_envs,)
