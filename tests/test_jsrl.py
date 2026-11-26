from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import AWAC, IQL, JSRL, WSRL, CalQL, OfflineEnvSpec
from rl_garden.algorithms.jsrl import _load_guide_policy
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObsGroups

_OBS_DIM = 6
_ACTION_DIM = 2


def _offline_env_spec() -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(-1.0, 1.0, (_OBS_DIM,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32),
        num_envs=1,
    )


def _make_iql_checkpoint(tmp_path, **overrides) -> str:
    kwargs = {"env": _offline_env_spec(), "device": "cpu", "net_arch": [8]}
    kwargs.update(overrides)
    agent = IQL(**kwargs)
    path = str(tmp_path / "iql.pt")
    agent.save(path)
    return path, agent


def _make_awac_checkpoint(tmp_path, **overrides) -> str:
    kwargs = {"env": _offline_env_spec(), "device": "cpu", "net_arch": [8]}
    kwargs.update(overrides)
    agent = AWAC(**kwargs)
    agent.policy.fit_obs_normalizer(torch.randn(100, _OBS_DIM) * 3 + 1)
    path = str(tmp_path / "awac.pt")
    agent.save(path)
    return path, agent


def _make_calql_checkpoint(tmp_path, **overrides) -> str:
    kwargs = {"env": _offline_env_spec(), "device": "cpu", "net_arch": {"pi": [8], "qf": [8]}}
    kwargs.update(overrides)
    agent = CalQL(**kwargs)
    path = str(tmp_path / "calql.pt")
    agent.save(path)
    return path, agent


def _make_wsrl_checkpoint(tmp_path, **overrides) -> str:
    kwargs = {
        "env": _ScriptedVecEnv([1_000]), "device": "cpu", "buffer_device": "cpu",
        "buffer_size": 16, "net_arch": {"pi": [8], "qf": [8]},
    }
    kwargs.update(overrides)
    agent = WSRL(**kwargs)
    path = str(tmp_path / "wsrl.pt")
    agent.save(path)
    return path, agent


class _ScriptedVecEnv(gym.Env):
    """Deterministic vector env: env i terminates every ``episode_len[i]`` steps,
    reporting ``final_info``/``_final_info`` with a deterministic ``return``."""

    def __init__(self, episode_len: list[int]) -> None:
        self.num_envs = len(episode_len)
        self.episode_len = list(episode_len)
        self.single_observation_space = spaces.Box(-1.0, 1.0, (_OBS_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.broadcast_to(self.single_observation_space.low, (self.num_envs, _OBS_DIM)),
            high=np.broadcast_to(self.single_observation_space.high, (self.num_envs, _OBS_DIM)),
            dtype=np.float32,
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (self.num_envs, _ACTION_DIM)),
            high=np.broadcast_to(self.single_action_space.high, (self.num_envs, _ACTION_DIM)),
            dtype=np.float32,
        )
        self._t = [0] * self.num_envs

    def reset(self, seed=None):
        del seed
        self._t = [0] * self.num_envs
        return torch.zeros(self.num_envs, _OBS_DIM), {}

    def step(self, actions):
        obs = torch.zeros(self.num_envs, _OBS_DIM)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        returns = torch.zeros(self.num_envs)
        done_mask = torch.zeros(self.num_envs, dtype=torch.bool)
        for i in range(self.num_envs):
            self._t[i] += 1
            if self._t[i] >= self.episode_len[i]:
                terminations[i] = True
                returns[i] = float(self._t[i])
                done_mask[i] = True
                self._t[i] = 0
        infos: dict = {}
        if done_mask.any():
            infos["final_info"] = {"episode": {"return": returns}}
            infos["_final_info"] = done_mask
        return obs, rewards, terminations, truncations, infos

    def close(self) -> None:
        return None


def _make_jsrl_agent(guide_checkpoint, guide_algorithm="iql", *, max_horizon=2, num_envs=2, episode_len=None, **overrides) -> JSRL:
    env = _ScriptedVecEnv(episode_len=episode_len or ([1_000] * num_envs))
    kwargs = {
        "env": env,
        "guide_checkpoint": guide_checkpoint,
        "guide_algorithm": guide_algorithm,
        "max_horizon": max_horizon,
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": 8,
        "learning_starts": 0,
        "training_freq": 4,
        "eval_freq": 0,
        "log_freq": 0,
        "net_arch": [8],
    }
    kwargs.update(overrides)
    return JSRL(**kwargs)


# --- guide policy loading ---


def test_load_guide_policy_iql_matches_source_policy(tmp_path):
    ckpt, agent = _make_iql_checkpoint(tmp_path, std_parameterization="uniform")
    guide = _load_guide_policy(
        ckpt, "iql", agent.env.single_observation_space, agent.env.single_action_space,
        device=torch.device("cpu"), std_parameterization="uniform",
    )
    obs = {"state": torch.randn(4, _OBS_DIM)}
    assert torch.allclose(
        guide.predict(obs, deterministic=True), agent.policy.predict(obs, deterministic=True)
    )
    for p in guide.parameters():
        assert not p.requires_grad


def _edit_checkpoint(path: str, edit) -> None:
    contents = torch.load(path)
    edit(contents)
    torch.save(contents, path)


def _forget_actor_options(contents) -> None:
    """What checkpoints that recorded no actor option beyond these look like."""
    hyperparameters = contents["metadata"]["hyperparameters"]
    for name in (
        "actor_use_layer_norm", "actor_use_group_norm", "num_groups", "actor_dropout_rate",
        "kernel_init", "backbone_type", "std_parameterization",
        "policy_log_std_multiplier", "policy_log_std_offset",
    ):
        hyperparameters.pop(name, None)


def _assert_guide_matches(checkpoint, algorithm, agent, **kwargs) -> None:
    guide = _load_guide_policy(
        checkpoint, algorithm, agent.env.single_observation_space, agent.env.single_action_space,
        device=torch.device("cpu"), **kwargs,
    )
    obs = {
        key: torch.randn(4, *space.shape)
        for key, space in agent.env.single_observation_space.spaces.items()
    }
    agent.policy.eval()
    assert torch.allclose(
        guide.predict(obs, deterministic=True), agent.policy.predict(obs, deterministic=True)
    )
    torch.manual_seed(0)
    sampled = guide.predict(obs, deterministic=False)
    torch.manual_seed(0)
    assert torch.allclose(sampled, agent.policy.predict(obs, deterministic=False))


# Differing critic options must not matter to the guide; the actor's all do.
_ACTOR_OPTIONS = {
    "layer_norm": {"actor_use_layer_norm": True, "critic_use_layer_norm": True},
    "group_norm": {
        "actor_use_layer_norm": False, "actor_use_group_norm": True, "num_groups": 2},
    "dropout": {"actor_dropout_rate": 0.5, "net_arch": [8, 8]},
    "layer_norm_dropout": {
        "actor_use_layer_norm": True, "actor_dropout_rate": 0.5, "net_arch": [8, 8]},
    "resnet": {"backbone_type": "mlp_resnet", "net_arch": [8, 8]},
    "resnet_layer_norm": {
        "backbone_type": "mlp_resnet", "actor_use_layer_norm": True, "net_arch": [8, 8]},
    "uniform_std": {"std_parameterization": "uniform"},
    "kernel_init": {"kernel_init": "orthogonal_near_zero_output"},
    "separate_encoders": {"encoder_sharing": "separate"},
}
# What a guide checkpoint without recorded actor options is still rebuilt with.
_WEIGHTS_DECIDE = {
    "actor_use_layer_norm", "critic_use_layer_norm", "std_parameterization", "kernel_init",
    "encoder_sharing",
}
_MAKE_CHECKPOINT = {
    "iql": _make_iql_checkpoint,
    "awac": _make_awac_checkpoint,
    "calql": _make_calql_checkpoint,
    "wsrl": _make_wsrl_checkpoint,
}


@pytest.mark.parametrize("legacy", [False, True], ids=["recorded", "legacy"])
@pytest.mark.parametrize("options", _ACTOR_OPTIONS.values(), ids=_ACTOR_OPTIONS.keys())
@pytest.mark.parametrize("algorithm", ["iql", "awac", "calql", "wsrl"])
def test_guide_is_rebuilt_with_the_options_it_was_trained_with(
    algorithm, options, legacy, tmp_path
):
    if legacy and not options.keys() <= _WEIGHTS_DECIDE:
        pytest.skip("a legacy checkpoint does not determine these options")
    checkpoint, agent = _MAKE_CHECKPOINT[algorithm](tmp_path, **options)
    if legacy:
        _edit_checkpoint(checkpoint, _forget_actor_options)
    _assert_guide_matches(checkpoint, algorithm, agent)


@pytest.mark.parametrize(
    "algorithm,options",
    [
        ("iql", {"backbone_type": "mlp_resnet", "net_arch": [8, 8]}),
        ("awac", {"actor_dropout_rate": 0.5, "net_arch": [8, 8]}),
        ("calql", {"policy_log_std_multiplier": 1.0, "policy_log_std_offset": -1.0}),
    ],
)
def test_legacy_guide_with_undetermined_options_is_refused(algorithm, options, tmp_path):
    checkpoint, agent = _MAKE_CHECKPOINT[algorithm](tmp_path, **options)
    _edit_checkpoint(checkpoint, _forget_actor_options)
    with pytest.raises(RuntimeError, match=r"actor\."):
        _assert_guide_matches(checkpoint, algorithm, agent)


@pytest.mark.parametrize("algorithm", ["calql", "wsrl"])
def test_guide_keeps_the_learned_log_std_scale(algorithm, tmp_path):
    checkpoint, agent = _MAKE_CHECKPOINT[algorithm](
        tmp_path, policy_log_std_multiplier=1.0, policy_log_std_offset=-1.0
    )
    _assert_guide_matches(checkpoint, algorithm, agent)


def test_legacy_guide_with_a_norm_of_unrecorded_kind_is_taken_as_layer_norm(tmp_path, caplog):
    checkpoint, agent = _make_iql_checkpoint(tmp_path, actor_use_layer_norm=True)
    _edit_checkpoint(checkpoint, _forget_actor_options)
    _assert_guide_matches(checkpoint, "iql", agent)
    assert "assuming LayerNorm" in caplog.text


def test_guide_keeps_its_observation_normalizer_and_groups(tmp_path):
    spec = OfflineEnvSpec(
        spaces.Dict({
            "state": spaces.Box(-1.0, 1.0, (_OBS_DIM,), dtype=np.float32),
            "state_priv": spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
        }),
        spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32),
        num_envs=1,
    )
    checkpoint, agent = _make_iql_checkpoint(
        tmp_path,
        env=spec,
        obs_groups=ObsGroups(actor=("state",), critic=("state", "state_priv")),
        encoder_config=EncoderConfig(normalize_obs=True),
    )
    agent.policy.train()
    agent.policy.update_normalizer(
        {"state": torch.randn(50, _OBS_DIM) * 3 + 1, "state_priv": torch.randn(50, 3)}
    )
    agent.save(checkpoint)
    _assert_guide_matches(checkpoint, "iql", agent)


@pytest.mark.parametrize(
    "recorded,match",
    [
        ({"std_parameterization": "exp"}, "fc_logstd.*log_stds"),
        ({"actor_use_layer_norm": True}, "actor.trunk"),
    ],
)
def test_guide_mismatch_names_the_keys(recorded, match, tmp_path):
    checkpoint, agent = _make_iql_checkpoint(
        tmp_path, std_parameterization="uniform", net_arch=[8, 8]
    )
    _edit_checkpoint(
        checkpoint, lambda c: c["metadata"]["hyperparameters"].update(recorded)
    )
    with pytest.raises(RuntimeError, match=match):
        _load_guide_policy(
            checkpoint, "iql", agent.env.single_observation_space,
            agent.env.single_action_space, device=torch.device("cpu"),
        )


def test_load_guide_policy_awac_preserves_obs_normalizer(tmp_path):
    ckpt, agent = _make_awac_checkpoint(tmp_path)
    guide = _load_guide_policy(
        ckpt, "awac", agent.env.single_observation_space, agent.env.single_action_space,
        device=torch.device("cpu"),
    )
    assert not torch.allclose(guide.obs_mean, torch.zeros(_OBS_DIM))
    obs = {"state": torch.randn(4, _OBS_DIM)}
    assert torch.allclose(
        guide.predict(obs, deterministic=True), agent.policy.predict(obs, deterministic=True)
    )


def test_load_guide_policy_calql_matches_source_policy(tmp_path):
    ckpt, agent = _make_calql_checkpoint(tmp_path)
    guide = _load_guide_policy(
        ckpt, "calql", agent.env.single_observation_space, agent.env.single_action_space,
        device=torch.device("cpu"),
    )
    obs = {"state": torch.randn(4, _OBS_DIM)}
    assert torch.allclose(
        guide.predict(obs, deterministic=True), agent.policy.predict(obs, deterministic=True)
    )


# --- guide/trainee blending ---


def test_guide_mask_inclusive_boundary(tmp_path):
    ckpt, _ = _make_iql_checkpoint(tmp_path)
    agent = _make_jsrl_agent(ckpt, max_horizon=2, n_curriculum_stages=2)
    assert agent.horizon == 2
    episode_step = torch.tensor([0, 1, 2, 3])
    # timesteps <= horizon (inclusive): step 2 still uses the guide, step 3 doesn't.
    assert agent._guide_mask(episode_step).tolist() == [True, True, True, False]


# --- episode-step tracking under heterogeneous termination ---


def test_episode_step_counter_resets_per_env_on_termination(tmp_path):
    ckpt, _ = _make_iql_checkpoint(tmp_path)
    agent = _make_jsrl_agent(ckpt, max_horizon=1, episode_len=[2, 5])
    obs, _ = agent.env.reset()
    agent._on_env_reset(obs)
    assert agent._episode_step.tolist() == [0, 0]
    # env 0 terminates every 2 steps, env 1 every 5 -- hand-traced sequence
    # of per-env counters, reset independently on each env's own termination
    # (SAME_STEP autoreset: reset happens on the step just taken, from
    # terminations|truncations, not upstream's buf_dones convention).
    expected = [[1, 1], [0, 2], [1, 3], [0, 4], [1, 0], [0, 1]]
    for expected_after in expected:
        _, _, terminations, truncations, _ = agent.env.step(torch.zeros(2, _ACTION_DIM))
        agent._post_rollout_step(None, terminations, truncations, {})
        assert agent._episode_step.tolist() == expected_after


# --- curriculum ---


def test_curriculum_holds_until_best_established_then_advances(tmp_path):
    # Upstream's own update rule (JSRLAfterEvalCallback._on_step) is a
    # three-way *exclusive* if/elif/elif: a call either (a) returns early
    # (ring buffer not yet full), (b) initializes best_moving_mean_reward
    # (the first non-early-return call), or (c) checks whether to advance
    # the horizon -- never (b) and (c) on the same call.
    ckpt, _ = _make_iql_checkpoint(tmp_path)
    agent = _make_jsrl_agent(ckpt, max_horizon=4, n_curriculum_stages=4, window_size=2, tolerance=0.0)
    assert agent._horizons == [4, 3, 2, 1, 0]
    assert agent.horizon == 4

    agent._update_jsrl_curriculum(1.0)  # buffer not full yet -> early return
    assert agent._horizon_step == 0
    assert agent._best_moving_mean_reward == float("-inf")

    agent._update_jsrl_curriculum(1.0)  # buffer full: initializes best/tolerated to 1.0
    assert agent._horizon_step == 0
    assert agent._best_moving_mean_reward == 1.0
    assert agent._tolerated_moving_mean_reward == 1.0  # tolerance=0.0

    agent._update_jsrl_curriculum(1.0)  # moving_mean(1.0) >= tolerated(1.0) -> advance
    assert agent._horizon_step == 1
    assert agent.horizon == 3

    agent._update_jsrl_curriculum(0.0)  # moving_mean(0.5) < tolerated(1.0) -> holds
    assert agent._horizon_step == 1
    assert agent.horizon == 3


def test_curriculum_stops_advancing_once_horizon_reaches_zero(tmp_path):
    ckpt, _ = _make_iql_checkpoint(tmp_path)
    agent = _make_jsrl_agent(ckpt, max_horizon=1, n_curriculum_stages=1, window_size=1, tolerance=0.0)
    assert agent._horizons == [1, 0]

    agent._update_jsrl_curriculum(1.0)  # initializes best/tolerated=1.0, no advance
    assert agent._horizon_step == 0

    agent._update_jsrl_curriculum(5.0)  # moving_mean(5.0) >= tolerated(1.0) -> advance to horizon=0
    assert agent._horizon_step == 1
    assert agent.horizon == 0
    assert agent._best_moving_mean_reward == 5.0

    # horizon <= 0 now: moving_mean_reward still updates (unconditional),
    # but best/tolerated/horizon_step are frozen -- matches upstream's own
    # early-return ordering (roll + mean computed before the gate check).
    agent._update_jsrl_curriculum(100.0)
    assert agent._horizon_step == 1
    assert agent._best_moving_mean_reward == 5.0
    assert agent._moving_mean_reward == 100.0


def test_evaluate_reports_jsrl_curriculum_metrics(tmp_path):
    ckpt, _ = _make_iql_checkpoint(tmp_path)
    agent = _make_jsrl_agent(
        ckpt, max_horizon=2, n_curriculum_stages=2, num_eval_steps=10,
        eval_env=_ScriptedVecEnv(episode_len=[2, 2]),
    )
    metrics = agent._evaluate()
    assert "jsrl_horizon" in metrics
    assert "jsrl_moving_mean_reward" in metrics
    assert "return" in metrics


# --- checkpointing ---


def test_checkpoint_roundtrip_preserves_curriculum_state(tmp_path):
    ckpt, _ = _make_iql_checkpoint(tmp_path)
    agent = _make_jsrl_agent(ckpt, max_horizon=4, n_curriculum_stages=4, window_size=2)
    agent._update_jsrl_curriculum(1.0)  # buffer not full -> early return
    agent._update_jsrl_curriculum(1.0)  # buffer full -> initializes best/tolerated
    agent._update_jsrl_curriculum(1.0)  # moving_mean >= tolerated -> advances
    assert agent._horizon_step == 1

    save_path = tmp_path / "jsrl.pt"
    agent.save(save_path)

    resumed = _make_jsrl_agent(ckpt, max_horizon=4, n_curriculum_stages=4, window_size=2)
    resumed.load(save_path, load_replay_buffer=False)
    assert resumed._horizon_step == 1
    assert resumed._best_moving_mean_reward == agent._best_moving_mean_reward
    assert resumed._tolerated_moving_mean_reward == agent._tolerated_moving_mean_reward
    assert torch.equal(resumed._mean_rewards, agent._mean_rewards)

    metadata = agent._checkpoint_metadata()
    assert metadata["guide_checkpoint"] == ckpt
    assert metadata["guide_algorithm"] == "iql"


def test_checkpoint_embeds_the_guide(tmp_path):
    ckpt, source = _make_iql_checkpoint(tmp_path, actor_use_layer_norm=True, net_arch=[8, 8])
    agent = _make_jsrl_agent(ckpt, max_horizon=4)
    agent.save(tmp_path / "jsrl.pt")
    extra = torch.load(tmp_path / "jsrl.pt")["state"]["extra"]
    assert not any(key.startswith(("critic", "value")) for key in extra["guide"]["policy"])

    restored = _make_jsrl_agent(None, max_horizon=4)
    obs = {"state": torch.randn(2, _OBS_DIM)}
    with pytest.raises(RuntimeError, match="no guide policy"):
        restored._guide_action(obs, deterministic=True)
    restored.load(tmp_path / "jsrl.pt", load_replay_buffer=False)
    assert restored.guide_checkpoint == ckpt
    assert torch.allclose(
        restored.guide_policy.predict(obs, deterministic=True),
        source.policy.predict(obs, deterministic=True),
    )
    assert not any(p.requires_grad for p in restored.guide_policy.parameters())

    restored.save(tmp_path / "again.pt")
    again = torch.load(tmp_path / "again.pt")["state"]["extra"]["guide"]
    assert again["source"] == ckpt


def test_missing_guide_checkpoint_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="gone.pt"):
        _make_jsrl_agent(str(tmp_path / "gone.pt"))
