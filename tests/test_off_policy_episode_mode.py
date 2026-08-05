"""Unit tests for OffPolicyAlgorithm's optional episode-collection rollout mode.

`online_episodes_per_iteration` is a generic `OffPolicyAlgorithm` rollout
option (default `None` = today's fixed-`steps_per_env` loop, unchanged).
These tests exercise it through `Off2OnCalQL`, the first algorithm wired to
expose it, but the behavior under test lives entirely in
`OffPolicyAlgorithm.learn()`.
"""
import torch
from gymnasium import spaces

from rl_garden.algorithms.off2on_calql import Off2OnCalQL


class _ScriptedVecEnv:
    """Deterministic vector env: env i terminates every `episode_len[i]` steps."""

    def __init__(self, episode_len: list[int]) -> None:
        self.episode_len = list(episode_len)
        self.num_envs = len(self.episode_len)
        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=float)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype=float)
        self._steps_since_reset = [0] * self.num_envs

    def reset(self, seed=None):
        del seed
        self._steps_since_reset = [0] * self.num_envs
        return torch.zeros(self.num_envs, 4), {}

    def step(self, actions):
        obs = torch.zeros(self.num_envs, 4)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        for i in range(self.num_envs):
            self._steps_since_reset[i] += 1
            if self._steps_since_reset[i] >= self.episode_len[i]:
                terminations[i] = True
                self._steps_since_reset[i] = 0
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None


def _agent(env, **overrides) -> Off2OnCalQL:
    kwargs = dict(
        env=env,
        buffer_size=1000,
        buffer_device="cpu",
        learning_starts=0,
        batch_size=4,
        gamma=0.99,
        tau=0.005,
        training_freq=4,
        utd=1.0,
        net_arch={"pi": [8], "qf": [8]},
        n_critics=2,
        critic_subsample_size=2,
        use_cql_loss=False,
        use_calql=False,
        device="cpu",
        seed=0,
    )
    kwargs.update(overrides)
    return Off2OnCalQL(**kwargs)


def _stub_train(agent, monkeypatch) -> list[int]:
    """Replace agent.train with a no-op recorder of `gradient_steps`."""
    seen: list[int] = []

    def fake_train(gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        seen.append(gradient_steps)
        return {}

    monkeypatch.setattr(agent, "train", fake_train)
    return seen


def test_episode_mode_none_by_default_preserves_fixed_step_rollout(monkeypatch):
    env = _ScriptedVecEnv(episode_len=[1_000, 1_000])  # never completes in this test
    agent = _agent(env)
    assert agent.online_episodes_per_iteration is None
    grad_steps_seen = _stub_train(agent, monkeypatch)

    total = agent.num_envs * agent.steps_per_env * 3
    agent.learn(total_timesteps=total)

    assert agent._global_step == total
    assert grad_steps_seen
    assert all(g == agent.grad_steps_per_iteration for g in grad_steps_seen)


def test_episode_mode_single_env_collects_exact_trajectory_count(monkeypatch):
    episode_len = 3
    env = _ScriptedVecEnv(episode_len=[episode_len])
    agent = _agent(env, online_episodes_per_iteration=1, utd=2.0)
    grad_steps_seen = _stub_train(agent, monkeypatch)

    agent.learn(total_timesteps=episode_len * 3)

    # Every iteration collects exactly one `episode_len`-step trajectory
    # (single env, target=1), so grad steps == episode_len * utd every time.
    assert grad_steps_seen
    assert all(g == int(episode_len * 2.0) for g in grad_steps_seen)
    # global_step advances in exact episode_len increments, not fixed substeps.
    assert agent._global_step % episode_len == 0


def test_episode_mode_multi_env_grad_steps_match_collected_times_utd(monkeypatch):
    # env 0 completes every 2 steps, env 1 every 5 -- overshoot is expected
    # once every env has completed >= 1 episode.
    env = _ScriptedVecEnv(episode_len=[2, 5])
    agent = _agent(env, online_episodes_per_iteration=1, utd=1.5)
    grad_steps_seen = _stub_train(agent, monkeypatch)

    agent.learn(total_timesteps=20)

    assert grad_steps_seen
    # The first iteration must run until BOTH envs have completed >= 1
    # episode each -- env 1 (period 5) is the constraint, so it takes 5
    # substeps (5 transitions/env * 2 envs = 10 collected transitions),
    # overshooting env 0's 2-step episodes along the way.
    first_iter_transitions = 5 * agent.num_envs
    assert grad_steps_seen[0] == int(first_iter_transitions * 1.5)


def test_episode_mode_grad_steps_never_below_one(monkeypatch):
    # utd small enough that collected*utd would floor to 0 without the floor.
    env = _ScriptedVecEnv(episode_len=[1])
    agent = _agent(env, online_episodes_per_iteration=1, utd=0.01)
    grad_steps_seen = _stub_train(agent, monkeypatch)

    agent.learn(total_timesteps=5)

    assert grad_steps_seen
    assert all(g >= 1 for g in grad_steps_seen)


def test_episode_mode_eval_fires_when_boundary_is_outside_a_training_freq_window(monkeypatch):
    """Regression: the eval-boundary check used to test a hardcoded
    `training_freq`-wide window ending at the current global_step, not the
    just-finished iteration's actual span. In episode_mode an iteration can
    collect far more than `training_freq` steps, so an eval_freq boundary
    crossed earlier in that span (outside the narrow tail window) used to be
    missed entirely.

    training_freq=4 (the `_agent` default) and episode_len=20 -> the old
    `(global_step - training_freq)` window only ever covered the iteration's
    last 4 steps, so a boundary at step 15 (eval_freq=15) inside a [0, 20)
    span fell outside it and was never detected.
    """
    episode_len = 20
    env = _ScriptedVecEnv(episode_len=[episode_len])
    agent = _agent(env, online_episodes_per_iteration=1, eval_freq=15, num_eval_steps=1)
    _stub_train(agent, monkeypatch)

    eval_calls: list[int] = []
    monkeypatch.setattr(agent, "_evaluate", lambda: eval_calls.append(agent._global_step) or {})

    agent.learn(total_timesteps=episode_len * 2)

    # Iteration 1 spans [0, 20) and crosses the eval_freq=15 boundary; that
    # crossing is detected at the top of iteration 2 (global_step=20 there).
    assert eval_calls == [20]


def test_learning_has_started_seeded_from_global_step_not_always_false(monkeypatch):
    """A resumed/post-offline run already past learning_starts must not take
    a random first action -- _rollout_action(learning_has_started=False)
    falls back to exploration."""
    env = _ScriptedVecEnv(episode_len=[1_000])
    agent = _agent(env, learning_starts=0)
    _stub_train(agent, monkeypatch)
    agent._global_step = 100  # simulate resumed/post-offline-pretraining state

    seen_flags: list[bool] = []
    original = agent._rollout_action

    def spy(obs, learning_has_started):
        seen_flags.append(learning_has_started)
        return original(obs, learning_has_started)

    monkeypatch.setattr(agent, "_rollout_action", spy)
    agent.learn(total_timesteps=agent._global_step + agent.num_envs)

    assert seen_flags
    assert seen_flags[0] is True
