"""Unit tests for OffPolicyAlgorithm's optional episode-collection rollout mode.

`online_episodes_per_iteration` is a generic `OffPolicyAlgorithm` rollout
option (default `None` = today's fixed-`steps_per_env` loop, unchanged).
These tests exercise it through `Off2OnCalQL`, the first algorithm wired to
expose it, but the behavior under test lives entirely in
`OffPolicyAlgorithm.learn()`.
"""
import re

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


def test_policy_is_in_eval_mode_during_rollout_collection(monkeypatch):
    """Regression: rollout action selection never toggled the policy to eval
    mode -- policy.eval()/.train() only appeared in BaseAlgorithm._evaluate()
    (the eval-env loop), never around OffPolicyAlgorithm.learn()'s rollout
    substep loop. If actor_dropout_rate/critic_dropout_rate were set, the
    actions used for real env interaction (and stored in the replay buffer)
    would be computed with dropout still active. learn() now brackets the
    rollout substep loop with policy.eval()/.train(), mirroring the existing
    _evaluate() pattern."""
    env = _ScriptedVecEnv(episode_len=[1_000, 1_000])  # never completes in this test
    agent = _agent(env)
    _stub_train(agent, monkeypatch)

    training_mode_during_rollout: list[bool] = []
    original = agent._rollout_action

    def spy(obs, learning_has_started):
        training_mode_during_rollout.append(agent.policy.training)
        return original(obs, learning_has_started)

    monkeypatch.setattr(agent, "_rollout_action", spy)

    assert agent.policy.training is True  # freshly constructed nn.Module default

    total = agent.num_envs * agent.steps_per_env * 3
    agent.learn(total_timesteps=total)

    assert training_mode_during_rollout
    assert all(mode is False for mode in training_mode_during_rollout)
    assert agent.policy.training is True  # restored before/after train()


class _ScriptedVecEnvWithEpisodeInfo(_ScriptedVecEnv):
    """Like _ScriptedVecEnv, but also populates the Gymnasium vector-env
    final_info/_final_info convention learn() reads for episode metrics.

    `reward_per_env`, if given, overrides the per-env reward value reported
    as the episode "return" (the base class always uses a constant 1.0 for
    every env, which can't distinguish which env a given completion came
    from -- some tests need per-env-distinct values for that)."""

    def __init__(self, episode_len: list[int], reward_per_env: list[float] | None = None) -> None:
        super().__init__(episode_len)
        self.reward_per_env = (
            torch.tensor(reward_per_env, dtype=torch.float32)
            if reward_per_env is not None
            else None
        )

    def step(self, actions):
        obs, rewards, terminations, truncations, _ = super().step(actions)
        if self.reward_per_env is not None:
            rewards = self.reward_per_env.clone()
        done_mask = terminations | truncations
        infos = {}
        if bool(done_mask.any()):
            infos["_final_info"] = done_mask
            infos["final_info"] = {"episode": {"return": rewards.clone()}}
        return obs, rewards, terminations, truncations, infos


def test_stats_window_size_none_preserves_per_iteration_nan(monkeypatch, capsys):
    """stats_window_size defaults to None: return=/success_at_end= must keep
    their original per-iteration-only meaning (nan when the iteration
    completed zero episodes), unchanged from pre-rolling-window behavior, and
    no return_w{N}=/success_w{N}= field should be printed at all.

    episode_len=20, training_freq=4 (the _agent default), num_envs=1 ->
    steps_per_env=4, so global_step advances by 4 every iteration and the
    episode (length 20) completes only once every 5 iterations. The last of
    7 iterations (steps 25-28) completes zero episodes, so its return= must
    be nan even though an earlier iteration did complete one.
    """
    episode_len = 20
    env = _ScriptedVecEnvWithEpisodeInfo(episode_len=[episode_len])
    agent = _agent(env, log_freq=1)
    assert agent.stats_window_size is None
    _stub_train(agent, monkeypatch)

    agent.learn(total_timesteps=4 * 7)

    out = capsys.readouterr().out
    train_lines = [line for line in out.splitlines() if line.startswith("[train]")]
    assert len(train_lines) == 7
    last_line = train_lines[-1]
    assert "step=28/28" in last_line
    match = re.search(r"return=(\S+)", last_line)
    assert match is not None
    assert match.group(1) == "nan"
    assert "return_w" not in last_line
    assert "success_w" not in last_line


def test_stats_window_size_set_adds_separate_per_episode_weighted_field(monkeypatch, capsys):
    """With stats_window_size set, a separate return_w{N}= field must appear
    (even on an iteration whose own return= is nan) and must be a genuine
    per-episode mean -- not skewed toward iterations where more envs happened
    to finish simultaneously in the same completion batch.

    Two envs: env 0 has episode_len=2 (finishes every iteration, alone), env
    1 has episode_len=8 (finishes once, together with env 0, at iteration 4).
    reward_per_env gives env 0's completions a distinct "return" marker
    (2.0) from env 1's (8.0), so a batch-mean-weighted window (the old bug)
    and a true per-episode-weighted window produce different numbers,
    letting the test distinguish them.
    """
    env = _ScriptedVecEnvWithEpisodeInfo(episode_len=[2, 8], reward_per_env=[2.0, 8.0])
    agent = _agent(env, log_freq=1, stats_window_size=10)
    assert agent.stats_window_size == 10
    _stub_train(agent, monkeypatch)

    # training_freq=4, num_envs=2 -> steps_per_env=2, so 4 iterations cover
    # global_step 0->8: env 0 finishes at iterations 1,2,3,4 (4 completions,
    # each alone); env 1 finishes only at iteration 4 (alongside env 0).
    agent.learn(total_timesteps=4 * 4)

    out = capsys.readouterr().out
    train_lines = [line for line in out.splitlines() if line.startswith("[train]")]
    assert len(train_lines) == 4

    # Iteration 3 (global_step 6->8 is iteration 4; iteration 3 is
    # global_step 4->6) completes zero episodes on its own for env 1, but
    # env 0 finishes at every iteration boundary -- pick the first line to
    # confirm the window field is present independent of that iteration's
    # own completions.
    first_line = train_lines[0]
    match = re.search(r"return_w10=(\S+)", first_line)
    assert match is not None
    assert match.group(1) != "nan"

    # Five episode completions total: env 0 finishes 4 times (return=2.0
    # each) and env 1 finishes once (return=8.0), all within the window
    # (maxlen=10). True per-episode mean: (4*2.0 + 1*8.0) / 5 = 3.2.
    # A batch-mean-weighted window (the old bug) would instead average one
    # entry per completion *batch* -- 3 solo-env-0 batches (2.0 each) and 1
    # joint batch containing both env 0 and env 1 ((2.0+8.0)/2=5.0) -> a
    # batch-weighted mean of (3*2.0 + 1*5.0) / 4 = 2.75, a different value.
    last_line = train_lines[-1]
    match = re.search(r"return_w10=(\S+)", last_line)
    assert match is not None
    assert abs(float(match.group(1)) - 3.2) < 1e-6


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
