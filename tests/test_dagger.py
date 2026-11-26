"""Tests for DAgger: composes BC (unmodified) with DemoInterventionMixin's
growing buffer (unmodified) for interactive imitation learning."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms.dagger import DAgger

OBS_DIM = 4
ACTION_DIM = 2


class _FakeEnv:
    """Fixed-episode-length auto-reset fake vector env, same shape as
    test_acfql_smoke.py's fixture."""

    def __init__(self, num_envs: int = 3, episode_len: int = 5) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self._step_count = torch.zeros(num_envs, dtype=torch.long)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def reset(self, seed=None):
        del seed
        self._step_count.zero_()
        return torch.randn(self.num_envs, OBS_DIM), {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= self.episode_len
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        reward = torch.ones(self.num_envs)
        self._step_count[terminated] = 0
        obs = torch.randn(self.num_envs, OBS_DIM)
        return obs, reward, terminated, truncated, {}


class _ConstantExpert:
    """Deterministic mock expert: always returns a fixed, easily-identified
    action so tests can distinguish expert-labeled data from policy actions."""

    def __init__(self, value: float = 0.75, action_dim: int = ACTION_DIM) -> None:
        self.value = value
        self.action_dim = action_dim
        self.call_count = 0

    def __call__(self, obs) -> torch.Tensor:
        self.call_count += 1
        # obs is Dict({"state": Box}) -- _FakeEnv's bare Box observation
        # space is boundary-normalized by BaseAlgorithm.__init__ (see
        # rl_garden.envs.wrappers.VectorizedDictStateWrapper).
        batch = obs["state"].shape[0]
        return torch.full((batch, self.action_dim), self.value)


def _make_agent(**kwargs) -> DAgger:
    defaults = {
        "env": _FakeEnv(num_envs=3),
        "expert": _ConstantExpert(),
        "demo_buffer_size": 1000,
        "rollout_steps_per_round": 10,
        "gradient_steps_per_round": 2,
        "batch_size": 8,
        "buffer_device": "cpu",
        "device": "cpu",
        "net_arch": [16, 16],
    }
    defaults.update(kwargs)
    return DAgger(**defaults)


def test_mro_puts_mixin_before_bc():
    mro_names = [c.__name__ for c in DAgger.__mro__]
    assert mro_names.index("DemoInterventionMixin") < mro_names.index("BC")


def test_sample_train_batch_resolves_to_mixing_version():
    agent = _make_agent()
    # BC's own _sample_train_batch takes no args; the mixin's takes
    # batch_size explicitly. If MRO were wrong this call would raise.
    agent.collect_round(num_steps=5, beta=1.0)
    data = agent._sample_train_batch(agent.batch_size)
    assert data.actions.shape[0] == agent.batch_size


def test_online_replay_buffer_stays_empty_offline_buffer_grows():
    agent = _make_agent()
    assert len(agent.replay_buffer) == 0
    agent.collect_round(num_steps=10, beta=1.0)
    assert len(agent.replay_buffer) == 0  # DAgger never writes here
    assert len(agent.offline_replay_buffer) == 10 * agent.num_envs


def test_collect_round_always_labels_with_expert_action_even_when_beta_low():
    expert = _ConstantExpert(value=0.75)
    agent = _make_agent(expert=expert, beta_rounds=1)
    beta = 0.0  # policy drives execution entirely, expert only labels
    agent.collect_round(num_steps=20, beta=beta)
    assert expert.call_count == 20  # expert queried every step regardless of beta

    n = len(agent.offline_replay_buffer)
    sample = agent.offline_replay_buffer.sample(n)
    assert torch.allclose(sample.actions, torch.full_like(sample.actions, 0.75))


def test_beta_schedule_decays_linearly_to_zero():
    agent = _make_agent(beta_rounds=10)
    assert agent.beta_schedule(0) == 1.0
    assert agent.beta_schedule(5) == 0.5
    assert agent.beta_schedule(10) == 0.0
    assert agent.beta_schedule(20) == 0.0  # clamped, not negative


def test_learn_runs_rounds_and_grows_demo_buffer():
    agent = _make_agent(rollout_steps_per_round=5, gradient_steps_per_round=2, beta_rounds=3)
    total_timesteps = 5 * agent.num_envs * 4  # 4 rounds worth
    agent.learn(total_timesteps)
    assert agent._round_num == 4
    assert len(agent.offline_replay_buffer) == 5 * agent.num_envs * 4
    assert len(agent.replay_buffer) == 0


def test_dagger_policy_is_plain_bc_policy_class():
    # Confirms DAgger reuses BC's policy machinery unmodified rather than
    # introducing a new policy type.
    from rl_garden.policies.bc_policy import BCPolicy

    agent = _make_agent()
    assert isinstance(agent.policy, BCPolicy)


def test_demo_buffer_is_saved_and_loaded_with_the_replay_buffer(tmp_path):
    agent = _make_agent()
    agent.collect_round(num_steps=4, beta=1.0)
    path = agent.save(tmp_path / "checkpoint_1.pt", include_replay_buffer=True)

    loaded = _make_agent()
    loaded.load(path, load_replay_buffer=False)
    assert len(loaded.offline_replay_buffer) == 0
    loaded.load(path, load_replay_buffer=True)

    samples = []
    for dagger in (agent, loaded):
        torch.manual_seed(0)
        samples.append(dagger.offline_replay_buffer.sample(8))
    assert torch.equal(samples[0].actions, samples[1].actions)
    assert all(torch.equal(samples[0].obs[key], samples[1].obs[key]) for key in samples[0].obs)


def test_demo_buffer_is_not_written_without_the_replay_buffer(tmp_path):
    agent = _make_agent()
    agent.collect_round(num_steps=4, beta=1.0)
    agent.save(tmp_path / "checkpoint_1.pt")
    assert not list(tmp_path.glob("demo_buffer_*"))


def test_missing_demo_buffer_file_is_named(tmp_path):
    path = _make_agent().save(tmp_path / "checkpoint_1.pt", include_replay_buffer=True)
    (tmp_path / "demo_buffer_1.pt").unlink()
    with pytest.raises(FileNotFoundError, match="demo_buffer_1.pt"):
        _make_agent().load(path, load_replay_buffer=True)
    _make_agent().load(path, load_replay_buffer=False)


def _forget_demo_buffer_reference(path) -> None:
    contents = torch.load(path)
    del contents["state"]["extra"]["demo_buffer_path"]
    torch.save(contents, path)


def test_checkpoint_that_records_no_demo_buffer_resumes_with_a_warning(tmp_path):
    path = _make_agent().save(tmp_path / "checkpoint_1.pt", include_replay_buffer=True)
    _forget_demo_buffer_reference(path)
    (tmp_path / "demo_buffer_1.pt").unlink()
    agent = _make_agent()
    with pytest.warns(UserWarning, match="demo_buffer_1.pt"):
        agent.load(path, load_replay_buffer=True)
    assert len(agent.offline_replay_buffer) == 0


def test_checkpoint_that_records_no_demo_buffer_loads_the_file_next_to_it(tmp_path):
    agent = _make_agent()
    agent.collect_round(num_steps=4, beta=1.0)
    path = agent.save(tmp_path / "checkpoint_1.pt", include_replay_buffer=True)
    _forget_demo_buffer_reference(path)
    loaded = _make_agent()
    loaded.load(path, load_replay_buffer=True)
    assert len(loaded.offline_replay_buffer) == len(agent.offline_replay_buffer) > 0


def test_renamed_checkpoint_that_records_no_demo_buffer_follows_its_replay_buffer(tmp_path):
    agent = _make_agent()
    agent.collect_round(num_steps=4, beta=1.0)
    path = agent.save(tmp_path / "checkpoint_1.pt", include_replay_buffer=True)
    _forget_demo_buffer_reference(path)
    path = path.rename(tmp_path / "best.pt")
    loaded = _make_agent()
    loaded.load(path, load_replay_buffer=True)
    assert len(loaded.offline_replay_buffer) == len(agent.offline_replay_buffer) > 0

    (tmp_path / "demo_buffer_1.pt").unlink()
    with pytest.warns(UserWarning, match="demo_buffer_1.pt"):
        _make_agent().load(path, load_replay_buffer=True)


def test_checkpoint_saved_without_buffers_loads_silently(tmp_path):
    path = _make_agent().save(tmp_path / "checkpoint_1.pt")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _make_agent().load(path, load_replay_buffer=True)


def test_renamed_checkpoint_loads_the_demo_buffer_it_references(tmp_path):
    agent = _make_agent()
    agent.collect_round(num_steps=4, beta=1.0)
    path = agent.save(tmp_path / "checkpoint_1.pt", include_replay_buffer=True)
    loaded = _make_agent()
    loaded.load(path.rename(tmp_path / "best.pt"), load_replay_buffer=True)
    assert len(loaded.offline_replay_buffer) == len(agent.offline_replay_buffer) > 0


# --- CLI mock experts, devices, logging, checkpoints ---


class _RecordingLogger:
    def __init__(self) -> None:
        self.scalars: dict[str, list[tuple[float, int]]] = {}

    def add_scalar(self, key, value, step) -> None:
        self.scalars.setdefault(key, []).append((float(value), step))

    def log_metrics(self, metrics, step) -> None:
        for key, value in metrics.items():
            self.add_scalar(f"losses/{key}", value, step)


def _cli_expert(name, space=None):
    from types import SimpleNamespace

    from rl_garden.training.online.entrypoints.dagger import _mock_expert_from_args

    space = space or spaces.Box(-2.0, 3.0, (ACTION_DIM,), np.float32)
    return _mock_expert_from_args(SimpleNamespace(expert=name), space)


def test_mock_experts_answer_dict_observations_with_bounded_float32_actions():
    obs = {
        "state": torch.zeros(5, OBS_DIM),
        "rgb": torch.zeros(5, 3, 8, 8, dtype=torch.uint8),
    }
    zero = _cli_expert("zero")(obs)
    assert zero.shape == (5, ACTION_DIM) and zero.dtype == torch.float32 and not zero.any()
    uniform = _cli_expert("random_uniform")(obs)
    assert uniform.shape == (5, ACTION_DIM) and uniform.dtype == torch.float32
    assert bool(((uniform >= -2.0) & (uniform <= 3.0)).all())


def test_mock_experts_refuse_discrete_actions():
    import pytest

    with pytest.raises(NotImplementedError, match="Box"):
        _cli_expert("zero", spaces.Discrete(3))


def test_learn_logs_every_round_and_saves_periodic_and_final_checkpoints(tmp_path):
    logger = _RecordingLogger()
    agent = _make_agent(
        expert=_cli_expert("zero"),
        logger=logger,
        checkpoint_dir=str(tmp_path),
        checkpoint_freq=30,
        rollout_steps_per_round=5,
        beta_rounds=4,
    )
    agent.learn(5 * agent.num_envs * 4)
    assert [step for _, step in logger.scalars["dagger/beta"]] == [15, 30, 45, 60]
    assert [v for v, _ in logger.scalars["dagger/beta"]] == [1.0, 0.75, 0.5, 0.25]
    assert [v for v, _ in logger.scalars["dagger/demo_buffer_size"]] == [15, 30, 45, 60]
    assert len(logger.scalars["losses/actor_loss"]) == 4
    names = {path.name for path in tmp_path.glob("*.pt")}
    assert names == {"checkpoint_30.pt", "checkpoint_60.pt", "final.pt"}


def test_resumed_run_continues_the_beta_schedule_and_demo_buffer(tmp_path):
    first = _make_agent(
        checkpoint_dir=str(tmp_path), save_replay_buffer=True, rollout_steps_per_round=5,
        beta_rounds=6,
    )
    first.learn(5 * first.num_envs * 2)
    assert first._round_num == 2

    resumed = _make_agent(rollout_steps_per_round=5, beta_rounds=6)
    resumed.load(tmp_path / "final.pt", load_replay_buffer=True)
    assert resumed._round_num == 2
    assert resumed.beta_schedule(resumed._round_num) == first.beta_schedule(2)
    assert len(resumed.offline_replay_buffer) == len(first.offline_replay_buffer) == 30
    resumed.learn(5 * resumed.num_envs * 3)
    assert resumed._round_num == 3
    assert len(resumed.offline_replay_buffer) == 45


def test_cpu_env_with_cuda_policy_collects_on_the_buffer_device():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    agent = _make_agent(device="cuda", buffer_device="cuda", expert=_cli_expert("random_uniform"))
    agent.collect_round(num_steps=6, beta=0.5)
    buffer = agent.offline_replay_buffer
    assert len(buffer) == 6 * agent.num_envs
    assert buffer.actions.device.type == "cuda" and buffer.dones.device.type == "cuda"
