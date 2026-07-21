from __future__ import annotations

import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.training.hitl.residual_hil_serl import (
    ResidualHilSerlArgs,
    _build_env,
)


class _FakeEnv(gym.Env):
    num_envs = 1

    def __init__(self):
        self.single_observation_space = spaces.Box(-1, 1, (4,), dtype="float32")
        self.single_action_space = spaces.Box(-1, 1, (6,), dtype="float32")
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.action_space = batch_space(self.single_action_space, 1)


class _FakeTeleop:
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass


class _FakePolicy:
    def eval(self):
        pass


class _FakeAgent:
    def __init__(self):
        self.policy = _FakePolicy()
        self.device = torch.device("cpu")
        self.demo_init = None

    def init_demo_buffer(self, buffer_size, demo_data_ratio):
        self.demo_init = (buffer_size, demo_data_ratio)


def _args(**overrides) -> ResidualHilSerlArgs:
    kwargs = dict(
        env_backend="maniskill",
        obs_mode="state",
        control_mode="pd_ee_twist",
        base_policy="zero",
        hidden_dim=8,
        actor_hidden_layers=1,
        critic_hidden_layers=1,
    )
    kwargs.update(overrides)
    return ResidualHilSerlArgs(**kwargs)


def test_build_env_wraps_teleop_for_maniskill_without_classifier(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )
    monkeypatch.setattr(
        "rl_garden.envs.wrappers.teleop_intervention.EETwistTeleOpWrapper",
        _FakeTeleop,
    )

    env = _build_env(
        _args(teleop_record_gripper=False),
        env_request=None,
        enable_teleop=True,
        enable_classifier=True,
    )

    from rl_garden.envs.wrappers.reward_classifier import RewardClassifierWrapper
    from rl_garden.envs.wrappers.teleop_intervention import TeleopInterventionWrapper

    assert isinstance(env, TeleopInterventionWrapper)
    assert env.record_gripper is False
    assert not isinstance(env.env, RewardClassifierWrapper)


def test_run_actor_builds_scratch_agent_and_residual_actor_loop(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "rl_garden.training.hitl.residual_hil_serl._build_env",
        lambda args, env_request, **kwargs: _FakeEnv(),
    )

    def _fake_build(args, env, eval_env, logger, checkpoint_dir):
        captured["build_args"] = args
        captured["logger"] = logger
        captured["checkpoint_dir"] = checkpoint_dir
        return _FakeAgent()

    monkeypatch.setattr(
        "rl_garden.training.hitl.residual_hil_serl.build_residual_hil_serl",
        _fake_build,
    )

    class _FakeLoop:
        def __init__(self, env, agent, sync_client, **kwargs):
            captured["loop_kwargs"] = dict(
                env=env, agent=agent, sync_client=sync_client, **kwargs
            )

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(
        "rl_garden.training.hitl.residual_hil_serl.ResidualHilSerlActorLoop",
        _FakeLoop,
    )

    from rl_garden.training.hitl.residual_hil_serl import _run_actor

    args = _args(role="actor", sync_host="10.0.0.1", sync_port=7000, buffer_size=1000)
    _run_actor(args)

    assert captured["build_args"].buffer_size == 8
    assert captured["build_args"].offline_dataset_path is None
    assert captured["logger"] is None
    assert captured["checkpoint_dir"] is None
    assert captured["ran"] is True
    assert captured["loop_kwargs"]["sync_client"]._base_url == "http://10.0.0.1:7000"


def test_run_learner_initializes_demo_buffer_and_loop(monkeypatch, tmp_path):
    captured = {}
    fake_agent = _FakeAgent()

    monkeypatch.setattr(
        "rl_garden.training.hitl.residual_hil_serl._build_env",
        lambda args, env_request, **kwargs: _FakeEnv(),
    )
    monkeypatch.setattr(
        "rl_garden.training.hitl.residual_hil_serl.build_residual_hil_serl",
        lambda args, env, eval_env, logger, checkpoint_dir: fake_agent,
    )

    class _FakeLogger:
        def close(self):
            captured["closed"] = True

    monkeypatch.setattr("rl_garden.common.Logger.create", lambda **kwargs: _FakeLogger())

    class _FakeLearnerLoop:
        def __init__(self, agent, host, port, **kwargs):
            captured["learner"] = dict(agent=agent, host=host, port=port, **kwargs)

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(
        "rl_garden.real_world.hil_serl.HilSerlLearnerLoop",
        _FakeLearnerLoop,
    )

    from rl_garden.training.hitl.residual_hil_serl import _run_learner

    args = _args(
        role="learner",
        sync_host="0.0.0.0",
        sync_port=6000,
        log_dir=str(tmp_path),
        log_type="none",
        demo_buffer_size=64,
        demo_data_ratio=0.25,
        buffer_period=50,
    )
    _run_learner(args)

    assert fake_agent.demo_init == (64, 0.25)
    assert captured["ran"] is True
    assert captured["closed"] is True
    assert captured["learner"]["host"] == "0.0.0.0"
    assert captured["learner"]["port"] == 6000
    assert captured["learner"]["buffer_period"] == 50
