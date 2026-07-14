"""Tests that training/real_world/hil_serl.py's entrypoints wire the right
pieces together -- env/wrapper composition, agent construction, and Loop
class selection (including the FWBW dual-agent case) -- without actually
running a robot or a training loop."""
from __future__ import annotations

import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.training.real_world.hil_serl import HilSerlArgs, _build_env


def test_hil_serl_args_override_real_robot_recipe_defaults_locally():
    """HilSerlArgs overrides backup_entropy/image_augmentation/pooling_method
    to match HIL-SERL's own real-robot recipe -- scoped here only, per
    docs/hil_serl_roadmap.md's "Behavioral Divergences" section."""
    hil_serl_args = HilSerlArgs()
    assert hil_serl_args.backup_entropy is False
    assert hil_serl_args.image_augmentation == "random_shift"
    assert hil_serl_args.pooling_method == "spatial_learned_embeddings"
    assert hil_serl_args.convert_obs_rotation is False

    from rl_garden.training.online.rlpd_hybrid import RLPDHybridArgs

    sim_args = RLPDHybridArgs()
    assert sim_args.backup_entropy is True
    assert sim_args.image_augmentation == "none"
    assert sim_args.pooling_method == "spatial_softmax"


class _FakeEnv(gym.Env):
    num_envs = 1

    def __init__(self):
        self.single_observation_space = spaces.Box(-1, 1, (4,), dtype="float32")
        self.single_action_space = spaces.Box(-1, 1, (3,), dtype="float32")
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.action_space = batch_space(self.single_action_space, 1)


class _FakeTeleop:
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def poll(self):
        return _NoInterventionSample()


class _NoInterventionSample:
    intervened = False
    episode_end = False


def _base_args(**overrides) -> HilSerlArgs:
    kwargs = dict(
        teleop_device="pico",
        obs_mode="state",
        hidden_dim=8,
        discrete_hidden_dim=8,
    )
    kwargs.update(overrides)
    return HilSerlArgs(**kwargs)


def test_build_env_skips_reward_classifier_wrapper_when_no_checkpoint_given(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )
    monkeypatch.setattr(
        "rl_garden.envs.wrappers.teleop_intervention.EETwistTeleOpWrapper", _FakeTeleop
    )
    args = _base_args(classifier_checkpoint=None)
    env = _build_env(args, env_request=None)

    from rl_garden.envs.wrappers.reward_classifier import RewardClassifierWrapper
    from rl_garden.envs.wrappers.teleop_intervention import TeleopInterventionWrapper

    assert isinstance(env, TeleopInterventionWrapper)
    assert not isinstance(env.env, RewardClassifierWrapper)


def test_build_env_composes_reward_classifier_wrapper_when_checkpoint_given(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_WristFakeEnv(), None),
    )
    monkeypatch.setattr(
        "rl_garden.envs.wrappers.teleop_intervention.EETwistTeleOpWrapper", _FakeTeleop
    )

    captured = {}

    def _fake_load_classifier_fn(checkpoint_path, observation_space, image_keys, device):
        captured["checkpoint_path"] = checkpoint_path
        captured["image_keys"] = image_keys
        return lambda obs: torch.zeros(1)

    monkeypatch.setattr(
        "rl_garden.models.reward.success.model.load_classifier_fn", _fake_load_classifier_fn
    )

    args = _base_args(classifier_checkpoint="/tmp/classifier.pt")
    env = _build_env(args, env_request=None)

    from rl_garden.envs.wrappers.reward_classifier import RewardClassifierWrapper
    from rl_garden.envs.wrappers.teleop_intervention import TeleopInterventionWrapper

    assert isinstance(env, TeleopInterventionWrapper)
    assert isinstance(env.env, RewardClassifierWrapper)
    assert captured["checkpoint_path"] == "/tmp/classifier.pt"
    assert captured["image_keys"] == ("wrist",)


def test_build_env_wraps_fwbw_outermost_when_fwbw_enabled(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )
    monkeypatch.setattr(
        "rl_garden.envs.wrappers.teleop_intervention.EETwistTeleOpWrapper", _FakeTeleop
    )
    args = _base_args(fwbw=True)
    env = _build_env(args, env_request=None)

    from rl_garden.envs.wrappers.fwbw_reset_free import FWBWResetFreeWrapper

    assert isinstance(env, FWBWResetFreeWrapper)


class _WristFakeEnv(_FakeEnv):
    def __init__(self):
        super().__init__()
        self.single_observation_space = spaces.Dict(
            {
                "state": spaces.Box(-1, 1, (4,), dtype="float32"),
                "wrist": spaces.Box(0, 255, (3, 8, 8), dtype="uint8"),
            }
        )


def test_run_actor_builds_scratch_agent_and_starts_actor_loop(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )
    monkeypatch.setattr(
        "rl_garden.envs.wrappers.teleop_intervention.EETwistTeleOpWrapper", _FakeTeleop
    )

    class _FakeActorLoop:
        def __init__(self, env, policy, sync_client, **kwargs):
            captured["kwargs"] = kwargs
            captured["sync_client"] = sync_client

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("rl_garden.real_world.hil_serl.HilSerlActorLoop", _FakeActorLoop)

    from rl_garden.training.real_world.hil_serl import _run_actor

    args = _base_args(role="actor", sync_host="10.0.0.1", sync_port=7000)
    _run_actor(args)

    assert captured["ran"] is True
    assert captured["sync_client"]._base_url == "http://10.0.0.1:7000"
    assert captured["kwargs"]["control_hz"] == args.control_hz


def test_run_actor_fwbw_builds_two_agents_and_starts_fwbw_actor_loop(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )
    monkeypatch.setattr(
        "rl_garden.envs.wrappers.teleop_intervention.EETwistTeleOpWrapper", _FakeTeleop
    )

    class _FakeFWBWActorLoop:
        def __init__(self, env, policies, sync_clients, **kwargs):
            captured["policies"] = set(policies)
            captured["sync_clients"] = {k: v._base_url for k, v in sync_clients.items()}

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("rl_garden.real_world.FWBWActorLoop", _FakeFWBWActorLoop)

    from rl_garden.training.real_world.hil_serl import _run_actor

    args = _base_args(role="actor", sync_host="10.0.0.1", sync_port=7000, fwbw=True)
    _run_actor(args)

    assert captured["ran"] is True
    assert captured["policies"] == {"forward", "backward"}
    assert captured["sync_clients"] == {
        "forward": "http://10.0.0.1:7000",
        "backward": "http://10.0.0.1:7001",
    }


def test_run_learner_builds_full_agent_and_starts_learner_loop(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )
    monkeypatch.setattr(
        "rl_garden.envs.wrappers.teleop_intervention.EETwistTeleOpWrapper", _FakeTeleop
    )

    class _FakeLearnerLoop:
        def __init__(self, agent, host, port, **kwargs):
            captured["host"] = host
            captured["port"] = port
            captured["kwargs"] = kwargs
            captured["demo_buffer_initialized"] = agent.offline_replay_buffer is not None

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("rl_garden.real_world.hil_serl.HilSerlLearnerLoop", _FakeLearnerLoop)

    from rl_garden.training.real_world.hil_serl import _run_learner

    args = _base_args(
        role="learner",
        sync_host="0.0.0.0",
        sync_port=6500,
        log_dir=str(tmp_path),
        log_type="none",
        demo_buffer_size=64,
        demo_data_ratio=0.3,
        buffer_period=50,
        demo_dataset_paths=["/tmp/demos/*.pkl"],
    )
    _run_learner(args)

    assert captured["ran"] is True
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 6500
    assert captured["demo_buffer_initialized"] is True
    assert captured["kwargs"]["buffer_period"] == 50
    assert captured["kwargs"]["demo_dataset_paths"] == ["/tmp/demos/*.pkl"]
    assert "checkpoint_dir" in captured["kwargs"]


def test_run_learner_fwbw_backward_binds_to_backward_port(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )
    monkeypatch.setattr(
        "rl_garden.envs.wrappers.teleop_intervention.EETwistTeleOpWrapper", _FakeTeleop
    )

    class _FakeLearnerLoop:
        def __init__(self, agent, host, port, **kwargs):
            captured["port"] = port

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("rl_garden.real_world.hil_serl.HilSerlLearnerLoop", _FakeLearnerLoop)

    from rl_garden.training.real_world.hil_serl import _run_learner

    args = _base_args(
        role="learner",
        sync_port=6500,
        fwbw=True,
        fwbw_direction="backward",
        log_dir=str(tmp_path),
        log_type="none",
    )
    _run_learner(args)

    assert captured["port"] == 6501


class _FakeEvalEnv(_FakeEnv):
    """Every episode is exactly one step and always succeeds (terminated=True),
    so eval_n_trajs episodes -> eval_n_trajs env.step calls, all successes."""

    def reset(self, seed=None):
        del seed
        return torch.zeros(1, 4), {}

    def step(self, action):
        return (
            torch.zeros(1, 4),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )


class _FakeEvalPolicy:
    def __init__(self):
        self.predict_calls: list[bool] = []

    def eval(self):
        pass

    def predict(self, obs, deterministic=False):
        self.predict_calls.append(deterministic)
        return torch.zeros(1, 3)


class _FakeEvalAgent:
    def __init__(self):
        self.policy = _FakeEvalPolicy()

    def _obs_to_policy_device(self, obs):
        return obs


def test_run_eval_requires_load_checkpoint(monkeypatch):
    from rl_garden.training.real_world.hil_serl import _run_eval

    args = _base_args(role="eval", load_checkpoint=None)
    try:
        _run_eval(args)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_eval_rejects_fwbw(monkeypatch):
    from rl_garden.training.real_world.hil_serl import _run_eval

    args = _base_args(role="eval", fwbw=True, load_checkpoint="/tmp/ckpt.pt")
    try:
        _run_eval(args)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_run_eval_runs_deterministic_episodes_and_reports_success_rate(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEvalEnv(), None),
    )
    monkeypatch.setattr(
        "rl_garden.envs.wrappers.teleop_intervention.EETwistTeleOpWrapper", _FakeTeleop
    )

    captured = {}
    fake_agent = _FakeEvalAgent()

    def _fake_build_rlpd_hybrid(args, env, offline_data, logger=None, checkpoint_dir=None):
        captured["load_checkpoint"] = args.load_checkpoint
        captured["checkpoint_dir"] = checkpoint_dir
        return fake_agent

    monkeypatch.setattr(
        "rl_garden.training.online.rlpd_hybrid.build_rlpd_hybrid", _fake_build_rlpd_hybrid
    )

    from rl_garden.training.real_world.hil_serl import _run_eval

    args = _base_args(role="eval", load_checkpoint="/tmp/ckpt.pt", eval_n_trajs=3)
    _run_eval(args)

    assert captured["load_checkpoint"] == "/tmp/ckpt.pt"
    assert captured["checkpoint_dir"] is None
    assert fake_agent.policy.predict_calls == [True, True, True]
