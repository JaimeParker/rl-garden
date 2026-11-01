from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_garden.training import evaluation as eval_checkpoint


def _load_example_wrapper():
    path = Path(__file__).resolve().parents[1] / "examples" / "eval_checkpoint.py"
    spec = importlib.util.spec_from_file_location("example_eval_checkpoint", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_wrapper_reexports_the_package_evaluation():
    wrapper = _load_example_wrapper()

    for name in (
        "EvalCheckpointArgs",
        "evaluate_agent",
        "evaluate_checkpoint",
        "main",
        "_algorithm_from_class",
        "_apply_mapping_to_args",
        "_builder_from_module",
        "_config_args",
        "_env_request",
        "_headline_metrics",
        "_load_training_args",
        "_resolve_phase_and_algorithm",
        "_training_module",
        "_write_json",
    ):
        assert getattr(wrapper, name) is getattr(eval_checkpoint, name), name


class _Policy:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _Agent:
    def __init__(self):
        self.policy = _Policy()
        self.started = False
        self.finalized = False
        self.hook_steps = 0

    def _eval_start_hook(self):
        self.started = True

    def _eval_action_and_critic_action(self, obs):
        batch = obs.shape[0]
        action = torch.zeros(batch, 1)
        return action, action

    def _eval_step_hook(
        self, obs_before, critic_action, rewards, terminations, truncations, infos
    ):
        self.hook_steps += 1

    def _eval_finalize_hook(self):
        self.finalized = True
        return {"extra_metric": 7.0}


class _VectorEvalEnv:
    num_envs = 2

    def __init__(self):
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return torch.zeros(self.num_envs, 3), {}

    def step(self, action):
        self.step_count += 1
        obs = torch.zeros(self.num_envs, 3)
        rewards = torch.tensor([1.0, 2.0])
        terminations = torch.tensor([self.step_count == 1, self.step_count == 2])
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.step_count == 1:
            infos = {
                "_final_info": torch.tensor([True, False]),
                "final_info": {
                    "episode": {
                        "return": torch.tensor([11.0, 0.0]),
                        "success_at_end": torch.tensor([1.0, 0.0]),
                    }
                },
            }
        else:
            infos = {
                "_final_info": torch.tensor([False, True]),
                "final_info": {
                    "episode": {
                        "return": torch.tensor([0.0, 22.0]),
                        "success_at_end": torch.tensor([0.0, 0.0]),
                    }
                },
            }
        return obs, rewards, terminations, truncations, infos


def test_algorithm_from_class_handles_off2on_prefix_and_alias():
    assert eval_checkpoint._algorithm_from_class("Off2OnIQL") == "iql"
    assert eval_checkpoint._algorithm_from_class("OfflineCalQL") == "calql"


def test_headline_metrics_prefers_success_at_end():
    metrics = {
        "return": 3.5,
        "success_at_end": 0.25,
        "success_once": 0.75,
        "episodes_completed": 8.0,
    }

    headline = eval_checkpoint._headline_metrics(metrics)

    assert headline["average_return"] == 3.5
    assert headline["success_rate"] == 0.25
    assert headline["success_once"] == 0.75


def test_apply_mapping_to_args_updates_nested_dataclass():
    args = eval_checkpoint.EvalCheckpointArgs()

    eval_checkpoint._apply_mapping_to_args(
        args,
        {
            "env_id": "StackCube-v1",
            "maniskill": {"sim_backend": "physx_cpu", "reward_mode": "dense"},
            "unknown": "ignored",
        },
    )

    assert args.env_id == "StackCube-v1"
    assert args.maniskill.sim_backend == "physx_cpu"
    assert args.maniskill.reward_mode == "dense"
    assert not hasattr(args, "unknown")


def test_apply_mapping_to_args_replaces_frozen_dataclass():
    from rl_garden.observations import ObservationConfig

    args = eval_checkpoint.EvalCheckpointArgs()

    obs_mapping = {
        "depth": [],
        "extra_state": [],
        "frame_stack": 1,
        "image_size": None,
        "rgb": ["cam0"],
        "state": True,
        "unknown": 1,
    }
    eval_checkpoint._apply_mapping_to_args(args, {"obs": obs_mapping})

    assert isinstance(args.obs, ObservationConfig)
    assert args.obs.rgb == ("cam0",)
    assert args.obs.depth == ()
    assert args.obs.frame_stack == 1
    assert not hasattr(args.obs, "unknown")

    from rl_garden.training.off2on.registry import registry as off2on_registry

    off2on_registry.discover()
    calql_args_cls = off2on_registry.entries()["calql"].args_cls
    calql_args = calql_args_cls()
    eval_checkpoint._apply_mapping_to_args(calql_args, {"obs": obs_mapping})
    assert calql_args.obs.rgb == ("cam0",)


def test_apply_mapping_to_args_skips_none_for_dataclass_field():
    """Checkpoint metadata serializes an unset sentinel dataclass field (see
    resolve_obs_groups_config in rl_garden/common/cli_args.py) as a bare
    ``None``. Applying it must not replace the dataclass attribute with
    ``None`` (real crash path: off2on CalQL args, hyperparameters carrying
    ``obs_groups: None``, then resolve_obs_groups_config dereferencing
    ``.actor`` on ``None``)."""
    from rl_garden.observations import ObsGroups
    from rl_garden.training.off2on.registry import registry as off2on_registry

    off2on_registry.discover()
    calql_args = off2on_registry.entries()["calql"].args_cls()

    eval_checkpoint._apply_mapping_to_args(calql_args, {"obs_groups": None})

    assert isinstance(calql_args.obs_groups, ObsGroups)
    assert calql_args.obs_groups.actor is None
    assert calql_args.obs_groups.critic is None


def test_apply_mapping_to_args_converts_json_float_sentinels():
    """rl_garden.common.effective_config.json_value() serializes non-finite
    float fields (e.g. CQLArgs.cql_clip_diff_min/max defaulting to +-inf) as
    the JSON strings "Infinity"/"-Infinity"/"NaN", because config.json is
    written with json.dumps(..., allow_nan=False). Reading such a run's
    config.json back through _apply_mapping_to_args must invert that
    encoding, or a float-typed field ends up holding a Python str and
    rl_garden/algorithms/cql.py's `_cql_regularizer` crashes with
    `TypeError: clamp() received (Tensor, str, str)`."""
    from rl_garden.training.off2on.registry import registry as off2on_registry

    off2on_registry.discover()
    calql_args = off2on_registry.entries()["calql"].args_cls()

    eval_checkpoint._apply_mapping_to_args(
        calql_args,
        {
            "cql_clip_diff_min": "-Infinity",
            "cql_clip_diff_max": "Infinity",
            "env_id": "Infinity",
            "seed": 5,
        },
    )

    assert calql_args.cql_clip_diff_min == float("-inf")
    assert isinstance(calql_args.cql_clip_diff_min, float)
    assert calql_args.cql_clip_diff_max == float("inf")
    assert isinstance(calql_args.cql_clip_diff_max, float)
    # A non-float field holding the literal string "Infinity" must not be
    # coerced -- the sentinel conversion is gated on the current attribute
    # already being a float.
    assert calql_args.env_id == "Infinity"
    assert calql_args.seed == 5

    import math

    eval_checkpoint._apply_mapping_to_args(
        calql_args, {"cql_clip_diff_min": "NaN"}
    )
    assert math.isnan(calql_args.cql_clip_diff_min)

    # Finite float values pass through unchanged.
    eval_checkpoint._apply_mapping_to_args(
        calql_args, {"cql_clip_diff_max": -200.0}
    )
    assert calql_args.cql_clip_diff_max == -200.0


def test_append_metric_values_masks_final_info():
    metrics = {}
    mask = torch.tensor([False, True, True])

    appended = eval_checkpoint._append_metric_values(
        metrics,
        {"return": torch.tensor([1.0, 2.0, 3.0])},
        mask,
        remaining=1,
    )

    assert appended == 1
    assert metrics == {"return": [2.0]}


def test_append_metric_values_aliases_raw_gym_keys_and_drops_wall_clock_time():
    metrics = {}
    mask = torch.tensor([True])

    appended = eval_checkpoint._append_metric_values(
        metrics,
        {
            "r": torch.tensor([5.0]),
            "l": torch.tensor([12.0]),
            "t": torch.tensor([0.031]),
            "success": torch.tensor([1.0]),
            "_r": torch.tensor([True]),
        },
        mask,
        remaining=1,
    )

    assert appended == 1
    assert metrics == {
        "return": [5.0],
        "episode_len": [12.0],
        "success_at_end": [1.0],
    }


def test_evaluate_agent_collects_completed_episode_metrics():
    agent = _Agent()
    env = _VectorEvalEnv()

    metrics = eval_checkpoint.evaluate_agent(
        agent,
        env,
        seed=123,
        num_eval_episodes=2,
        max_eval_steps=4,
    )

    assert metrics["return"] == pytest.approx(16.5)
    assert metrics["success_at_end"] == pytest.approx(0.5)
    assert metrics["episodes_completed"] == 2.0
    assert metrics["eval_steps"] == 2.0
    assert metrics["extra_metric"] == 7.0
    assert agent.started
    assert agent.finalized
    assert agent.hook_steps == 2
    assert agent.policy.training


def test_resolve_phase_auto_reports_ambiguous_algorithm(monkeypatch):
    class _Registry:
        def __init__(self, entries):
            self._entries = entries

        def discover(self):
            pass

        def entries(self):
            return self._entries

    registries = {
        "online": _Registry({}),
        "offline": _Registry({"iql": SimpleNamespace()}),
        "off2on": _Registry({"iql": SimpleNamespace()}),
    }
    monkeypatch.setattr(
        eval_checkpoint,
        "_registry_for_phase",
        lambda phase: registries[phase],
    )

    checkpoint = {"metadata": {"algorithm_class": "Off2OnIQL"}}

    with pytest.raises(SystemExit, match="ambiguous"):
        eval_checkpoint._resolve_phase_and_algorithm("auto", "auto", checkpoint)


_TRUNK_GRAPH = """
nodes:
  obs:     {type: input}
  encoder: {type: encoder, config: {normalize_obs: true}}
  trunk:   {type: mlp, net_arch: [8]}
  actor:   {type: actor}
  value:   {type: value}
edges:
  - {from: obs, to: encoder}
  - {from: encoder, to: trunk}
  - {from: trunk, to: actor, detach: true}
  - {from: trunk, to: value}
"""


@pytest.mark.parametrize("with_config_json", [False, True], ids=["metadata-only", "config-json"])
@pytest.mark.parametrize(
    "algorithm,cli",
    [
        pytest.param("ppo", ["--policy-graph", _TRUNK_GRAPH], id="ppo-inline"),
        pytest.param(
            "recurrent_ppo",
            ["--policy-graph", "builtin:recurrent", "--rnn-type", "gru", "--rnn-hidden-size", "8"],
            id="recurrent_ppo-builtin",
        ),
    ],
)
def test_policy_graph_round_trips_through_eval_checkpoint(
    tmp_path, algorithm, cli, with_config_json
):
    """The checkpoint records the canonical graph text; eval_checkpoint's
    overlay (the run's config.json first, then the checkpoint metadata)
    hands it back to the builder, which rebuilds a policy the checkpoint
    loads into."""
    import json

    from _decision_fakes import _FixedHoldVecEnv

    from rl_garden.common.checkpoint import load_checkpoint_file
    from rl_garden.common.effective_config import json_value
    from rl_garden.training.online import registry

    builder = eval_checkpoint._builder_from_module(
        eval_checkpoint._training_module("online", algorithm), algorithm
    )
    args = registry.parse_args(
        [algorithm, *cli, "--num-minibatches", "2", "--log-type", "none"]
    )
    agent = builder(args, _FixedHoldVecEnv(emit_decisions=False), None, None, None)
    run_dir = tmp_path / "run"
    path = agent.save(run_dir / "checkpoints" / "checkpoint.pt")
    checkpoint = load_checkpoint_file(path, map_location="cpu")
    canonical = checkpoint["metadata"]["hyperparameters"]["policy_graph"]
    assert canonical == agent.policy.graph_spec.to_yaml()
    assert not canonical.startswith("builtin:")

    if with_config_json:
        # What a run records: the args as given (a built-in name stays a name).
        (run_dir / "config.json").write_text(json.dumps({"inputs": json_value(args)}))
        eval_args = eval_checkpoint.EvalCheckpointArgs(checkpoint_path=str(path))
    else:
        eval_args = eval_checkpoint.EvalCheckpointArgs(
            checkpoint_path=str(path), config_path=str(tmp_path / "missing.json")
        )
    restored_args = eval_checkpoint._load_training_args(
        "online", algorithm, eval_args, checkpoint
    )
    assert restored_args.policy_graph == canonical

    restored = builder(restored_args, _FixedHoldVecEnv(emit_decisions=False), None, None, None)
    restored.load(path, load_replay_buffer=False)
    expected = agent.policy.state_dict()
    actual = restored.policy.state_dict()
    assert actual.keys() == expected.keys()
    assert any(key.startswith("graph_nodes.") for key in expected)
    for key, value in expected.items():
        assert torch.equal(actual[key].cpu(), value.cpu()), key


@pytest.mark.parametrize(
    "algorithm_class,expected",
    [
        ("RecurrentPPO", ("online", "recurrent_ppo")),
        ("TransformerPPO", ("online", "transformer_ppo")),
        ("SACFlow", ("online", "sac_flow")),
    ],
)
def test_resolve_auto_matches_registered_algorithm_class_names(algorithm_class, expected):
    checkpoint = {"metadata": {"algorithm_class": algorithm_class}}

    assert (
        eval_checkpoint._resolve_phase_and_algorithm("auto", "auto", checkpoint)
        == expected
    )


def test_resolve_auto_falls_back_to_name_table_without_registered_class(monkeypatch):
    class _Registry:
        def __init__(self, entries):
            self._entries = entries

        def discover(self):
            pass

        def entries(self):
            return self._entries

    registries = {
        "online": _Registry({}),
        "offline": _Registry({}),
        "off2on": _Registry({"iql": SimpleNamespace(algorithm_cls=None)}),
    }
    monkeypatch.setattr(
        eval_checkpoint, "_registry_for_phase", lambda phase: registries[phase]
    )

    checkpoint = {"metadata": {"algorithm_class": "Off2OnIQL"}}

    assert eval_checkpoint._resolve_phase_and_algorithm("auto", "auto", checkpoint) == (
        "off2on",
        "iql",
    )


def _save_run_config(run_dir, args, *, algorithm: str = "ppo"):
    from rl_garden.common.effective_config import (
        effective_config_json,
        resolve_active_environment,
        resolve_effective_config,
    )

    config = resolve_effective_config(
        args,
        training_phase="online",
        algorithm=algorithm,
        sources={},
        active_environment=resolve_active_environment(args),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(effective_config_json(config))


def _ppo_run_args(**backend_sections):
    from rl_garden.training.online.registry import registry

    registry.discover()
    args = registry.entries()["ppo"].args_cls()
    for name, value in backend_sections.items():
        setattr(args, name, value)
    return args


def _restored_args(tmp_path, run_args, **eval_kwargs):
    run_dir = tmp_path / "run"
    _save_run_config(run_dir, run_args)
    eval_args = eval_checkpoint.EvalCheckpointArgs(
        checkpoint_path=str(run_dir / "checkpoints" / "checkpoint.pt"), **eval_kwargs
    )
    args = eval_checkpoint._load_training_args("online", "ppo", eval_args, {"metadata": {}})
    return args, eval_args


def test_eval_rebuilds_gymnasium_backend_config_from_the_run(tmp_path):
    from rl_garden.common.env_args import GymnasiumConfig

    run_args = _ppo_run_args(
        env_backend="gymnasium",
        gymnasium=GymnasiumConfig(
            import_modules=("my_pkg.envs",),
            vector_env_factory="my_pkg.envs:make_vector_env",
            vector_env_returns_torch=True,
            device="cuda:0",
        ),
    )

    args, eval_args = _restored_args(tmp_path, run_args, env_backend="gymnasium")

    assert args.env_backend == "gymnasium"
    assert tuple(args.gymnasium.import_modules) == ("my_pkg.envs",)
    assert args.gymnasium.vector_env_factory == "my_pkg.envs:make_vector_env"
    assert args.gymnasium.vector_env_returns_torch is True
    assert args.gymnasium.device == "cuda:0"
    backend_config = eval_checkpoint._env_request(args, eval_args).backend_config
    assert backend_config.vector_env_factory == "my_pkg.envs:make_vector_env"


def test_eval_cli_backend_override_wins_only_for_changed_fields(tmp_path):
    from rl_garden.common.env_args import GymnasiumConfig

    run_args = _ppo_run_args(
        env_backend="gymnasium",
        gymnasium=GymnasiumConfig(
            import_modules=("my_pkg.envs",),
            vector_env_factory="my_pkg.envs:make_vector_env",
            device="cuda:0",
        ),
    )

    args, _ = _restored_args(
        tmp_path,
        run_args,
        env_backend="gymnasium",
        gymnasium=GymnasiumConfig(device="cuda:1"),
    )

    assert args.gymnasium.device == "cuda:1"
    assert args.gymnasium.vector_env_factory == "my_pkg.envs:make_vector_env"
    assert tuple(args.gymnasium.import_modules) == ("my_pkg.envs",)


def test_eval_restores_maniskill_run_config_and_applies_cli_override(tmp_path):
    from rl_garden.common.env_args import ManiSkillConfig

    run_args = _ppo_run_args(
        maniskill=ManiSkillConfig(sim_backend="physx_cpu", reward_mode="dense")
    )

    args, _ = _restored_args(tmp_path, run_args)
    assert args.maniskill.sim_backend == "physx_cpu"
    assert args.maniskill.reward_mode == "dense"

    args, _ = _restored_args(
        tmp_path, run_args, maniskill=ManiSkillConfig(render_backend="cpu")
    )
    assert args.maniskill.sim_backend == "physx_cpu"
    assert args.maniskill.reward_mode == "dense"
    assert args.maniskill.render_backend == "cpu"


def test_eval_env_count_need_not_divide_by_the_run_num_minibatches(
    tmp_path, monkeypatch
):
    """A RecurrentPPO run trained with ``num_minibatches=2`` evaluates with
    ``--num-eval-envs 3``: eval never updates, so the minibatch count must not
    constrain the eval env count."""
    from _decision_fakes import _FixedHoldVecEnv

    from rl_garden.common.checkpoint import load_checkpoint_file
    from rl_garden.training.online import registry

    builder = eval_checkpoint._builder_from_module(
        eval_checkpoint._training_module("online", "recurrent_ppo"), "recurrent_ppo"
    )
    args = registry.parse_args(
        [
            "recurrent_ppo",
            "--num-minibatches",
            "2",
            "--rnn-type",
            "gru",
            "--rnn-hidden-size",
            "8",
            "--log-type",
            "none",
        ]
    )
    trained = builder(args, _FixedHoldVecEnv(emit_decisions=False), None, None, None)
    path = trained.save(tmp_path / "run" / "checkpoints" / "checkpoint.pt")
    checkpoint = load_checkpoint_file(path, map_location="cpu")
    assert checkpoint["metadata"]["hyperparameters"]["num_minibatches"] == 2

    def make_three_env_pair(env_backend, request):
        del env_backend, request
        return (
            _FixedHoldVecEnv(emit_decisions=False, holds=(1, 2, 3)),
            _FixedHoldVecEnv(emit_decisions=False, holds=(1, 2, 3)),
        )

    monkeypatch.setattr(eval_checkpoint, "make_training_envs", make_three_env_pair)

    result = eval_checkpoint.evaluate_checkpoint(
        eval_checkpoint.EvalCheckpointArgs(
            checkpoint_path=str(path),
            config_path=str(tmp_path / "missing.json"),
            num_eval_envs=3,
            num_eval_episodes=2,
            max_eval_steps=20,
            device="cpu",
        )
    )

    assert result["algorithm"] == "recurrent_ppo"
    assert result["num_eval_envs"] == 3
    assert result["metrics"]["episodes_completed"] == 2.0


def _gymnasium_run_args():
    from rl_garden.common.env_args import GymnasiumConfig
    from rl_garden.observations import ObservationConfig

    run_args = _ppo_run_args(
        env_backend="gymnasium",
        gymnasium=GymnasiumConfig(vector_env_factory="my_pkg.envs:make_vector_env"),
    )
    run_args.env_id = "MyToy-v0"
    run_args.control_mode = "joint_torque"
    run_args.obs = ObservationConfig(extra_state=("goal",))
    return run_args


def test_eval_keeps_the_run_environment_when_the_cli_leaves_it_out(tmp_path):
    """No ``--env-backend``/``--env-id``/``--obs.*``/``--control-mode`` on the
    eval CLI: a gymnasium run is rebuilt on its own backend and env."""
    args, eval_args = _restored_args(tmp_path, _gymnasium_run_args())

    assert eval_args.env_backend is None and eval_args.env_id is None
    assert args.env_backend == "gymnasium"
    assert args.env_id == "MyToy-v0"
    assert args.control_mode == "joint_torque"
    assert args.obs.extra_state == ("goal",)
    request = eval_checkpoint._env_request(args, eval_args)
    assert request.env_id == "MyToy-v0"
    assert request.backend_config.vector_env_factory == "my_pkg.envs:make_vector_env"


def test_eval_cli_environment_fields_override_the_run_when_passed(tmp_path):
    import tyro

    from rl_garden.observations import ObservationConfig

    run_dir = tmp_path / "run"
    _save_run_config(run_dir, _gymnasium_run_args())
    eval_args = tyro.cli(
        eval_checkpoint.EvalCheckpointArgs,
        args=[
            "--checkpoint-path",
            str(run_dir / "checkpoints" / "checkpoint.pt"),
            "--env-id",
            "OtherToy-v0",
            "--control-mode",
            "pd_joint_delta_pos",
            "--obs.rgb",
            "cam0",
        ],
    )
    args = eval_checkpoint._load_training_args("online", "ppo", eval_args, {"metadata": {}})
    assert args.env_id == "OtherToy-v0"
    # A value equal to the training default still overrides the run's value.
    assert args.control_mode == "pd_joint_delta_pos"
    assert args.obs == ObservationConfig(rgb=("cam0",))
    assert args.env_backend == "gymnasium"

    args, _ = _restored_args(tmp_path, _gymnasium_run_args(), env_backend="maniskill")
    assert args.env_backend == "maniskill"
    assert args.obs.extra_state == ("goal",)


def test_eval_builds_the_agent_with_the_registry_entry_builder(monkeypatch):
    from rl_garden.training.online.entrypoints import ppo as ppo_module

    def declared_builder(*args):
        raise AssertionError("not called")

    class _Registry:
        def entries(self):
            return {
                "declared": SimpleNamespace(build_agent=declared_builder),
                "ppo": SimpleNamespace(build_agent=None),
            }

    monkeypatch.setattr(eval_checkpoint, "_registry_for_phase", lambda phase: _Registry())

    assert eval_checkpoint._registered_builder("online", "declared") is declared_builder
    # Without a declared builder, the entry's module in the phase's
    # entrypoints package supplies it.
    assert eval_checkpoint._registered_builder("online", "ppo") is ppo_module.build_ppo
