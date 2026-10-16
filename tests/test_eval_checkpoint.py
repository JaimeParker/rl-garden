from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "eval_checkpoint.py"
    spec = importlib.util.spec_from_file_location("eval_checkpoint", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_checkpoint = _load_module()


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

    from rl_garden.training.off2on._registry import registry as off2on_registry

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
    from rl_garden.training.off2on._registry import registry as off2on_registry

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
    from rl_garden.training.off2on._registry import registry as off2on_registry

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
