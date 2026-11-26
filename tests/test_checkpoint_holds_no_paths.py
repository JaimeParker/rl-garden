"""A checkpoint holds no absolute path and no command line, and everything that
reads it still works: evaluation, resume, off2on stage inference, checkpoints
saved with paths."""
from __future__ import annotations

import collections
import dataclasses
import json
import re
import shutil
from pathlib import Path

import pytest
import test_checkpoint_eval_is_self_contained as self_contained
import test_entrypoint_builds as builds
import torch
import yaml
from gymnasium import spaces

from rl_garden.algorithms import Off2OnIQL
from rl_garden.common.checkpoint import (
    REMOVED_PATH,
    checkpoint_dict,
    holds_removed_path,
    validate_checkpoint_metadata,
    without_local_paths,
)
from rl_garden.common.effective_config import ConfigError, reject_unrecorded_import_path
from rl_garden.common.env_args import GymnasiumConfig
from rl_garden.encoders.resnet import ResNetEncoder
from rl_garden.envs import backend_registry
from rl_garden.training import evaluation
from rl_garden.training.evaluation import EvalCheckpointArgs, evaluate_checkpoint

shared_inputs = builds.shared_inputs
cpu_only_and_restored_rng = builds.cpu_only_and_restored_rng

GRAPH = """\
edges:
- {from: obs, to: encoder}
- {from: encoder, to: actor}
- {from: encoder, to: value}
nodes:
  obs: {type: input}
  encoder: {type: encoder, config: {backbone: resnet10, features_dim: 8, pretrained_weights: WEIGHTS}}
  actor: {type: actor}
  value: {type: value}
"""


# --- what counts as a path ----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "/data/set.h5", "/", "//host/share", "~", "~/runs", "~someone/runs", "file:///data/x",
        "file:/data/x", "FILE:///data/x", "C:\\Users\\me", "d:/data", "\\\\server\\share\\x",
        "checkpoint:/data/base.pt?algo=bc", "act:~/base.pt", "host:/data/x", "a.h5,/data/b.h5",
        "--flag=/data/x", "--flag /data/x", "'/data/x'", "lmdb:///data/db", "with spaces /my dir/x",
        '{"model": "/data/scene.xml", "gain": 2}', '{"model":"C:\\\\a"}', "see https://a.b/c and /data/x",
        Path("/data/x"), '{"root": "/"}', "--root=//server/share", '{"asset": "/!assets/model.xml"}',
    ],
)
def test_removed(text):
    assert without_local_paths(text) == REMOVED_PATH


@pytest.mark.parametrize(
    "text",
    [
        "", "runs", "runs/a/checkpoints", "./a/b", "../a", "a/b.pt", "https://host/a/b?x=/y",
        "s3://bucket/key", "hf://org/name", "gs://b/a", "org/model-name", "resnet10-imagenet",
        "pkg.mod:attr", "pkg.mod:Cls.method", "pixels=rgb_front", "cuda:0", "PickCube-v1",
        "mjlab/Task-v0", "and/or", "a / b", "<env>", "~5", "x ~ y", "{}", '{"gain": 0.5}', "argv",
        "torch.nn.modules.linear.Linear", "auto_0.5", "1/2", "builtin:mlp", "a: b/c",
        "<unrecorded: <function <lambda> at 0x7f>>", REMOVED_PATH,
        yaml.safe_dump(yaml.safe_load(GRAPH.replace("WEIGHTS", "resnet10-imagenet"))),
    ],
)
def test_kept(text):
    assert without_local_paths(text) is text


def test_containers_are_copied_only_where_something_is_removed():
    tensor = torch.zeros(2)
    weights = collections.OrderedDict(w=tensor)
    weights._metadata = {"": {"version": 1}}
    tree = {
        "weights": weights,
        "nested": [{"a": ("x", "/data/x")}, {"b": tensor}],
        "/data/key": 1,
        "runtime": {"argv": ["train.py", "--x", "1"], "python": "3.10"},
        "none": None,
    }
    before = json.dumps(tree, default=str)
    cleaned = without_local_paths(tree)
    assert json.dumps(tree, default=str) == before
    assert cleaned["weights"] is weights
    assert cleaned["nested"] == [{"a": ("x", REMOVED_PATH)}, {"b": tensor}]
    assert cleaned["nested"][1]["b"] is tensor
    assert cleaned["runtime"] == {"python": "3.10"}
    assert cleaned["none"] is None
    assert "/data/key" not in cleaned and 1 in cleaned.values()
    weights["source"] = "/data/x"
    cleaned = without_local_paths(weights)
    assert type(cleaned) is collections.OrderedDict
    assert cleaned._metadata == weights._metadata and cleaned["w"] is tensor
    assert without_local_paths(cleaned) is cleaned


@dataclasses.dataclass(frozen=True)
class _Record:
    path: str
    weights: torch.Tensor


def test_dataclasses_bytes_and_keys():
    tensor = torch.ones(1)
    record = _Record("/data/x", tensor)
    cleaned = without_local_paths({"extra": record, "raw": b"/data/x", "fine": b"abc"})
    assert cleaned["extra"] == _Record(REMOVED_PATH, tensor) and record.path == "/data/x"
    assert cleaned["extra"].weights is tensor
    assert cleaned["raw"] == REMOVED_PATH.encode() and cleaned["fine"] == b"abc"
    assert holds_removed_path(cleaned["extra"]) and holds_removed_path(cleaned["raw"])
    keys = without_local_paths({Path("/data/a"): 1, ("x", "/data/b"): 2, ("x", "y"): 3})
    assert sorted(map(str, keys)) == sorted([f"{REMOVED_PATH} 0", f"{REMOVED_PATH} 1", "('x', 'y')"])


def test_renamed_keys_do_not_collide():
    cleaned = without_local_paths({"/private/key": "first", f"{REMOVED_PATH} 0": "second"})
    assert sorted(cleaned.values()) == ["first", "second"]


def test_parameter_names_are_kept():
    """Loading matches them exactly; one that looks like a path stays in the file."""
    weights = torch.nn.ModuleDict({"a": torch.nn.Linear(1, 1)}).state_dict()
    weights["/odd.weight"] = torch.zeros(1)
    assert without_local_paths(weights) is weights


def test_an_unrecorded_import_path_stays_recognizable():
    cleaned = without_local_paths("<unrecorded: Factory(path='/data/x')>")
    assert "/data" not in cleaned
    with pytest.raises(ConfigError):
        reject_unrecorded_import_path("actor_factory", cleaned)


# --- real checkpoints ---------------------------------------------------------

_LOOKS_ABSOLUTE = re.compile(r"""(^|[\s=:,"'(])(/|~/|file:|[A-Za-z]:\\)(?=[^\s/])""")


def _strings(value, where=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{where}.<key>", key
            yield from _strings(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            yield from _strings(item, f"{where}[{index}]")
    elif isinstance(value, str):
        yield where, value


def assert_holds_no_paths(checkpoint: Path, *roots: Path) -> None:
    contents = torch.load(checkpoint, map_location="cpu")
    strings = [(where, text) for where, text in _strings(contents) if isinstance(text, str)]
    assert not [item for item in strings if _LOOKS_ABSOLUTE.search(item[1])]
    assert "argv" not in contents["metadata"].get("run_config", {}).get("runtime", {})
    raw = checkpoint.read_bytes()
    for root in roots:
        assert str(root).encode() not in raw


def _weights_file(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    encoder = ResNetEncoder(**self_contained._resnet_kwargs())
    torch.save(
        {key: value for key, value in encoder.state_dict().items() if key.startswith(("stem", "blocks"))},
        directory / "resnet.pt",
    )
    return str(directory / "resnet")


SCANNED = [
    ("online", "sac"), ("online", "ppo"), ("online", "rlpd"), ("online", "jsrl"),
    ("online", "supe"), ("offline", "iql"), ("off2on", "wsrl"),
]
SCANNED_SLOW = [
    ("online", "ncrl"), ("online", "dagger"), ("online", "policy_distillation"),
    ("offline", "opal"), ("offline", "diffusion_bc"), ("offline", "consistency_distill_bc"),
    ("offline", "rssm_pretrain"), ("offline", "tdmpc2_multitask"), ("offline", "pwm"),
    ("offline", "bppo"),
]


def _train(phase, name, shared, tmp_path, *extra, **values) -> Path:
    """The checkpoint of a run given absolute paths wherever it takes one."""
    inputs = builds.inputs_for(phase, name, shared, tmp_path)
    inputs.update(self_contained.OPTIONAL_INPUTS.get((phase, name), lambda s: {})(shared))
    argv = builds.build_argv(
        phase, name, tmp_path / "runs", with_eval=(phase, name) in builds.EVAL_ONLY,
        inputs={
            **self_contained.SMALL_TRAINING, **builds.OVERRIDES.get((phase, name), {}),
            **self_contained.TRAINING_OVERRIDES.get((phase, name), {}), **inputs,
            "eval_output_dir": str(tmp_path / "videos"), **values,
        },
    )
    result = builds.run_entry(builds.PHASES[phase][0], [*argv, *extra])
    assert result.returncode == 0, f"{result.stderr[-1500:]!r} {result.stdout[-500:]!r}"
    found = sorted((tmp_path / "runs").glob("**/checkpoints/*.pt"))
    return next((c for c in found if c.name == "final.pt"), found[-1])


@pytest.mark.parametrize(
    "phase,name",
    [*(pytest.param(*key, id="-".join(key)) for key in SCANNED)]
    + [pytest.param(*key, id="-".join(key), marks=pytest.mark.slow) for key in SCANNED_SLOW],
)
def test_checkpoints_of_real_runs(phase, name, tmp_path, tmp_path_factory, shared_inputs):
    preset = tmp_path / "preset.yaml"
    preset.write_text("seed: 3\n")
    values = {}
    if (phase, name) == ("online", "ncrl"):
        artifacts = self_contained.make_pretrain_artifacts(tmp_path / "ncrl")
        values = {
            "pretrained_world_model": str(artifacts["checkpoint"]),
            "retrieval_db": str(artifacts["retrieval_db"]),
            "prior_dataset_dir": str(artifacts["dataset_dir"]),
        }
    if (phase, name) == ("online", "sac"):
        values = {"mmap_dir": str(tmp_path / "mmap")}
    if (phase, name) == ("offline", "pwm"):
        values = {"horizon": "2"}
    checkpoint = _train(
        phase, name, shared_inputs, tmp_path, "--config", str(preset), **values,
    )
    for path in checkpoint.parent.glob("*.pt"):
        if not path.name.startswith(("replay_buffer", "offline_buffer", "demo_buffer")):
            assert_holds_no_paths(path, tmp_path_factory.getbasetemp(), Path.home())

    # Only the checkpoint changes, and nothing else in it.
    config = json.loads((checkpoint.parent.parent / "config.json").read_text())
    assert config["runtime"]["argv"]
    assert config["inputs"]["log_dir"] == str(tmp_path / "runs")
    if (phase, name) not in self_contained.EVALUABLE:
        return
    run_config = torch.load(checkpoint)["metadata"]["run_config"]
    assert run_config == without_local_paths(config)
    assert run_config["inputs"]["log_dir"] == REMOVED_PATH
    assert run_config["inputs"]["seed"] == 3


def test_embedded_records_hold_no_paths(tmp_path, shared_inputs):
    """A guide whose own checkpoint was saved with paths."""
    guide = tmp_path / "guide.pt"
    contents = torch.load(shared_inputs.iql_guide_checkpoint)
    contents["metadata"]["hyperparameters"]["encoder_config"] = {"pretrained_weights": "/data/w"}
    torch.save(contents, guide)
    checkpoint = _train("online", "jsrl", shared_inputs, tmp_path, guide_checkpoint=str(guide))
    assert_holds_no_paths(checkpoint, tmp_path)
    record = torch.load(checkpoint)["state"]["extra"]["guide"]
    assert record["source"] == REMOVED_PATH
    assert record["hyperparameters"]["encoder_config"] == {"pretrained_weights": REMOVED_PATH}
    # Resumed without the guide file, and saved again.
    guide.unlink()
    again = _train(
        "online", "jsrl", shared_inputs, tmp_path / "again", "--load-checkpoint", str(checkpoint),
        guide_checkpoint=str(guide), total_timesteps="48",
    )
    assert_holds_no_paths(again, tmp_path)
    inputs = torch.load(again)["metadata"]["run_config"]["inputs"]
    assert inputs["load_checkpoint"] == REMOVED_PATH


# --- evaluation ---------------------------------------------------------------


def _lone(checkpoint: Path, directory: Path) -> Path:
    directory.joinpath("checkpoints").mkdir(parents=True)
    return Path(shutil.copy(checkpoint, directory / "checkpoints"))


def _evaluate(checkpoint: Path, **kwargs):
    for name, value in kwargs.pop("robomimic", {}).items():
        kwargs.setdefault("robomimic", EvalCheckpointArgs().robomimic)
        setattr(kwargs["robomimic"], name, value)
    return evaluate_checkpoint(
        EvalCheckpointArgs(
            checkpoint_path=str(checkpoint), num_eval_envs=1, num_eval_episodes=1,
            max_eval_steps=50, device="cpu", **kwargs,
        )
    )


@pytest.fixture
def asset_backend(monkeypatch):
    """A ``robomimic`` whose environment reads the file ``dataset_path`` names
    and, like the real one, silently is another environment without it. Returns
    what each environment was made with."""
    made = []
    gymnasium = backend_registry._get_backend("gymnasium")

    def inner(req):
        return dataclasses.replace(
            req, backend_config=GymnasiumConfig(import_modules=(builds.__name__,))
        )

    class AssetBackend(backend_registry.EnvBackend):
        api_version = 2
        config_field = "robomimic"

        @classmethod
        def resolve_config(cls, req, *, is_eval):
            return gymnasium.resolve_config(inner(req), is_eval=is_eval)

        @classmethod
        def make_train_env(cls, req):
            return gymnasium.make_train_env(inner(req))

        @classmethod
        def make_eval_env(cls, req):
            path = req.backend_config.dataset_path
            made.append(None if path is None else Path(path).read_text())
            return gymnasium.make_eval_env(inner(req))

    monkeypatch.setitem(backend_registry._REGISTRY, "robomimic", AssetBackend)
    return made


def test_a_backend_path_comes_from_the_flag_or_the_runs_config_or_is_an_error(
    asset_backend, tmp_path, shared_inputs, monkeypatch
):
    for name in ("gymnasium.device", "gymnasium.import_modules"):
        monkeypatch.delitem(builds.SMALL_VALUES, name)
    asset = tmp_path / "assets" / "scene.hdf5"
    asset.parent.mkdir()
    asset.write_text("original")
    (tmp_path / "other.hdf5").write_text("moved")
    checkpoint = _train(
        "online", "ppo", shared_inputs, tmp_path, env_backend="robomimic",
        **{"robomimic.dataset_path": str(asset), "robomimic.horizon": "9"},
    )
    assert_holds_no_paths(checkpoint, tmp_path)
    lone = _lone(checkpoint, tmp_path / "lone")

    # config.json of the run, beside the checkpoint.
    _evaluate(checkpoint)
    # Moved: the flag, or the config given explicitly.
    _evaluate(lone, robomimic={"dataset_path": str(tmp_path / "other.hdf5")})
    _evaluate(lone, config_path=str(checkpoint.parent.parent / "config.json"))
    assert asset_backend == ["original", "moved", "original"]

    # Neither: named, and no environment is made.
    flags = r"robomimic\.dataset_path \(--robomimic\.dataset-path\).*--config-path"
    with pytest.raises(SystemExit, match=flags):
        _evaluate(lone)
    # Beside it the configuration of another run, which differs only in its paths.
    config = json.loads((checkpoint.parent.parent / "config.json").read_text())
    config["active_environment"]["config"]["dataset_path"] = str(tmp_path / "other.hdf5")
    (tmp_path / "lone" / "config.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit, match=flags):
        _evaluate(lone)
    (tmp_path / "lone" / "config.json").write_text("not json")
    with pytest.raises(SystemExit, match=flags):
        _evaluate(lone)
    assert len(asset_backend) == 3

    # Not needed by another backend.
    _evaluate(lone, env_backend="gymnasium", gymnasium=GymnasiumConfig(import_modules=(builds.__name__,)))


def test_environment_kwargs_containing_a_path(tmp_path, shared_inputs):
    kwargs = '{"g":9.0}'
    checkpoint = _train(
        "online", "ppo", shared_inputs, tmp_path, **{"gymnasium.env_kwargs_json": kwargs}
    )

    def with_asset(contents):
        config = contents["metadata"]["run_config"]["active_environment"]["config"]
        config["env_kwargs_json"] = without_local_paths('{"g": 9.0, "xml": "/data/x.xml"}')

    self_contained._edit_checkpoint(lone := _lone(checkpoint, tmp_path / "lone"), with_asset)
    with pytest.raises(SystemExit, match=r"--gymnasium\.env-kwargs-json"):
        _evaluate(lone)
    _evaluate(
        lone,
        gymnasium=GymnasiumConfig(env_kwargs_json=kwargs, import_modules=(builds.__name__,)),
    )
    _evaluate(checkpoint)


@pytest.mark.parametrize(
    "name,extra_key,argument",
    [("jsrl", "guide", "guide_checkpoint"), ("supe", "opal_vae", "opal_checkpoint")],
)
def test_a_network_file_is_asked_for_only_when_it_is_needed(
    name, extra_key, argument, tmp_path, shared_inputs, caplog
):
    checkpoint = _train("online", name, shared_inputs, tmp_path)
    lone = _lone(checkpoint, tmp_path / "lone")
    _evaluate(lone)

    self_contained._edit_checkpoint(lone, lambda c: c["state"]["extra"].pop(extra_key))
    flag = f"--{argument.replace('_', '-')}"
    with pytest.raises(FileNotFoundError, match=flag) as error:
        _evaluate(lone)
    assert REMOVED_PATH not in str(error.value)
    assert f"{argument} ({flag})" in caplog.text
    _evaluate(lone, **{argument: builds.inputs_for("online", name, shared_inputs, tmp_path)[argument]})
    # Or from the run's own config.json.
    shutil.copy(checkpoint.parent.parent / "config.json", tmp_path / "lone")
    _evaluate(lone)


def test_a_run_with_training_only_paths_needs_none_of_them(tmp_path, shared_inputs, caplog):
    checkpoint = _train("offline", "iql", shared_inputs, tmp_path)
    _evaluate(_lone(checkpoint, tmp_path / "lone"))
    assert REMOVED_PATH not in caplog.text


def test_pretrained_weights_of_a_policy_graph(tmp_path, shared_inputs):
    weights = _weights_file(tmp_path / "weights")
    image = {"env_id": builds.IMAGE_ENV_ID, "obs.rgb": "front"}
    graph = tmp_path / "graph.yaml"
    graph.write_text(GRAPH.replace("WEIGHTS", weights))
    checkpoint = _train(
        "online", "ppo", shared_inputs, tmp_path / "graph", policy_graph=str(graph), **image
    )
    assert_holds_no_paths(checkpoint, tmp_path)
    text = torch.load(checkpoint)["metadata"]["hyperparameters"]["policy_graph"]
    assert yaml.safe_load(text)["nodes"]["encoder"]["config"]["pretrained_weights"] == REMOVED_PATH
    assert yaml.safe_load(text)["nodes"]["encoder"]["config"]["backbone"] == "resnet10"

    # The same graph resumes it, another does not.
    _train(
        "online", "ppo", shared_inputs, tmp_path / "resumed", "--load-checkpoint", str(checkpoint),
        policy_graph=str(graph), total_timesteps="64", **image,
    )
    other = tmp_path / "other.yaml"
    other.write_text(GRAPH.replace("WEIGHTS", weights).replace("features_dim: 8", "features_dim: 9"))
    with pytest.raises(ValueError, match="policy_graph mismatch"):
        _train(
            "online", "ppo", shared_inputs, tmp_path / "other", "--load-checkpoint",
            str(checkpoint), policy_graph=str(other), total_timesteps="64", **image,
        )
    shutil.rmtree(tmp_path / "weights")
    _evaluate(_lone(checkpoint, tmp_path / "lone"))
    _evaluate(checkpoint)


def test_pretrained_weights_are_not_needed(tmp_path, shared_inputs):
    weights = _weights_file(tmp_path / "weights")
    checkpoint = _train(
        "online", "ppo", shared_inputs, tmp_path, env_id=builds.IMAGE_ENV_ID, **{
            "obs.rgb": "front", "encoder.backbone": "resnet10",
            "encoder.pretrained_weights": weights, "encoder.freeze_resnet_encoder": True,
        },
    )
    assert_holds_no_paths(checkpoint, tmp_path)
    shutil.rmtree(tmp_path / "weights")
    _evaluate(_lone(checkpoint, tmp_path / "lone"))
    _evaluate(checkpoint)


def test_a_complete_policy_graph_is_kept():
    """A node of another package may need the path its run's config.json has."""
    complete = yaml.safe_dump({"nodes": {"a": {"type": "x", "file": "/data/x"}}})
    stored = without_local_paths(complete)
    args = type("Args", (), {"policy_graph": complete})()
    evaluation._set_existing_attr(args, "policy_graph", stored)
    assert args.policy_graph == complete
    args.policy_graph = "builtin:mlp"
    evaluation._set_existing_attr(args, "policy_graph", stored)
    assert args.policy_graph == stored


# --- other readers ------------------------------------------------------------


def test_off2on_stage_is_not_inferred_from_a_run_that_loaded_a_checkpoint():
    def inferred(load_checkpoint):
        inputs = {
            "online_replay_mode": "mixed", "offline_data_ratio": 0.5,
            "load_checkpoint": load_checkpoint,
        }
        checkpoint = without_local_paths({"metadata": {"run_config": {"inputs": inputs}}})
        return Off2OnIQL._infer_online_stage(None, checkpoint, None)

    assert inferred(None) == {"replay_mode": "mixed", "offline_data_ratio": 0.5}
    assert inferred("/data/final.pt") is None
    assert inferred("runs/final.pt") is None


def test_policy_graphs_are_compared_without_their_paths():
    complete, stored = "nodes:\n  file: /data/x\n", without_local_paths("nodes:\n  file: /data/x\n")

    box = spaces.Box(-1, 1, (1,))

    def validate(saved, current):
        checkpoint = checkpoint_dict(
            algorithm_class="PPO", global_step=0, global_update=0, observation_space=box,
            action_space=box, hyperparameters={"policy_graph": saved}, state={},
        )
        validate_checkpoint_metadata(
            checkpoint, algorithm_class="PPO", compatible_algorithms=("PPO",),
            observation_space=box, action_space=box, strict=True, policy_graph=current,
        )

    for saved in (complete, stored):
        for current in (complete, stored):
            validate(saved, current)
    with pytest.raises(ValueError, match="policy_graph mismatch"):
        validate(stored, "nodes:\n  file: relative\n")


# --- checkpoints saved with their paths ---------------------------------------


@pytest.mark.parametrize("name", ["jsrl", "rlpd"])
def test_a_checkpoint_saved_with_paths_still_evaluates_and_resumes(name, tmp_path, shared_inputs):
    checkpoint = _train("online", name, shared_inputs, tmp_path)
    config = json.loads((checkpoint.parent.parent / "config.json").read_text())
    inputs = builds.inputs_for("online", name, shared_inputs, tmp_path)

    def with_paths(contents):
        metadata = contents["metadata"]
        metadata["run_config"] = config
        del metadata["run_config_digest"]
        if name == "jsrl":
            metadata["hyperparameters"]["guide_checkpoint"] = inputs["guide_checkpoint"]
            contents["state"]["extra"]["guide"]["source"] = inputs["guide_checkpoint"]

    old = _lone(checkpoint, tmp_path / "old")
    self_contained._edit_checkpoint(old, with_paths)
    _evaluate(old)
    if name == "jsrl":
        # Its recorded path is read when it embeds no guide.
        bare = _lone(old, tmp_path / "bare")
        self_contained._edit_checkpoint(bare, lambda c: c["state"]["extra"].pop("guide"))
        _evaluate(bare)
    again = _train(
        "online", name, shared_inputs, tmp_path / "again", "--load-checkpoint", str(old),
        total_timesteps="48",
    )
    assert_holds_no_paths(again, tmp_path)
