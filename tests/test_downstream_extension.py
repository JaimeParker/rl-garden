"""The extension contract, end to end, from a package outside ``rl_garden``.

``tests/fixtures/downstream_extension/rlg_downstream_example`` is a downstream
package (see ``docs/guides/extending.md``): a ``RecurrentPPO`` subclass whose
policy builds a two-head actor, with an auxiliary loss and an extra metric,
registered with two extra CLI flags. Here it is registered on the real online
registry, trained through the registered run function, checkpointed and
evaluated with ``--algorithm auto``, all in-process on CPU.
"""

from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "downstream_extension"
PACKAGE = "rlg_downstream_example"
ALGORITHM = "aux_split_rppo"
RL_GARDEN_ROOT = Path(__file__).resolve().parents[1] / "rl_garden"
#: Seams whose contract is "build this module": an override supplies the
#: module and has nothing to extend through super().
BUILDER_SEAMS = {"_build_actor"}


@pytest.fixture
def online_registry(monkeypatch):
    """The online registry, restored after the test; the fixture package
    importable and imported afresh; CUDA hidden so every agent is on CPU."""
    from rl_garden.training.online import registry

    registry.discover()
    monkeypatch.setattr(registry, "_entries", dict(registry._entries))
    monkeypatch.syspath_prepend(str(FIXTURE_ROOT))
    for name in [name for name in sys.modules if name.split(".")[0] == PACKAGE]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    return registry


def _train_argv(run_root: Path) -> list[str]:
    return [
        ALGORITHM,
        "--env-backend",
        "gymnasium",
        "--env-id",
        "AuxDecisionToy-v0",
        "--gymnasium.vector-env-factory",
        f"{PACKAGE}.toy_env:make_aux_decision_toy_env",
        "--gymnasium.vector-env-returns-torch",
        "--gymnasium.device",
        "cpu",
        "--num-envs",
        "4",
        "--num-eval-envs",
        "2",
        "--num-steps",
        "16",
        "--num-minibatches",
        "2",
        "--update-epochs",
        "2",
        "--total-timesteps",
        "192",
        "--rnn-type",
        "gru",
        "--rnn-hidden-size",
        "16",
        "--decision-level",
        "--aux-weight",
        "0.2",
        "--aux-hidden",
        "8",
        "--eval-freq",
        "0",
        "--log-freq",
        "64",
        "--log-type",
        "none",
        "--log-dir",
        str(run_root),
        "--exp-name",
        "run",
        "--no-capture-video",
        "--seed",
        "1",
    ]


def test_downstream_algorithm_trains_checkpoints_and_evaluates(
    online_registry, tmp_path, monkeypatch, capsys
):
    from rl_garden.common.checkpoint import load_checkpoint_file
    from rl_garden.training.evaluation import EvalCheckpointArgs, evaluate_checkpoint

    importlib.import_module(f"{PACKAGE}.registration")
    entry = online_registry.entries()[ALGORITHM]

    with pytest.raises(SystemExit) as exit_info:
        online_registry.run_cli([ALGORITHM, "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--aux-weight" in help_text and "--aux-hidden" in help_text

    algorithm_cls = entry.algorithm_cls()
    update_metrics: list[dict[str, float]] = []
    train_scalars: list[str] = []
    train, log_rollout_metric = algorithm_cls.train, algorithm_cls._log_rollout_metric

    def recording_train(self):
        metrics = train(self)
        update_metrics.append(metrics)
        return metrics

    def recording_log_rollout_metric(self, key, value, step):
        train_scalars.append(key)
        log_rollout_metric(self, key, value, step)

    monkeypatch.setattr(algorithm_cls, "train", recording_train)
    monkeypatch.setattr(algorithm_cls, "_log_rollout_metric", recording_log_rollout_metric)

    command = online_registry.parse_command(_train_argv(tmp_path))
    assert (command.args.aux_weight, command.args.aux_hidden) == (0.2, 8)
    entry.run_fn(command.args)

    assert len(update_metrics) == 3
    for metrics in update_metrics:
        assert {"aux_logit_mean", "aux/bce", "policy_loss"} <= set(metrics)
    assert "aux_logit_mean" in train_scalars

    checkpoint_path = tmp_path / "run" / "checkpoints" / "final.pt"
    metadata = load_checkpoint_file(checkpoint_path, map_location="cpu")["metadata"]
    assert metadata["algorithm_class"] == "AuxSplitRecurrentPPO"
    assert metadata["hyperparameters"]["aux_weight"] == 0.2
    assert metadata["hyperparameters"]["aux_hidden"] == 8

    result = evaluate_checkpoint(
        EvalCheckpointArgs(
            checkpoint_path=str(checkpoint_path),
            algorithm="auto",
            strict=True,
            num_eval_envs=2,
            num_eval_episodes=3,
            device="cpu",
        )
    )

    assert (result["phase"], result["algorithm"]) == ("online", ALGORITHM)
    assert result["env_backend"] == "gymnasium"
    assert result["metrics"]["episodes_completed"] == 3.0


def test_downstream_algorithm_registers_through_the_entry_point_group(
    online_registry, monkeypatch
):
    from rl_garden.training import algorithm_registry

    declared = (FIXTURE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = declared.split('[project.entry-points."rlgarden.algorithms"]', 1)[1]
    name, module = re.search(r'^(\w+) = "([\w.]+)"$', section, re.M).groups()

    class _EntryPoint:
        def __init__(self):
            self.name, self.value = name, module

        def load(self):
            return importlib.import_module(module)

    def entry_points(*, group):
        assert group == algorithm_registry.ALGORITHM_ENTRY_POINT_GROUP
        return [_EntryPoint()]

    monkeypatch.setattr(algorithm_registry, "_ALGORITHM_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(algorithm_registry.importlib.metadata, "entry_points", entry_points)
    monkeypatch.setattr(online_registry, "_discovered", False)
    assert ALGORITHM not in online_registry.entries()

    command = online_registry.parse_command([ALGORITHM, "--aux-weight", "0.5"])

    assert command.algorithm == ALGORITHM
    assert command.args.aux_weight == 0.5
    assert type(command.args).__module__ == f"{PACKAGE}.registration"


def _function_bodies(tree: ast.AST) -> dict[str, list[str]]:
    """``{name: [ast dump of the body without its docstring]}`` of every
    function and method in ``tree``."""
    bodies: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            bodies.setdefault(node.name, []).append(ast.dump(ast.Module(body=body, type_ignores=[])))
    return bodies


def _calls_super(function: ast.FunctionDef) -> bool:
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super"
        for node in ast.walk(function)
    )


def test_the_downstream_package_copies_no_rl_garden_code(online_registry):
    """By construction: every override extends its parent through super()
    (a builder seam excepted), no function body equals one of rl-garden's,
    and nothing private is imported from rl_garden."""
    library: dict[str, set[str]] = {}
    for path in RL_GARDEN_ROOT.rglob("*.py"):
        for name, bodies in _function_bodies(ast.parse(path.read_text(encoding="utf-8"))).items():
            library.setdefault(name, set()).update(bodies)

    copied, replaced, private_imports = [], [], []
    for path in sorted((FIXTURE_ROOT / PACKAGE).glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, bodies in _function_bodies(tree).items():
            copied += [f"{path.name}:{name}" for body in bodies if body in library.get(name, ())]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("rl_garden"):
                private_imports += [alias.name for alias in node.names if alias.name.startswith("_")]
            if not isinstance(node, ast.ClassDef):
                continue
            cls = getattr(importlib.import_module(f"{PACKAGE}.{path.stem}"), node.name)
            library_bases = [base for base in cls.__mro__[1:] if base.__module__.startswith("rl_garden")]
            for method in node.body:
                overrides = isinstance(method, ast.FunctionDef) and any(
                    method.name in vars(base) for base in library_bases
                )
                if overrides and method.name not in BUILDER_SEAMS and not _calls_super(method):
                    replaced.append(f"{path.name}:{node.name}.{method.name}")

    assert copied == []
    assert replaced == []
    assert private_imports == []
