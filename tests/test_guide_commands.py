"""Validate published guide commands in the fast lane, without running training."""
from __future__ import annotations

import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import tyro

from rl_garden.training.evaluation import EvalCheckpointArgs

try:
    from tests._cli import run_entry
except ModuleNotFoundError as exc:
    if exc.name != "tests._cli":
        raise
    # Some environments install a top-level tests package that shadows this one.
    from _cli import run_entry

ROOT = Path(__file__).resolve().parents[1]
GUIDES = sorted((ROOT / "docs/guides").glob("*.md"))
ENTRIES = {
    "examples/train_online.py": "train_online",
    "examples/pretrain_offline.py": "pretrain_offline",
    "examples/train_off2on.py": "train_off2on",
    "examples/eval_checkpoint.py": "eval",
    "rl-garden-train-online": "train_online",
    "rl-garden-train-offline": "pretrain_offline",
    "rl-garden-train-off2on": "train_off2on",
    "rl-garden-eval": "eval",
}
START = re.compile(
    r"^\s*(?:[A-Za-z_][\w]*=(?:\"[^\"]*\"|'[^']*'|\S+)\s+)*"
    r"(?:(?:python(?:3)?\s+(?:-u\s+)?)?(?P<entry>"
    + "|".join(re.escape(name) for name in ENTRIES)
    + r"))(?=\s|$)(?P<args>.*)$"
)
PLACEHOLDER = re.compile(r"<[^>]+>|/path/to/[^\s]+")
# Exact (guide basename, argv tuple) -> explanation; stale entries fail below.
EXEMPTIONS: dict[tuple[str, tuple[str, ...]], str] = {}
EXTERNAL_PATHS = {
    ("docs/design/iql-overview.md", "configs/offline/iql/antmaze/umaze_v2.yaml"):
        "The comparison cites a config in the CORL repository.",
    ("docs/guides/ncrl.md", "scripts/finetune.sh"):
        "The surrounding text names the official NCRL repository's finetune script.",
    ("docs/real-robot.rst", "scripts/install.sh"):
        "This command installs the downstream real-world-rl package from its own root.",
}
# These fields are Optional[bool], so tyro requires an explicit value.
OPTIONAL_BOOLS = {
    "isaaclab.final-obs", "maniskill.fix-peg-pose", "maniskill.fix-box",
    "detach-encoder-on-actor",
}


@dataclass(frozen=True)
class Command:
    guide: str
    line: int
    entry: str
    argv: tuple[str, ...]
    error: str = ""

    @property
    def id(self) -> str:
        return f"{self.guide}:{self.line}"


def extract_commands(markdown: str, guide: str) -> list[Command]:
    """Read physical shell lines, retaining source locations and broken commands.

    Starting at the inner Python line also handles documented SSH/Docker wrappers;
    no shell text, environment assignment, pipe, or redirection is executed.
    """
    commands = []
    lines = markdown.splitlines()
    fence = None
    shell = False
    index = 0
    while index < len(lines):
        line = lines[index]
        marker = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if marker:
            if fence is None:
                fence = marker[1]
                shell = marker[2].strip() in {"bash", "sh"}
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence, shell = None, False
            index += 1
            continue
        match = START.match(line) if shell else None
        if not match:
            index += 1
            continue
        start = index + 1
        pieces = []
        part = match["args"]
        error = ""
        while True:
            continued = part.rstrip().endswith("\\")
            pieces.append(part.rstrip()[:-1] if continued else part)
            index += 1
            if not continued:
                following = index
                while following < len(lines) and not lines[following].strip():
                    following += 1
                if following < len(lines) and lines[following].lstrip().startswith("--"):
                    error = "option line separated from command by missing continuation"
                break
            if index >= len(lines) or not lines[index].strip():
                error = "blank line or EOF inside a continued command"
                break
            part = lines[index]
            if re.match(r"^\s*[`~]{3,}", part):
                error = "unterminated continuation before closing fence"
                break
            if part.lstrip().startswith(("'", '\\"', '"')):
                # Closing quote of the enclosing remote shell, not Python argv.
                break
        try:
            # Angle-bracket placeholders can contain spaces; preserve one argv value.
            joined = re.sub(r"<[^>]+>", lambda m: m[0].replace(" ", "_"), " ".join(pieces))
            argv = shlex.split(joined, comments=True)
            for stop, token in enumerate(argv):
                if token in {"|", "&&", ";", ">", "2>&1", "&"}:
                    argv = argv[:stop]
                    break
        except ValueError as exc:
            argv, error = [], str(exc)
        commands.append(Command(guide, start, ENTRIES[match["entry"]], tuple(argv), error))
    return commands


COMMANDS = [command for path in GUIDES for command in
            extract_commands(path.read_text(), path.name)]


def prepare_argv(argv: tuple[str, ...], tmp_path: Path) -> list[str]:
    result = []
    index = 0
    while index < len(argv):
        token = argv[index]
        flag, equals, value = token.partition("=")
        normalized = flag.replace("_", "-")
        if normalized in {"--print-config", "--dry-run", "--explain-param"}:
            index += 2 if normalized == "--explain-param" and not equals else 1
            continue
        if flag.startswith("--"):
            if not equals and index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                value = argv[index + 1]
                index += 1
            if value:
                config = normalized == "--config" or normalized.endswith(("-config", "-config-path"))
                runtime_file = normalized in {"--load-checkpoint", "--offline-dataset"} or normalized.endswith(
                    ("-checkpoint", "-checkpoint-path")
                )
                graph = normalized == "--policy-graph" and value.endswith((".yaml", ".yml"))
                if graph and not (ROOT / value).is_file():
                    # The CLI reads graph YAML at preflight, so an empty file is insufficient.
                    from rl_garden.networks.graph.builtin import mlp_actor_critic, recurrent_actor_critic

                    spec = (mlp_actor_critic() if argv[0] == "ppo" else
                            recurrent_actor_critic("gtrxl" if argv[0] == "transformer_ppo" else "gru"))
                    path = tmp_path / f"graph-{index}.yaml"
                    path.touch()
                    path.write_text(spec.to_yaml())
                    value = str(path)
                elif runtime_file or (config and PLACEHOLDER.search(value)):
                    path = tmp_path / f"input-{index}{'.yaml' if config else '.pt'}"
                    path.touch()
                    if config:
                        path.write_text("{}\n")
                    value = str(path)
                else:
                    replacement = "gymnasium" if normalized == "--env-backend" else "example"
                    value = PLACEHOLDER.sub(replacement, value)
                result.extend([flag, value])
            else:
                result.append(flag)
        else:
            result.append(PLACEHOLDER.sub("example", token))
        index += 1
    return result


@pytest.mark.parametrize("command", COMMANDS, ids=lambda command: command.id)
def test_guide_command(command: Command, tmp_path: Path) -> None:
    assert not command.error, f"{command.id}: {command.error}"
    exemption = EXEMPTIONS.get((command.guide, command.argv))
    if exemption:
        pytest.skip(exemption)
    argv = prepare_argv(command.argv, tmp_path)
    if command.entry == "eval":
        tyro.cli(EvalCheckpointArgs, args=argv)
    else:
        result = run_entry(command.entry, [*argv, "--print-config"], cwd=ROOT)
        assert result.returncode == 0, f"{command.id}\n{result.stderr}"
        assert json.loads(result.stdout)["status"] == "preflight"


def test_guide_command_coverage() -> None:
    assert len(GUIDES) >= 19
    assert len(COMMANDS) >= 40
    assert any(command.entry == "eval" for command in COMMANDS)
    actual = {(command.guide, command.argv) for command in COMMANDS}
    assert not EXEMPTIONS.keys() - actual, "stale command exemptions"
    assert all(EXEMPTIONS.values()), "every exemption needs a reason"
    scripts = (ROOT / "pyproject.toml").read_text()
    for name in ENTRIES:
        if name.startswith("rl-garden-"):
            assert re.search(rf"^{re.escape(name)}\s*=", scripts, re.MULTILINE)


@pytest.mark.parametrize("fence", ["bash", "sh"])
def test_extractor(fence: str) -> None:
    sample = (f"  ```{fence}\n  # comment\n"
              "  MUJOCO_GL=egl python -u examples/train_online.py sac \\\n"
              "    --env-id 'Pendulum-v1' \\\n"
              "    --num-envs 1 # trailing comment\n"
              "  rl-garden-eval --checkpoint-path /path/to/model.pt\n  ```\n")
    commands = extract_commands(sample, "sample.md")
    assert [(c.line, c.entry) for c in commands] == [(3, "train_online"), (6, "eval")]
    assert commands[0].argv == ("sac", "--env-id", "Pendulum-v1", "--num-envs", "1")
    assert not any(c.error for c in commands)


def test_broken_continuation() -> None:
    commands = extract_commands(
        "```bash\npython examples/train_online.py sac \\\n\n --num-envs 1\n```", "broken.md"
    )
    assert len(commands) == 1
    assert commands[0].error


def documentation_paths() -> list[Path]:
    # Published design pages come from the toctree: a remote checkout has no .git
    # to list, and untracked design notes are not published.
    design = ROOT / "docs/design"
    entries = re.findall(r"^   (?:.*<)?([\w./-]+?)>?$", (design / "index.rst").read_text(), re.MULTILINE)
    candidates = set(GUIDES) | {
        path for path in (design / f"{entry}.md" for entry in entries)
        if path.parent == design and path.is_file()
    }
    candidates.update((ROOT / "docs").rglob("*.rst"))
    return sorted(path for path in candidates if not any(
        part in {"_generated", "generated", "_build"} for part in path.relative_to(ROOT).parts
    ))


PATH_REFERENCE = re.compile(
    r"(?<![\w/.-])(?:\.{1,2}/)*(?P<path>"
    r"(?:scripts|configs|docs|examples|tools|rl_garden)/[^\s`\"'(),;|]+)"
)


def text_problems(path: Path) -> list[str]:
    problems = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        location = f"{path.relative_to(ROOT)}:{number}"
        for match in re.finditer(r"--([\w.-]+)(?:\s+|=)(True|False)\b", line, re.IGNORECASE):
            if match[1].replace("_", "-") not in OPTIONAL_BOOLS:
                problems.append(f"{location}: boolean value: {match[0]}")
        if path.parent == ROOT / "docs/guides" and ".agents/" in line:
            problems.append(f"{location}: private .agents/ reference")
        # Remove whole URL/external-repository tokens before looking for root paths.
        cleaned = re.sub(r"\S*(?:https?://|real_world/|real_world_rl/|3rd_party/)\S*", "", line)
        for match in PATH_REFERENCE.finditer(cleaned):
            reference = match["path"].rstrip(".:!]").split("#")[0].split(":")[0]
            if any(char in reference for char in "*?<>$[{") or reference.endswith("/..."):
                continue
            if (str(path.relative_to(ROOT)), reference) in EXTERNAL_PATHS:
                continue
            if not (ROOT / reference).exists():
                problems.append(f"{location}: missing path: {reference}")
    return problems


@pytest.mark.parametrize("path", documentation_paths(), ids=lambda p: str(p.relative_to(ROOT)))
def test_documentation_text(path: Path) -> None:
    problems = text_problems(path)
    assert not problems, "\n" + "\n".join(problems)


def test_external_path_exemptions_are_current() -> None:
    for (name, reference), reason in EXTERNAL_PATHS.items():
        assert reason
        assert reference in (ROOT / name).read_text(), f"stale exemption: {name}: {reference}"
        assert not (ROOT / reference).exists(), f"no longer external-only: {reference}"


@pytest.mark.parametrize("spelling", ["-", "_"])
@pytest.mark.parametrize("equals", [False, True])
def test_argument_substitutions(tmp_path: Path, spelling: str, equals: bool) -> None:
    def option(name: str, value: str) -> list[str]:
        flag = "--" + name.replace("-", spelling)
        return [f"{flag}={value}"] if equals else [flag, value]

    original = ("sac", *option("load-checkpoint", "/path/to/model.pt"),
                *option("offline-dataset", "runs/data.h5"),
                *option("config", "/path/to/config.yaml"),
                *option("checkpoint-freq", "10"),
                *option("explain-param", "gamma"), "--dry-run", "--print-config")
    prepared = prepare_argv(original, tmp_path)
    assert prepared[0] == "sac"
    for index in (2, 4, 6):
        assert Path(prepared[index]).is_file()
    assert prepared[-1] == "10"
    assert len(prepared) == 9


def test_extractor_remote_wrapper_and_placeholders() -> None:
    sample = "\n".join([
        "```sh", "ssh host", "  python examples/train_online.py sac " + chr(92),
        "    --env-id <task name> " + chr(92), "  ' 2>&1 | tee log", "```",
    ])
    commands = extract_commands(sample, "remote.md")
    assert len(commands) == 1
    assert commands[0].argv == ("sac", "--env-id", "<task_name>")
    assert not commands[0].error


def test_missing_backslash_is_not_silently_accepted() -> None:
    commands = extract_commands(
        "```bash\npython examples/train_online.py sac\n\n  --num-envs 1\n```", "broken.md"
    )
    assert len(commands) == 1
    assert commands[0].error


def test_text_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    guide = tmp_path / "docs/guides/sample.md"
    guide.parent.mkdir(parents=True)
    script = tmp_path / "scripts/present.py"
    script.parent.mkdir()
    script.touch()
    guide.write_text(
        "`--flag True`, `--lower.case false` and `--isaaclab.final_obs False`\n"
        "`scripts/present.py::main` and `scripts/missing.py`\n"
        "`scripts/*.sh` `scripts/test[ab].sh` `scripts/<name>.py`\n"
        "https://example.org/scripts/external.py `real_world/scripts/external.py`\n"
        "`.agents/private.md`\n"
    )
    problems = text_problems(guide)
    assert len(problems) == 4, problems
    assert ":1: boolean value: --flag True" in problems[0]
    assert ":1: boolean value: --lower.case false" in problems[1]
    assert ":2: missing path: scripts/missing.py" in problems[2]
    assert ":5: private .agents/ reference" in problems[3]
