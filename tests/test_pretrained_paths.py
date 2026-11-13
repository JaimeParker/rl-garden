"""Externally-sourced pretrained weights are looked up outside the repository tree."""

from __future__ import annotations

from pathlib import Path

from rl_garden.common import pretrained_paths
from rl_garden.encoders.resnet import ResNetEncoder
from rl_garden.models.act.config import act_pretrained_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env_var_overrides_every_kind(tmp_path, monkeypatch):
    monkeypatch.setenv(pretrained_paths.PRETRAINED_DIR_ENV, str(tmp_path))
    assert pretrained_paths.pretrained_dir("resnet") == tmp_path.resolve()
    assert ResNetEncoder.pretrained_dir() == tmp_path.resolve()
    assert act_pretrained_dir() == tmp_path.resolve()


def test_default_follows_xdg_cache_home(tmp_path, monkeypatch):
    monkeypatch.delenv(pretrained_paths.PRETRAINED_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    root = tmp_path / "rl_garden" / "pretrained"
    assert ResNetEncoder.pretrained_dir() == root / "resnet"
    assert act_pretrained_dir() == root / "act"


def test_default_is_under_home_cache_and_never_the_repo(monkeypatch):
    monkeypatch.delenv(pretrained_paths.PRETRAINED_DIR_ENV, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    default = pretrained_paths.default_pretrained_root()
    assert default == Path.home() / ".cache" / "rl_garden" / "pretrained"
    assert REPO_ROOT not in ResNetEncoder.pretrained_dir().parents
    assert REPO_ROOT not in act_pretrained_dir().parents
