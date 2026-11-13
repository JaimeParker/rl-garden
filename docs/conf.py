"""Sphinx configuration for rl-garden."""
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    # Sphinx supplies tomli on Python 3.10, the minimum supported Python.
    import tomli as tomllib

with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as metadata_file:
    metadata = tomllib.load(metadata_file)["project"]

project = metadata["name"]
release = metadata["version"]
version = release
html_title = project
html_theme = "furo"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
# No intersphinx until API pages exist: the remote build host cannot reach
# numpy.org, and the pages only use :doc:/:ref: targets inside this site.
extensions = ["myst_parser", "sphinx_copybutton"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
exclude_patterns = ["_build", "intro", "README.md"]
nitpicky = True
nitpick_ignore = []
# GitHub renders README anchors client-side, so linkcheck cannot see them.
linkcheck_anchors_ignore_for_url = [r"https://github\.com/.*"]
