"""Sphinx configuration for rl-garden."""
import os
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
html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
# No intersphinx until API pages exist: the remote build host cannot reach
# numpy.org, and the pages only use :doc:/:ref: targets inside this site.
extensions = ["myst_parser", "sphinx_copybutton", "sphinx_design"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
exclude_patterns = ["_build", "intro", "README.md"]
nitpicky = True
nitpick_ignore = []
# GitHub renders README anchors client-side, so linkcheck cannot see them.
linkcheck_anchors_ignore_for_url = [r"https://github\.com/.*"]

templates_path = ["_templates"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/sidebar-nav.js", "js/theme-toggle.js"]
html_show_sourcelink = False
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL", "https://rl-garden.readthedocs.io/en/latest/"
)
html_sidebars = {
    "**": ["sidebar-brand", "search-field", "sidebar-tools", "global-sidebar-nav"]
}
html_theme_options = {
    "search_bar_text": "Search docs…",
    "navbar_start": [],
    "navbar_center": [],
    "navbar_end": [],
    "navbar_persistent": [],
    "navbar_align": "left",
    "secondary_sidebar_items": {"**": ["page-toc"], "index": []},
    "show_nav_level": 1,
    "navigation_depth": 5,
    "collapse_navigation": False,
    "icon_links": [{
        "name": "GitHub",
        "url": "https://github.com/JaimeParker/rl-garden",
        "icon": "fab fa-github",
        "type": "fontawesome",
    }],
}
