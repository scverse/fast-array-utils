# SPDX-License-Identifier: MPL-2.0
"""Sphinx configuration."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import metadata
from pathlib import Path


HERE = Path(__file__).parent


# -- General configuration ------------------------------------------------


# General information
project = "fast-array-utils"
meta = metadata(project)
author = meta["author-email"].split('"')[1]
copyright = f"{datetime.now(tz=timezone.utc):%Y}, {author}."  # noqa: A001
version = meta["version"]
release = version

# default settings
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "scanpydoc",
]

#  API documentation when building
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
todo_include_todos = False

intersphinx_mapping = dict(
    numpy=("https://docs.scipy.org/doc/numpy/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/reference/", None),
)

# Options for HTML output
html_theme = "furo"
html_theme_options = dict(collapse_navigation=True)
html_context = dict(
    display_github=True,
    github_user="theislab",
    github_repo="anndata2ri",
    github_version="main",
    conf_py_path="/docs/",
)
