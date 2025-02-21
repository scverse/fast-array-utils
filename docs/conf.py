# SPDX-License-Identifier: MPL-2.0
"""Sphinx configuration."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import metadata
from pathlib import Path


HERE = Path(__file__).parent


# -- General configuration ------------------------------------------------


# General information
project = "fast-array-utils"
meta = metadata(project)
author = meta["author-email"].split('"')[1]
copyright = f"{datetime.now(tz=UTC):%Y}, {author}."  # noqa: A001
version = meta["version"]
release = version

# default settings
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "scanpydoc.elegant_typehints",
    "sphinx_autofixture",
]

#  API documentation when building
nitpicky = True
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "special-members": True,
    # everything except __call__ really, to avoid having to write autosummary templates
    "exclude-members": "__setattr__,__delattr__,__repr__,__eq__,__hash__,__weakref__,__init__",
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
todo_include_todos = False
intersphinx_mapping = dict(
    cupy=("https://docs.cupy.dev/en/stable/", None),
    dask=("https://docs.dask.org/en/stable/", None),
    h5py=("https://docs.h5py.org/en/stable/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    zarr=("https://zarr.readthedocs.io/en/stable/", None),
)
# Try overriding type paths
qualname_overrides = autodoc_type_aliases = {
    "np.dtype": "numpy.dtype",
    "np.number": "numpy.number",
    "np.integer": "numpy.integer",
    "np.random.Generator": "numpy.random.Generator",
    "ArrayLike": "numpy.typing.ArrayLike",
    "DTypeLike": "numpy.typing.DTypeLike",
    "NDArray": "numpy.typing.NDArray",
    "_pytest.fixtures.FixtureRequest": "pytest.FixtureRequest",
    **{
        k: v
        for k_plain, v in {
            "CSBase": "scipy.sparse.spmatrix",
            "CupyArray": "cupy.ndarray",
            "CupySparseMatrix": "cupyx.scipy.sparse.spmatrix",
            "DaskArray": "dask.array.Array",
            "H5Dataset": "h5py.Dataset",
            "ZarrArray": "zarr.Array",
        }.items()
        for k in (k_plain, f"types.{k_plain}")
    },
}
# If that doesn’t work, ignore them
nitpick_ignore = {
    ("py:class", "fast_array_utils.types.T_co"),
    ("py:class", "_DTypeLikeFloat32"),
    ("py:class", "_DTypeLikeFloat64"),
    # sphinx bugs, should be covered by `autodoc_type_aliases` above
    ("py:class", "Array"),
    ("py:class", "ArrayLike"),
    ("py:class", "DTypeLike"),
    ("py:class", "NDArray"),
    ("py:class", "_pytest.fixtures.FixtureRequest"),
}

# Options for HTML output
html_theme = "furo"
html_theme_options = dict(
    source_repository="https://github.com/scverse/fast-array-utils/",
    source_branch="main",
    source_directory="docs/",
)
