# SPDX-License-Identifier: MPL-2.0
"""Sphinx configuration."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import metadata
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from docutils.nodes import Element, reference
    from sphinx.addnodes import pending_xref
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment


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
master_doc = "index"
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
    "exclude-members": (
        "__setattr__,__delattr__,__repr__,__eq__,__or__,__ror__,__hash__,__weakref__,__init__,__new__"
    ),
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
    "np.bool": "numpy.bool",
    "np.dtype": "numpy.dtype",
    "np.number": "numpy.number",
    "np.integer": "numpy.integer",
    "np.float64": "numpy.float64",
    "np.floating": "numpy.floating",
    "np.random.Generator": "numpy.random.Generator",
    "ArrayLike": "numpy.typing.ArrayLike",
    "DTypeLike": "numpy.typing.DTypeLike",
    "NDArray": "numpy.typing.NDArray",
    "_pytest.fixtures.FixtureRequest": "pytest.FixtureRequest",
    **{
        k: v
        for k_plain, v in {
            "CSBase": "scipy.sparse.spmatrix",
            "CSDataset": "anndata.abc.CSRDataset",
            "CupyArray": "cupy.ndarray",
            "CupySparseMatrix": "cupyx.scipy.sparse.spmatrix",
            "DaskArray": "dask.array.Array",
            "H5Dataset": "h5py.Dataset",
            "ZarrArray": "zarr.Array",
        }.items()
        for k in (k_plain, f"types.{k_plain}")
    },
    **{t: f"fast_array_utils.typing.{t}" for t in ("CpuArray", "GpuArray", "DiskArray")},
}
# If that doesn’t work, ignore them
nitpick_ignore = {
    ("py:class", "fast_array_utils.types.T_co"),
    ("py:class", "Arr"),
    ("py:class", "testing.fast_array_utils._array_type.Arr"),
    ("py:class", "testing.fast_array_utils._array_type.Inner"),
    ("py:class", "_DTypeLikeFloat32"),
    ("py:class", "_DTypeLikeFloat64"),
    # sphinx bugs, should be covered by `autodoc_type_aliases` above
    ("py:class", "Array"),
    ("py:class", "ArrayLike"),
    ("py:class", "DTypeLike"),
    ("py:class", "NDArray"),
    ("py:class", "np.bool"),
    ("py:class", "np.float64"),
    ("py:class", "_pytest.fixtures.FixtureRequest"),
}

# Options for HTML output
html_theme = "furo"
html_theme_options = dict(
    source_repository="https://github.com/scverse/fast-array-utils/",
    source_branch="main",
    source_directory="docs/",
)


def resolve_type_aliases(
    app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: Element
) -> reference | None:
    """Resolve :class: references to our type aliases as :attr: instead."""
    if (
        node["refdomain"] == "py"
        and node["reftype"] == "class"
        and (t := autodoc_type_aliases.get(node["reftarget"]))
    ):
        rvs = env.get_domain("py").resolve_any_xref(
            env, node["refdoc"], app.builder, t, node, contnode
        )
        for _role, refnode in rvs:
            return refnode
        return None

    return None


def setup(app: Sphinx) -> None:  # noqa: D103
    app.connect("missing-reference", resolve_type_aliases)
