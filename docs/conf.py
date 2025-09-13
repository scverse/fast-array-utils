# SPDX-License-Identifier: MPL-2.0
"""Sphinx configuration."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from docutils.nodes import Text


if TYPE_CHECKING:
    from docutils.nodes import TextElement, reference
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
    "sphinx.ext.napoleon",
    # "scanpydoc.definition_list_typed_field",
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
    "exclude-members": "__setattr__,__delattr__,__repr__,__eq__,__or__,__ror__,__hash__,__weakref__,__init__,__new__",
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
todo_include_todos = False
intersphinx_mapping = dict(
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    cupy=("https://docs.cupy.dev/en/stable/", None),
    dask=("https://docs.dask.org/en/stable/", None),
    h5py=("https://docs.h5py.org/en/stable/", None),
    numba=("https://numba.readthedocs.io/en/stable/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    pytest=("https://docs.pytest.org/en/stable/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    zarr=("https://zarr.readthedocs.io/en/stable/", None),
)
nitpick_ignore = [
    ("py:class", "Arr"),
    ("py:class", "Array"),
    ("py:class", "ToDType"),
    ("py:class", "testing.fast_array_utils._array_type.Arr"),
    ("py:class", "testing.fast_array_utils._array_type.Inner"),
    ("py:class", "_DTypeLikeNum"),
]

# Options for HTML output
html_theme = "furo"
html_theme_options = dict(
    source_repository="https://github.com/scverse/fast-array-utils/",
    source_branch="main",
    source_directory="docs/",
)

_np_nocls = {"float64": "attr"}
_optional_types = {
    "CupyArray": "cupy.ndarray",
    "CupySpMatrix": "cupyx.scipy.sparse.spmatrix",
    "sparray": "scipy.sparse.sparray",
    "spmatrix": "scipy.sparse.spmatrix",
    "DaskArray": "dask.array.Array",
    "H5Dataset": "h5py.Dataset",
    "ZarrArray": "zarr.Array",
}


def find_type_alias(name: str) -> tuple[str, str, str | None] | tuple[None, None, None]:
    """Find a type alias."""
    import numpy.typing as npt

    from fast_array_utils import types, typing

    if name in typing.__all__:
        return "data", f"fast_array_utils.typing.{name}", None
    if name.startswith("types.") and name[6:] in {*types.__all__, *_optional_types}:
        if path := _optional_types.get(name[6:]):
            return "class", path, None
        return "data", f"fast_array_utils.{name}", None
    if name.startswith("np."):
        name = name.removeprefix("np.")
        return _np_nocls.get(name, "class"), f"numpy.{name}", f"np.{name}"
    if name in npt.__all__:
        return "data", f"numpy.typing.{name}", None
    return None, None, None


def resolve_type_aliases(app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: TextElement) -> reference | None:
    """Resolve :class: references to our type aliases as :attr: instead."""
    if (node["refdomain"], node["reftype"]) != ("py", "class"):
        return None
    typ, target, name = find_type_alias(node["reftarget"])
    if typ is None or target is None:
        return None
    if target.startswith("fast_array_utils."):
        ref = env.get_domain("py").resolve_xref(env, node["refdoc"], app.builder, typ, target, node, contnode)
    else:
        from sphinx.ext.intersphinx import resolve_reference_any_inventory

        node["reftype"] = typ
        node["reftarget"] = target
        ref = resolve_reference_any_inventory(env=env, honor_disabled_refs=False, node=node, contnode=contnode)
    if ref is None:
        msg = f"Could not resolve {typ} {target} (from {node['reftarget']})"
        raise AssertionError(msg)
    if name:
        ref.children[:] = [Text(name)]
    return ref


def setup(app: Sphinx) -> None:  # noqa: D103
    app.connect("missing-reference", resolve_type_aliases, priority=800)
