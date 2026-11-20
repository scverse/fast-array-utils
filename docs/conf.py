# SPDX-License-Identifier: MPL-2.0
"""Sphinx configuration."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import metadata

from scverse_doc.registry import intersphinx


project = "fast-array-utils"
meta = metadata(project)
author = meta["author-email"].split('"')[1]
copyright = f"{datetime.now(tz=UTC):%Y}, {author}."  # noqa: A001
version = meta["version"]
release = version

extensions = ["scverse_doc", "sphinx_autofixture"]
source_repository = "https://github.com/scverse/fast-array-utils"

autodoc_member_order = "bysource"
autodoc_default_options = {
    "special-members": True,
    # everything except __call__ really, to avoid having to write autosummary templates
    "exclude-members": "__setattr__,__delattr__,__repr__,__eq__,__or__,__ror__,__hash__,__weakref__,__init__,__new__",
}
intersphinx_mapping = intersphinx() | dict(
    cupy=("https://docs.cupy.dev/en/stable/", None),
    dask=("https://docs.dask.org/en/stable/", None),
    h5py=("https://docs.h5py.org/en/stable/", None),
    numba=("https://numba.readthedocs.io/en/stable/", None),
    pytest=("https://docs.pytest.org/en/stable/", None),
    zarr=("https://zarr.readthedocs.io/en/stable/", None),
)
xref_overrides = {
    "numpy._typing._array_like.NDArray": ("py:data", "numpy.typing.NDArray"),
    "numpy._typing._dtype_like.DTypeLike": ("py:data", "numpy.typing.DTypeLike"),
    "DTypeLike": ("py:data", "numpy.typing.DTypeLike"),
    "ArrayLike": ("py:data", "numpy.typing.ArrayLike"),
    "numpy.float64": ("py:attr", "numpy.float64"),
    "np.floating": "numpy.floating",
    "np.random.Generator": "numpy.random.Generator",
}
nitpick_ignore = [
    ("py:class", name) for name in ["A", "P", "R", "Arr", "Array", "Inner", "ToDType", "_DTypeLikeNum", "testing.fast_array_utils._array_type._DTypeLikeNum"]
]
