# SPDX-License-Identifier: MPL-2.0
"""Shared types that can be used e.g. in :func:`isinstance` checks."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, TypeVar


__all__ = [
    "CSArray",
    "CSBase",
    "CSDataset",
    "CSMatrix",
    "CupyArray",
    "CupyCSCMatrix",
    "CupyCSMatrix",
    "CupyCSRMatrix",
    "CupySpMatrix",
    "DaskArray",
    "H5Dataset",
    "H5Group",
    "ZarrArray",
    "ZarrGroup",
]

T_co = TypeVar("T_co", covariant=True)


# scipy sparse
if TYPE_CHECKING:
    from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, sparray, spmatrix
else:
    try:  # cs?_array isn’t available in older scipy versions
        from scipy.sparse import csc_array, csr_array, sparray
    except ImportError:  # pragma: no cover
        csc_array = type("csc_array", (), {})
        csr_array = type("csr_array", (), {})
        sparray = type("sparray", (), {})
        csc_array.__module__ = csr_array.__module__ = sparray.__module__ = "scipy.sparse"

    try:  # cs?_matrix is available when scipy is installed
        from scipy.sparse import csc_matrix, csr_matrix, spmatrix
    except ImportError:  # pragma: no cover
        csc_matrix = type("csc_matrix", (), {})
        csr_matrix = type("csr_matrix", (), {})
        spmatrix = type("spmatrix", (), {})
        csc_matrix.__module__ = csr_matrix.__module__ = spmatrix.__module__ = "scipy.sparse"
CSMatrix = csc_matrix | csr_matrix
CSArray = csc_array | csr_array
CSBase = CSMatrix | CSArray
"""A sparse compressed matrix or array."""


if TYPE_CHECKING or find_spec("cupy"):  # cupy always comes with cupyx
    from cupy import ndarray as CupyArray
    from cupyx.scipy.sparse import csc_matrix as CupyCSCMatrix
    from cupyx.scipy.sparse import csr_matrix as CupyCSRMatrix
    from cupyx.scipy.sparse import spmatrix as CupySpMatrix
else:  # pragma: no cover
    CupyArray = type("ndarray", (), {})
    CupyArray.__module__ = "cupy"
    CupyCSCMatrix = type("csc_matrix", (), {})
    CupyCSRMatrix = type("csr_matrix", (), {})
    CupySpMatrix = type("spmatrix", (), {})
    CupyCSCMatrix.__module__ = CupyCSRMatrix.__module__ = CupySpMatrix.__module__ = (
        "cupyx.scipy.sparse"
    )
CupyCSMatrix = CupyCSRMatrix | CupyCSCMatrix


if TYPE_CHECKING or find_spec("dask"):
    from dask.array import Array as DaskArray
else:  # pragma: no cover
    DaskArray = type("Array", (), {})
    DaskArray.__module__ = "dask.array"


if TYPE_CHECKING or find_spec("h5py"):
    from h5py import Dataset as H5Dataset
    from h5py import Group as H5Group
else:  # pragma: no cover
    H5Dataset = type("Dataset", (), {})
    H5Group = type("Group", (), {})
    H5Dataset.__module__ = H5Group.__module__ = "h5py"


if TYPE_CHECKING or find_spec("zarr"):
    from zarr import Array as ZarrArray
    from zarr import Group as ZarrGroup
else:  # pragma: no cover
    ZarrArray = type("Array", (), {})
    ZarrGroup = type("Group", (), {})
    ZarrArray.__module__ = ZarrGroup.__module__ = "zarr"


if TYPE_CHECKING:
    from anndata.abc import CSCDataset, CSRDataset  # type: ignore[import-untyped]
else:  # pragma: no cover
    try:  # only exists in anndata 0.11+
        from anndata.abc import CSCDataset, CSRDataset  # type: ignore[import-untyped]
    except ImportError:
        CSRDataset = type("CSRDataset", (), {})
        CSCDataset = type("CSCDataset", (), {})
        CSRDataset.__module__ = CSCDataset.__module__ = "anndata.abc"
CSDataset = CSRDataset | CSCDataset
"""Anndata sparse out-of-core matrices."""
