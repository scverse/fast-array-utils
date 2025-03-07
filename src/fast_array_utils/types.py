# SPDX-License-Identifier: MPL-2.0
"""Shared types."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, TypeVar


__all__ = [
    "CSBase",
    "CupyArray",
    "CupySparseMatrix",
    "DaskArray",
    "H5Dataset",
    "ZarrArray",
]

T_co = TypeVar("T_co", covariant=True)


# scipy sparse
if TYPE_CHECKING:
    from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

    CSArray = csr_array | csc_array
    CSMatrix = csr_matrix | csc_matrix
else:
    try:  # cs?_array isn’t available in older scipy versions
        from scipy.sparse import csc_array, csr_array

        CSArray = csr_array | csc_array
    except ImportError:  # pragma: no cover
        CSArray = type("CSArray", (), {})

    try:  # cs?_matrix is available when scipy is installed
        from scipy.sparse import csc_matrix, csr_matrix

        CSMatrix = csr_matrix | csc_matrix
    except ImportError:  # pragma: no cover
        CSMatrix = type("CSMatrix", (), {})
CSBase = CSMatrix | CSArray


if TYPE_CHECKING or find_spec("cupy"):
    from cupy import ndarray as CupyArray
else:  # pragma: no cover
    CupyArray = type("ndarray", (), {})


if TYPE_CHECKING or find_spec("cupyx"):
    from cupyx.scipy.sparse import spmatrix as CupySparseMatrix
else:  # pragma: no cover
    CupySparseMatrix = type("spmatrix", (), {})


if TYPE_CHECKING:  # https://github.com/dask/dask/issues/8853
    from dask.array.core import Array as DaskArray
elif find_spec("dask"):
    from dask.array import Array as DaskArray
else:  # pragma: no cover
    DaskArray = type("array", (), {})


if TYPE_CHECKING or find_spec("h5py"):
    from h5py import Dataset as H5Dataset
    from h5py import Group as H5Group
else:  # pragma: no cover
    H5Dataset = type("Dataset", (), {})
    H5Group = type("Group", (), {})


if TYPE_CHECKING or find_spec("zarr"):
    from zarr import Array as ZarrArray
    from zarr import Group as ZarrGroup
else:  # pragma: no cover
    ZarrArray = type("Array", (), {})
    ZarrGroup = type("Group", (), {})


if TYPE_CHECKING or find_spec("anndata"):
    from anndata.abc import CSCDataset, CSRDataset
else:  # pragma: no cover
    CSRDataset = type("CSRDataset", (), {})
    CSCDataset = type("CSCDataset", (), {})
CSDataset = CSRDataset | CSCDataset
