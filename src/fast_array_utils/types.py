# SPDX-License-Identifier: MPL-2.0
"""Shared types."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable


if TYPE_CHECKING:
    from typing import Any, TypeAlias

    import numpy as np


__all__ = [
    "CSBase",
    "CupyArray",
    "CupySparseMatrix",
    "DaskArray",
    "H5Dataset",
    "OutOfCoreDataset",
    "ZarrArray",
]

T_co = TypeVar("T_co", covariant=True)


# scipy sparse
if TYPE_CHECKING:
    from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

    _SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.generic)

    CSArray: TypeAlias = csr_array[Any, np.dtype[_SCT_co]] | csc_array[Any, np.dtype[_SCT_co]]
    CSMatrix: TypeAlias = csr_matrix[Any, np.dtype[_SCT_co]] | csc_matrix[Any, np.dtype[_SCT_co]]
    CSBase: TypeAlias = CSMatrix[_SCT_co] | CSArray[_SCT_co]
else:
    try:  # cs?_array isn’t available in older scipy versions
        from scipy.sparse import csc_array, csr_array

        CSArray = csr_array | csc_array
    except ImportError:
        CSArray = type("CSArray", (), {})

    try:  # cs?_matrix is available when scipy is installed
        from scipy.sparse import csc_matrix, csr_matrix

        CSMatrix = csr_matrix | csc_matrix
    except ImportError:
        CSMatrix = type("CSMatrix", (), {})

    CSBase = CSMatrix | CSArray


if find_spec("cupy") or TYPE_CHECKING:
    from cupy import ndarray as CupyArray
else:
    CupyArray = type("ndarray", (), {})


if find_spec("cupyx") or TYPE_CHECKING:
    from cupyx.scipy.sparse import spmatrix as CupySparseMatrix
else:
    CupySparseMatrix = type("spmatrix", (), {})


if find_spec("dask") or TYPE_CHECKING:
    from dask.array import Array as DaskArray
else:
    DaskArray = type("array", (), {})


if find_spec("h5py") or TYPE_CHECKING:
    from h5py import Dataset as H5Dataset
else:
    H5Dataset = type("Dataset", (), {})


if find_spec("zarr") or TYPE_CHECKING:
    from zarr import Array as ZarrArray
else:
    ZarrArray = type("Array", (), {})


@runtime_checkable
class OutOfCoreDataset(Protocol, Generic[T_co]):
    """An out-of-core dataset."""

    def to_memory(self) -> T_co:
        """Load data into memory."""
        ...
