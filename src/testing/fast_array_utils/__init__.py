# SPDX-License-Identifier: MPL-2.0
"""Testing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .array_type import ArrayType, ConversionContext, random_mat


if TYPE_CHECKING:
    from .array_type import Array, ToArray  # noqa: TC004


__all__ = [
    "SUPPORTED_TYPES",
    "SUPPORTED_TYPES_DASK",
    "SUPPORTED_TYPES_DISK",
    "SUPPORTED_TYPES_MEM",
    "SUPPORTED_TYPES_MEM_DENSE",
    "SUPPORTED_TYPES_MEM_SPARSE",
    "Array",
    "ArrayType",
    "ConversionContext",
    "ToArray",
    "random_mat",
]


_SUPPORTED_TYPE_NAMES_DISK = [
    "h5py.Dataset",
    "zarr.Array",
]
_SUPPORTED_TYPE_NAMES_DENSE = [
    "numpy.ndarray",
    "cupy.ndarray",
]
_SUPPORTED_TYPE_NAMES_SPARSE = [
    "scipy.sparse.csr_array",
    "scipy.sparse.csc_array",
    "scipy.sparse.csr_matrix",
    "scipy.sparse.csc_matrix",
    "cupyx.scipy.sparse.csr_matrix",
    "cupyx.scipy.sparse.csc_matrix",
]

SUPPORTED_TYPES_DISK: tuple[ArrayType, ...] = tuple(
    map(ArrayType.from_qualname, _SUPPORTED_TYPE_NAMES_DISK)
)
"""Supported array types that represent on-disk data

These on-disk array types are not supported inside dask arrays.
"""

SUPPORTED_TYPES_MEM_DENSE: tuple[ArrayType, ...] = tuple(
    map(ArrayType.from_qualname, _SUPPORTED_TYPE_NAMES_DENSE)
)
"""Supported dense in-memory array types."""

SUPPORTED_TYPES_MEM_SPARSE: tuple[ArrayType, ...] = tuple(
    map(ArrayType.from_qualname, _SUPPORTED_TYPE_NAMES_SPARSE)
)
"""Supported sparse in-memory array types."""

SUPPORTED_TYPES_MEM: tuple[ArrayType, ...] = (
    *SUPPORTED_TYPES_MEM_DENSE,
    *SUPPORTED_TYPES_MEM_SPARSE,
)
"""Supported array types that are valid inside dask arrays."""

SUPPORTED_TYPES_DASK: tuple[ArrayType, ...] = tuple(
    ArrayType("dask.array", "Array", t) for t in SUPPORTED_TYPES_MEM
)
"""Supported dask array types."""

SUPPORTED_TYPES: tuple[ArrayType, ...] = (
    *SUPPORTED_TYPES_MEM,
    *SUPPORTED_TYPES_DASK,
    *SUPPORTED_TYPES_DISK,
)
"""All supported array types."""
