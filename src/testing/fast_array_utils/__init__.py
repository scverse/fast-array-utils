# SPDX-License-Identifier: MPL-2.0
"""Testing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._array_type import ArrayType, ConversionContext, Flags, random_mat


if TYPE_CHECKING:
    from ._array_type import Array, ToArray  # noqa: TC004


__all__ = [
    "SUPPORTED_TYPES",
    "Array",
    "ArrayType",
    "ConversionContext",
    "Flags",
    "ToArray",
    "random_mat",
]


_TP_MEM = (
    ArrayType("numpy", "ndarray", Flags.Any),
    ArrayType("cupy", "ndarray", Flags.Any | Flags.Gpu),
    *(
        ArrayType("scipy.sparse", n, Flags.Any | Flags.Sparse)
        for n in ["csr_array", "csc_array", "csr_matrix", "csc_matrix"]
    ),
    *(
        ArrayType("cupyx.scipy.sparse", n, Flags.Any | Flags.Gpu | Flags.Sparse)
        for n in ["csr_matrix", "csc_matrix"]
    ),
)
_TP_DASK = tuple(ArrayType("dask.array", "Array", Flags.Dask | t.flags, inner=t) for t in _TP_MEM)
_TP_DISK = tuple(
    ArrayType(m, n, Flags.Any | Flags.Disk) for m, n in [("h5py", "Dataset"), ("zarr", "Array")]
)

SUPPORTED_TYPES: tuple[ArrayType, ...] = (*_TP_MEM, *_TP_DASK, *_TP_DISK)
"""All supported array types."""
