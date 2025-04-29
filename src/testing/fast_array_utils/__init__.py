# SPDX-License-Identifier: MPL-2.0
"""Testing utilities.

Ideally used through the :mod:`testing.fast_array_utils.pytest` plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ._array_type import ArrayType, ConversionContext, Flags, random_mat


if TYPE_CHECKING:
    from fast_array_utils.typing import CpuArray, DiskArray, GpuArray

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
    *(ArrayType("scipy.sparse", n, Flags.Any | Flags.Sparse) for n in ["csr_array", "csc_array"]),
    *(
        ArrayType(mod, n, Flags.Any | Flags.Sparse | Flags.Matrix | flags)
        for n in ["csr_matrix", "csc_matrix"]
        for (mod, flags) in [("scipy.sparse", Flags(0)), ("cupyx.scipy.sparse", Flags.Gpu)]
    ),
)
_TP_DASK = tuple(
    ArrayType("dask.array", "Array", Flags.Dask | t.flags, inner=t)  # type: ignore[type-var]
    for t in cast("tuple[ArrayType[CpuArray | GpuArray, None], ...]", _TP_MEM)
)
_TP_DISK_DENSE = tuple(ArrayType(m, n, Flags.Any | Flags.Disk) for m, n in [("h5py", "Dataset"), ("zarr", "Array")])
_TP_DISK_SPARSE = tuple(
    ArrayType("anndata.abc", n, Flags.Any | Flags.Disk | Flags.Sparse, inner=t)  # type: ignore[type-var]
    for t in cast("tuple[ArrayType[DiskArray, None], ...]", _TP_DISK_DENSE)
    for n in ["CSRDataset", "CSCDataset"]
)

SUPPORTED_TYPES: tuple[ArrayType, ...] = (*_TP_MEM, *_TP_DASK, *_TP_DISK_DENSE, *_TP_DISK_SPARSE)
"""All supported array types."""
