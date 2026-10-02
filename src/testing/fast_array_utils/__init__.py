# SPDX-License-Identifier: MPL-2.0
"""Testing utilities.

Ideally used through the :mod:`testing.fast_array_utils.pytest` plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


_TP_MEM: tuple[ArrayType[CpuArray | GpuArray, None], ...] = (
    ArrayType("numpy", "ndarray", Flags.Any),
    ArrayType("cupy", "ndarray", Flags.Any | Flags.Gpu),
    ArrayType("scipy.sparse", "csr_array", Flags.Any | Flags.Sparse),
    ArrayType("scipy.sparse", "csc_array", Flags.Any | Flags.Sparse),
    ArrayType("scipy.sparse", "csr_matrix", Flags.Any | Flags.Sparse | Flags.Matrix),
    ArrayType("scipy.sparse", "csc_matrix", Flags.Any | Flags.Sparse | Flags.Matrix),
    ArrayType("cupyx.scipy.sparse", "csr_matrix", Flags.Any | Flags.Sparse | Flags.Matrix | Flags.Gpu),
    ArrayType("cupyx.scipy.sparse", "csc_matrix", Flags.Any | Flags.Sparse | Flags.Matrix | Flags.Gpu),
)
_TP_DASK = tuple(ArrayType("dask.array", "Array", Flags.Dask | t.flags, inner=t) for t in _TP_MEM)
_TP_DISK_DENSE: tuple[ArrayType[DiskArray, None], ...] = (
    ArrayType("h5py", "Dataset", Flags.Any | Flags.Disk),
    ArrayType("zarr", "Array", Flags.Any | Flags.Disk),
)
_TP_DISK_SPARSE = tuple(
    ArrayType("anndata.abc", n, Flags.Any | Flags.Disk | Flags.Sparse, inner=t) for t in _TP_DISK_DENSE for n in ["CSRDataset", "CSCDataset"]
)

SUPPORTED_TYPES: tuple[ArrayType, ...] = (*_TP_MEM, *_TP_DASK, *_TP_DISK_DENSE, *_TP_DISK_SPARSE)
"""All supported array types."""
