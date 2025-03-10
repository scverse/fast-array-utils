# SPDX-License-Identifier: MPL-2.0
"""Type categories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from . import types


if TYPE_CHECKING:
    from typing import TypeAlias


CpuArray: TypeAlias = NDArray[Any] | types.CSBase
"""Arrays and matrices stored in CPU memory."""

GpuArray: TypeAlias = types.CupyArray | types.CupySparseMatrix
"""Arrays and matrices stored in GPU memory."""

# TODO(flying-sheep): types.CSDataset  # noqa: TD003
DiskArray: TypeAlias = types.H5Dataset | types.ZarrArray
"""Arrays and matrices stored on disk."""
