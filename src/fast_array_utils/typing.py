# SPDX-License-Identifier: MPL-2.0
"""Type categories to be used in type annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from . import types


if TYPE_CHECKING:
    from typing import TypeAlias


__all__ = ["CpuArray", "DiskArray", "GpuArray"]

# change to `type` syntax once this is released: https://github.com/sphinx-doc/sphinx/pull/13508

CpuArray: TypeAlias = NDArray[Any] | types.CSBase  # noqa: UP040
"""Arrays and matrices stored in CPU memory."""

GpuArray: TypeAlias = types.CupyArray | types.CupyCSMatrix  # noqa: UP040
"""Arrays and matrices stored in GPU memory."""

# TODO(flying-sheep): types.CSDataset  # noqa: TD003
DiskArray: TypeAlias = types.H5Dataset | types.ZarrArray  # noqa: UP040
"""Arrays and matrices stored on disk."""
