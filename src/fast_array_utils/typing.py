# SPDX-License-Identifier: MPL-2.0
"""Type categories to be used in type annotations."""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from . import types


__all__ = ["CpuArray", "DiskArray", "GpuArray"]


type CpuArray = NDArray[Any] | types.CSBase
"""Arrays and matrices stored in CPU memory."""

type GpuArray = types.CupyArray | types.CupyCSMatrix
"""Arrays and matrices stored in GPU memory."""

# TODO(flying-sheep): types.CSDataset  # noqa: TD003
type DiskArray = types.H5Dataset | types.ZarrArray  # type: ignore[type-arg]
"""Arrays and matrices stored on disk."""
