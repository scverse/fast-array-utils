# SPDX-License-Identifier: MPL-2.0
"""Conversion utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from ..typing import CpuArray, DiskArray, GpuArray  # noqa: TC001
from ._to_dense import to_dense_


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray

    from .. import types


__all__ = ["to_dense"]


@overload
def to_dense(
    x: CpuArray | DiskArray | types.CSDataset, /, *, to_memory: bool = False
) -> NDArray[Any]: ...


@overload
def to_dense(x: types.DaskArray, /, *, to_memory: Literal[False] = False) -> types.DaskArray: ...
@overload
def to_dense(x: types.DaskArray, /, *, to_memory: Literal[True]) -> NDArray[Any]: ...


@overload
def to_dense(x: GpuArray, /, *, to_memory: Literal[False] = False) -> types.CupyArray: ...
@overload
def to_dense(x: GpuArray, /, *, to_memory: Literal[True]) -> NDArray[Any]: ...


def to_dense(
    x: CpuArray | GpuArray | DiskArray | types.CSDataset | types.DaskArray,
    /,
    *,
    to_memory: bool = False,
) -> NDArray[Any] | types.DaskArray | types.CupyArray:
    """Convert x to a dense array.

    Parameters
    ----------
    x
        Input object to be converted.
    to_memory
        Also load data into memory (resulting in a :class:`numpy.ndarray`).

    Returns
    -------
    Dense form of ``x``

    """
    return to_dense_(x, to_memory=to_memory)
