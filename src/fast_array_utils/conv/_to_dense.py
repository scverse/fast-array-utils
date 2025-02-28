# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
from numpy.typing import NDArray

from .. import types


if TYPE_CHECKING:
    from typing import Literal, TypeAlias

    Array: TypeAlias = (
        NDArray[Any]
        | types.CSBase
        | types.DaskArray
        | types.OutOfCoreDataset[Any]
        | types.H5Dataset
        | types.ZarrArray
        | types.CupyArray
        | types.CupySparseMatrix
    )


__all__ = ["to_dense"]


@overload
def to_dense(x: NDArray[Any] | types.CSBase, /, *, to_memory: bool = False) -> NDArray[Any]: ...


@overload
def to_dense(x: types.DaskArray, /, *, to_memory: Literal[False] = False) -> types.DaskArray: ...
@overload
def to_dense(x: types.DaskArray, /, *, to_memory: Literal[True]) -> NDArray[Any]: ...


@overload
def to_dense(
    x: types.OutOfCoreDataset[types.CSBase | NDArray[Any]], /, *, to_memory: Literal[True]
) -> NDArray[Any]: ...


@overload
def to_dense(  # type: ignore[overload-cannot-match]
    x: types.CupyArray | types.CupySparseMatrix, /, *, to_memory: Literal[False] = False
) -> types.CupyArray: ...
@overload
def to_dense(  # type: ignore[overload-cannot-match]
    x: types.CupyArray | types.CupySparseMatrix, /, *, to_memory: Literal[True]
) -> NDArray[Any]: ...


def to_dense(
    x: Array, /, *, to_memory: bool = False
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
    return _to_dense(x, to_memory=to_memory)


# fallback’s arg0 type has to include types of registered functions
@singledispatch
def _to_dense(
    x: Array, /, *, to_memory: bool = False
) -> NDArray[Any] | types.DaskArray | types.CupyArray:
    del to_memory  # it already is
    return np.asarray(x)


@_to_dense.register(types.CSBase)
def _(x: types.CSBase, /, *, to_memory: bool = False) -> NDArray[Any]:
    from . import scipy

    del to_memory  # it already is
    return scipy.to_dense(x)


@_to_dense.register(types.DaskArray)
def _(x: types.DaskArray, /, *, to_memory: bool = False) -> NDArray[Any] | types.DaskArray:
    if TYPE_CHECKING:
        from dask.array.core import map_blocks
    else:
        from dask.array import map_blocks

    x = cast(types.DaskArray, map_blocks(to_dense, x))  # type: ignore[no-untyped-call]
    return x.compute() if to_memory else x  # type: ignore[no-untyped-call]


@_to_dense.register(types.OutOfCoreDataset)
def _(
    x: types.OutOfCoreDataset[types.CSBase | NDArray[Any]], /, *, to_memory: bool = False
) -> NDArray[Any]:
    if not to_memory:
        msg = "to_memory must be True if x is an OutOfCoreDataset"
        raise ValueError(msg)
    # TODO(flying-sheep): why is to_memory of type Any?  # noqa: TD003
    return to_dense(x.to_memory())  # type: ignore[no-any-return]


@_to_dense.register(types.CupyArray | types.CupySparseMatrix)
def _(
    x: types.CupyArray | types.CupySparseMatrix, /, *, to_memory: bool = False
) -> NDArray[Any] | types.CupyArray:
    x = cast(types.CupyArray, x.toarray()) if isinstance(x, types.CupySparseMatrix) else x
    return cast(NDArray[Any], x.get()) if to_memory else x
