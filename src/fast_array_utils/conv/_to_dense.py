# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, cast, overload

import numpy as np

from .. import types


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray

    Array = (
        NDArray[np.generic]
        | types.CSBase
        | types.CupyArray
        | types.CupySparseMatrix
        | types.DaskArray
        | types.OutOfCoreDataset[Any]
        | types.H5Dataset
        | types.ZarrArray
    )


__all__ = ["to_dense"]


@overload
def to_dense(
    x: (
        NDArray[np.generic]
        | types.CSBase
        | types.OutOfCoreDataset[Any]
        | types.H5Dataset
        | types.ZarrArray
    ),
    /,
    *,
    to_memory: bool = False,
) -> NDArray[np.generic]: ...


@overload
def to_dense(x: types.DaskArray, /, *, to_memory: Literal[False] = False) -> types.DaskArray: ...
@overload
def to_dense(x: types.DaskArray, /, *, to_memory: Literal[True]) -> NDArray[np.generic]: ...


@overload
def to_dense(
    x: types.CupyArray | types.CupySparseMatrix, /, *, to_memory: Literal[False] = False
) -> types.CupyArray: ...
@overload
def to_dense(
    x: types.CupyArray | types.CupySparseMatrix, /, *, to_memory: Literal[True]
) -> NDArray[np.generic]: ...


def to_dense(
    x: Array, /, *, to_memory: bool = False
) -> NDArray[np.generic] | types.DaskArray | types.CupyArray:
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
) -> NDArray[np.generic] | types.DaskArray | types.CupyArray:
    del to_memory  # it already is
    return np.asarray(x)


@_to_dense.register(types.CSBase)
def _to_dense_cs(x: types.CSBase, /, *, to_memory: bool = False) -> NDArray[np.generic]:
    from . import scipy

    del to_memory  # it already is
    return scipy.to_dense(x)


@_to_dense.register(types.DaskArray)
def _to_dense_dask(
    x: types.DaskArray, /, *, to_memory: bool = False
) -> NDArray[np.generic] | types.DaskArray:
    if TYPE_CHECKING:
        from dask.array.core import map_blocks
    else:
        from dask.array import map_blocks

    x = cast(types.DaskArray, map_blocks(to_dense, x))
    return x.compute() if to_memory else x


@_to_dense.register(types.OutOfCoreDataset)
def _to_dense_ooc(
    x: types.OutOfCoreDataset[types.CSBase | NDArray[np.generic]], /, *, to_memory: bool = False
) -> NDArray[np.generic]:
    if not to_memory:
        msg = "to_memory must be True if x is an OutOfCoreDataset"
        raise ValueError(msg)
    # TODO(flying-sheep): why is to_memory of type Any?  # noqa: TD003
    return to_dense(x.to_memory())


@_to_dense.register(types.CupyArray | types.CupySparseMatrix)
def _to_dense_cupy(
    x: types.CupyArray | types.CupySparseMatrix, /, *, to_memory: bool = False
) -> NDArray[np.generic] | types.CupyArray:
    x = x.toarray() if isinstance(x, types.CupySparseMatrix) else x
    return x.get() if to_memory else x
