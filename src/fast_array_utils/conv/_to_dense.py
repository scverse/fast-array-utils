# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, overload

import numpy as np

from .. import types


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray

    Array = (
        NDArray[Any]
        | types.CSBase
        | types.CupyArray
        | types.CupySparseMatrix
        | types.DaskArray
        | types.H5Dataset
        | types.ZarrArray
        | types.CSDataset
    )


__all__ = ["to_dense"]


@overload
def to_dense(
    x: NDArray[Any] | types.CSBase | types.H5Dataset | types.ZarrArray | types.CSDataset,
    /,
    *,
    to_memory: bool = False,
) -> NDArray[Any]: ...


@overload
def to_dense(x: types.DaskArray, /, *, to_memory: Literal[False] = False) -> types.DaskArray: ...
@overload
def to_dense(x: types.DaskArray, /, *, to_memory: Literal[True]) -> NDArray[Any]: ...


@overload
def to_dense(
    x: types.CupyArray | types.CupySparseMatrix, /, *, to_memory: Literal[False] = False
) -> types.CupyArray: ...
@overload
def to_dense(
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


@_to_dense.register(types.CSBase)  # type: ignore[call-overload,misc]
def _to_dense_cs(x: types.CSBase, /, *, to_memory: bool = False) -> NDArray[Any]:
    from . import scipy

    del to_memory  # it already is
    return scipy.to_dense(x)


@_to_dense.register(types.DaskArray)
def _to_dense_dask(
    x: types.DaskArray, /, *, to_memory: bool = False
) -> NDArray[Any] | types.DaskArray:
    import dask.array as da

    x = da.map_blocks(to_dense, x)  # type: ignore[arg-type]
    return x.compute() if to_memory else x  # type: ignore[return-value]


@_to_dense.register(types.CSDataset)
def _to_dense_ooc(x: types.CSDataset, /, *, to_memory: bool = False) -> NDArray[Any]:
    if not to_memory:
        msg = "to_memory must be True if x is an CS{R,C}Dataset"
        raise ValueError(msg)
    # TODO(flying-sheep): why is to_memory of type Any?  # noqa: TD003
    return to_dense(x.to_memory())


@_to_dense.register(types.CupyArray | types.CupySparseMatrix)  # type: ignore[call-overload,misc]
def _to_dense_cupy(
    x: types.CupyArray | types.CupySparseMatrix, /, *, to_memory: bool = False
) -> NDArray[Any] | types.CupyArray:
    x = x.toarray() if isinstance(x, types.CupySparseMatrix) else x
    return x.get() if to_memory else x
