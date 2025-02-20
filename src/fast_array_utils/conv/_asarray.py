# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np

from .. import types


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, NDArray


__all__ = ["asarray"]


# fallback’s arg0 type has to include types of registered functions
@singledispatch
def asarray(
    x: ArrayLike
    | types.CSBase
    | types.DaskArray
    | types.OutOfCoreDataset[Any]
    | types.H5Dataset
    | types.ZarrArray
    | types.CupyArray
    | types.CupySparseMatrix,
) -> NDArray[Any]:
    """Convert x to a numpy array.

    Parameters
    ----------
    x
        Input object to be converted.

    Returns
    -------
    Numpy array form of ``x``

    """
    return np.asarray(x)


@asarray.register(types.CSBase)
def _(x: types.CSBase) -> NDArray[Any]:
    from .scipy import to_dense

    return to_dense(x)


@asarray.register(types.DaskArray)
def _(x: types.DaskArray) -> NDArray[Any]:
    return asarray(x.compute())  # type: ignore[no-untyped-call]


@asarray.register(types.OutOfCoreDataset)
def _(x: types.OutOfCoreDataset[types.CSBase | NDArray[Any]]) -> NDArray[Any]:
    return asarray(x.to_memory())


@asarray.register(types.CupyArray)
def _(x: types.CupyArray) -> NDArray[Any]:
    return x.get()  # type: ignore[no-any-return]


@asarray.register(types.CupySparseMatrix)
def _(x: types.CupySparseMatrix) -> NDArray[Any]:
    return x.toarray().get()  # type: ignore[no-any-return]
