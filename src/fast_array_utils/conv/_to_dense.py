# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .. import types


if TYPE_CHECKING:
    from numpy.typing import ArrayLike


__all__ = ["to_dense"]


# fallback’s arg0 type has to include types of registered functions
@singledispatch
def to_dense(
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


@to_dense.register(types.CSBase)
def _(x: types.CSBase) -> NDArray[Any]:
    from . import scipy

    return scipy.to_dense(x)


@to_dense.register(types.DaskArray)
def _(x: types.DaskArray) -> NDArray[Any]:
    return to_dense(x.compute())  # type: ignore[no-untyped-call]


@to_dense.register(types.OutOfCoreDataset)
def _(x: types.OutOfCoreDataset[types.CSBase | NDArray[Any]]) -> NDArray[Any]:
    return to_dense(x.to_memory())


@to_dense.register(types.CupyArray)
def _(x: types.CupyArray) -> NDArray[Any]:
    return cast(NDArray[Any], x.get())


@to_dense.register(types.CupySparseMatrix)
def _(x: types.CupySparseMatrix) -> NDArray[Any]:
    return cast(NDArray[Any], x.toarray().get())
