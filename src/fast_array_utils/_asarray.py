# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np

from .types import CSBase, CupyArray, CupySparseMatrix, DaskArray, H5Dataset, OutOfCoreDataset


if TYPE_CHECKING:
    from typing import Any, TypeVar

    from numpy.typing import ArrayLike, NDArray

    DT_co = TypeVar("DT_co", covariant=True, bound=np.generic)


__all__ = ["OutOfCoreDataset", "asarray"]


# fallback’s arg0 type has to include types of registered functions
@singledispatch
def asarray(x: ArrayLike | CSBase[DT_co] | OutOfCoreDataset[DT_co]) -> NDArray[DT_co]:
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


@asarray.register(CSBase)  # type: ignore[call-overload,misc]
def _(x: CSBase[DT_co]) -> NDArray[DT_co]:
    from .scipy import to_dense

    return to_dense(x)


@asarray.register(DaskArray)
def _(x: DaskArray[DT_co]) -> NDArray[DT_co]:
    return asarray(x.compute())


@asarray.register(OutOfCoreDataset)
def _(x: OutOfCoreDataset[CSBase[DT_co] | NDArray[DT_co]]) -> NDArray[DT_co]:
    return asarray(x.to_memory())


@asarray.register(H5Dataset)
def _(x: H5Dataset) -> NDArray[Any]:
    return x[...]  # type: ignore[no-any-return]


@asarray.register(CupyArray)
def _(x: CupyArray) -> NDArray[Any]:
    return x.get()  # type: ignore[no-any-return]


@asarray.register(CupySparseMatrix)
def _(x: CupySparseMatrix) -> NDArray[Any]:
    return x.toarray().get()  # type: ignore[no-any-return]
