# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np

from .types import CSBase, CupyArray, CupySparseMatrix, DaskArray, H5Dataset, OutOfCoreDataset


if TYPE_CHECKING:
    from numpy.typing import ArrayLike


__all__ = ["asarray"]


@singledispatch
def asarray(x: ArrayLike) -> np.ndarray:
    """Convert x to a numpy array."""
    return np.asarray(x)


@asarray.register(CSBase)
def _(x: CSBase) -> np.ndarray:
    from .scipy import to_dense

    return to_dense(x)


@asarray.register(DaskArray)
def _(x: DaskArray) -> np.ndarray:
    return asarray(x.compute())


@asarray.register(OutOfCoreDataset)
def _(x: OutOfCoreDataset) -> np.ndarray:
    return asarray(x.to_memory())


@asarray.register(H5Dataset)
def _(x: H5Dataset) -> np.ndarray:
    return x[...]


@asarray.register(CupyArray)
def _(x: CupyArray) -> np.ndarray:
    return x.get()


@asarray.register(CupySparseMatrix)
def _(x: CupySparseMatrix) -> np.ndarray:
    return x.toarray().get()
