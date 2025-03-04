# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from ._sum import sum as sum_


if TYPE_CHECKING:
    from typing import Literal, TypeVar

    from numpy.typing import NBitBase, NDArray
    from optype.numpy import ToDType

    from .. import types

    # all supported types except Dask and OutOfCoreDataset (TODO)
    NonDaskArray = (
        NDArray[np.number[NBitBase]]
        | types.CSBase
        | types.H5Dataset
        | types.ZarrArray
        | types.CupyArray
        | types.CupySparseMatrix
    )
    Sc = TypeVar("Sc", bound=np.number[NBitBase])


@overload
def mean(x: NonDaskArray, /, *, axis: Literal[None] = None, dtype: None = None) -> np.float64: ...
@overload
def mean(x: NonDaskArray, /, *, axis: Literal[None] = None, dtype: ToDType[Sc]) -> Sc: ...
@overload
def mean(x: NonDaskArray, /, *, axis: Literal[0, 1], dtype: None = None) -> NDArray[np.float64]: ...
@overload
def mean(x: NonDaskArray, /, *, axis: Literal[0, 1], dtype: ToDType[Sc]) -> NDArray[Sc]: ...
@overload
def mean(
    x: types.DaskArray, /, *, axis: Literal[0, 1], dtype: ToDType[Sc] | None = None
) -> types.DaskArray: ...


def mean(
    x: NonDaskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: ToDType[Sc] | None = None,
) -> NDArray[Sc] | Sc | types.DaskArray:
    """Mean over both or one axis.

    Returns
    -------
    If ``axis`` is :data:`None`, then the sum over all elements is returned as a scalar.
    Otherwise, the sum over the given axis is returned as a 1D array.

    See Also
    --------
    :func:`numpy.mean`
    """
    total = sum_(x, axis=axis, dtype=dtype)
    n = np.prod(x.shape) if axis is None else x.shape[axis]
    return total / n
