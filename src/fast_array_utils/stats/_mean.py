# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from ._sum import sum as sum_


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy._typing._array_like import _ArrayLikeFloat_co as ArrayLike
    from numpy.typing import DTypeLike, NDArray
    from optype.numpy import ToDType

    from .. import types

    # all supported types except Dask and OutOfCoreDataset (TODO)
    NonDaskArray = (
        NDArray[Any]
        | types.CSBase
        | types.H5Dataset
        | types.ZarrArray
        | types.CupyArray
        | types.CupySparseMatrix
    )
    Array = NonDaskArray | types.DaskArray


@overload
def mean(
    x: ArrayLike | NonDaskArray, /, *, axis: Literal[None] = None, dtype: DTypeLike | None = None
) -> np.number[Any]: ...
@overload
def mean(
    x: ArrayLike | NonDaskArray, /, *, axis: Literal[0, 1], dtype: DTypeLike | None = None
) -> NDArray[np.number[Any]]: ...
@overload
def mean(
    x: types.DaskArray, /, *, axis: Literal[0, 1], dtype: ToDType[Any] | None = None
) -> types.DaskArray: ...


def mean(
    x: ArrayLike | Array,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
) -> NDArray[np.number[Any]] | np.number[Any] | types.DaskArray:
    """Mean over both or one axis.

    Returns
    -------
    If ``axis`` is :data:`None`, then the sum over all elements is returned as a scalar.
    Otherwise, the sum over the given axis is returned as a 1D array.

    See Also
    --------
    :func:`numpy.mean`
    """
    if not hasattr(x, "shape"):
        raise NotImplementedError  # TODO(flying-sheep): infer shape  # noqa: TD003
    if TYPE_CHECKING:
        assert isinstance(x, Array)  # type:ignore[unused-ignore]
    total = sum_(x, axis=axis, dtype=dtype)
    n = np.prod(x.shape) if axis is None else x.shape[axis]
    return total / n
