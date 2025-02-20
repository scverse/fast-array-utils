# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import partial, singledispatch
from typing import TYPE_CHECKING, overload

import numpy as np

from .. import types


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import ArrayLike, DTypeLike, NDArray


@overload
def sum(
    x: ArrayLike, /, *, axis: None = None, dtype: DTypeLike | None = None
) -> np.number[Any]: ...
@overload
def sum(
    x: ArrayLike, /, *, axis: Literal[0, 1], dtype: DTypeLike | None = None
) -> NDArray[Any]: ...
@overload
def sum(
    x: types.DaskArray, /, *, axis: Literal[0, 1] | None = None, dtype: DTypeLike | None = None
) -> types.DaskArray: ...


def sum(
    x: ArrayLike, /, *, axis: Literal[0, 1, None] = None, dtype: DTypeLike | None = None
) -> NDArray[Any] | np.number[Any] | types.DaskArray:
    """Sum over both or one axis.

    Returns
    -------
    If ``axis`` is :data:`None`, then the sum over all elements is returned as a scalar.
    Otherwise, the sum over the given axis is returned as a 1D array.

    See Also
    --------
    :func:`numpy.sum`

    """
    return _sum(x, axis=axis, dtype=dtype)


@singledispatch
def _sum(
    x: ArrayLike | types.CSBase | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
) -> NDArray[Any] | np.number[Any] | types.DaskArray:
    assert not isinstance(x, types.CSBase | types.DaskArray)
    return np.sum(x, axis=axis, dtype=dtype)  # type: ignore[no-any-return]


@_sum.register(types.CSBase)
def _(
    x: types.CSBase, /, *, axis: Literal[0, 1, None] = None, dtype: DTypeLike | None = None
) -> NDArray[Any] | np.number[Any]:
    import scipy.sparse as sp

    if isinstance(x, types.CSMatrix):
        x = sp.csr_array(x) if x.format == "csr" else sp.csc_array(x)
    return np.sum(x, axis=axis, dtype=dtype)  # type: ignore[no-any-return]


@_sum.register(types.DaskArray)
def _(
    x: types.DaskArray, /, *, axis: Literal[0, 1, None] = None, dtype: DTypeLike | None = None
) -> types.DaskArray:
    if TYPE_CHECKING:
        from dask.array.reductions import reduction
    else:
        from dask.array import reduction

    if isinstance(x._meta, np.matrix):  # pragma: no cover  # noqa: SLF001
        msg = "sum does not support numpy matrices"
        raise TypeError(msg)

    def sum_drop_keepdims(
        a: NDArray[Any] | types.CSBase,
        /,
        *,
        axis: tuple[Literal[0], Literal[1]] | Literal[0, 1] | None = None,
        dtype: DTypeLike | None = None,
        keepdims: bool = False,
    ) -> NDArray[Any]:
        del keepdims
        match axis:
            case (0 | 1 as n,):
                axis = n
            case (0, 1) | (1, 0):
                axis = None
            case tuple():  # pragma: no cover
                msg = f"`sum` can only sum over `axis=0|1|(0,1)` but got {axis} instead"
                raise ValueError(msg)
        rv = sum(a, axis=axis, dtype=dtype)
        rv = np.array(rv, ndmin=1)  # make sure rv is at least 1D
        return rv.reshape((1, len(rv)))

    if dtype is None:
        # Explicitly use numpy result dtype (e.g. `NDArray[bool].sum().dtype == int64`)
        dtype = np.zeros(1, dtype=x.dtype).sum().dtype

    return reduction(  # type: ignore[no-any-return,no-untyped-call]
        x,
        sum_drop_keepdims,
        partial(np.sum, dtype=dtype),
        axis=axis,
        dtype=dtype,
        meta=np.array([], dtype=dtype),
    )
