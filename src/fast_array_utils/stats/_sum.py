# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import partial, singledispatch
from typing import TYPE_CHECKING

import numpy as np

from ..types import CSBase, CSMatrix, DaskArray


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import ArrayLike, DTypeLike, NDArray


# TODO(flying-sheep): overload so axis=None returns np.floating  # noqa: TD003


@singledispatch
def sum(
    x: ArrayLike,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
) -> NDArray[Any]:
    """Sum over both or one axis."""
    return np.sum(x, axis=axis, dtype=dtype)  # type: ignore[no-any-return]


@sum.register(CSBase)  # type: ignore[misc,call-overload]
def _(
    x: CSBase, *, axis: Literal[0, 1, None] = None, dtype: DTypeLike | None = None
) -> NDArray[Any]:
    import scipy.sparse as sp

    if isinstance(x, CSMatrix):
        x = sp.csr_array(x) if x.format == "csr" else sp.csc_array(x)
    return np.sum(x, axis=axis, dtype=dtype)  # type: ignore[call-overload,no-any-return]


@sum.register(DaskArray)
def _(
    x: DaskArray, *, axis: Literal[0, 1, None] = None, dtype: DTypeLike | None = None
) -> DaskArray:
    if TYPE_CHECKING:
        from dask.array.reductions import reduction
    else:
        from dask.array import reduction

    if isinstance(x._meta, np.matrix):  # pragma: no cover  # noqa: SLF001
        msg = "sum does not support numpy matrices"
        raise TypeError(msg)

    def sum_drop_keepdims(
        a: NDArray[Any] | CSBase,
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
        rv: NDArray[Any] | np.number[Any] = sum(a, axis=axis, dtype=dtype)  # type: ignore[arg-type]
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
