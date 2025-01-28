# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from ..types import DaskArray


if TYPE_CHECKING:
    from typing import Literal, TypeVar

    from numpy.typing import NDArray

    from ..types import CSBase, CSMatrix

    DT_co = TypeVar("DT_co", covariant=True, bound=np.generic)

    _Ax = tuple[Literal[0, 1], ...] | Literal[0, 1]


@overload
def sum(
    x: NDArray[DT_co] | CSBase[DT_co], *, axis: _Ax | None = None, dtype: None = None
) -> NDArray[DT_co]: ...


@overload
def sum(
    x: NDArray[Any] | CSBase[Any], *, axis: _Ax | None = None, dtype: np.dtype[DT_co]
) -> NDArray[DT_co]: ...


@singledispatch
def sum(
    x: NDArray[DT_co] | CSBase[DT_co],
    *,
    axis: _Ax | None = None,
    dtype: np.dtype[DT_co] | None = None,
) -> NDArray[DT_co]:
    return np.sum(np.asarray(x), axis=axis, dtype=dtype)


@sum.register(DaskArray)
def _(x: DaskArray, *, axis: _Ax | None = None, dtype: np.dtype[DT_co] | None = None) -> DaskArray:
    import dask.array as da

    # TODO(@ilan-gold): why is this so complicated?
    # https://github.com/scverse/scanpy/pull/2856/commits/feac6bc7bea69e4cc343a35855307145854a9bc8
    if dtype is None:
        dtype = getattr(np.zeros(1, dtype=x.dtype).sum(), "dtype", object)

    if isinstance(x._meta, np.ndarray) and not isinstance(x._meta, np.matrix):
        return x.sum(axis=axis, dtype=dtype)

    def sum_drop_keepdims(*args, **kwargs):
        kwargs.pop("computing_meta", None)
        # masked operations on sparse produce which numpy matrices gives the same API issues handled here
        if isinstance(x._meta, CSMatrix | np.matrix) or isinstance(args[0], CSMatrix | np.matrix):
            kwargs.pop("keepdims", None)
            axis = kwargs["axis"]
            if isinstance(axis, tuple):
                if len(axis) != 1:
                    msg = (
                        "`axis_sum` can only sum over one axis "
                        f"when `axis` arg is provided but got {axis} instead"
                    )
                    raise ValueError(msg)
                kwargs["axis"] = axis[0]
        # returns a np.matrix normally, which is undesireable
        return np.array(np.sum(*args, dtype=dtype, **kwargs))

    def aggregate_sum(*args, **kwargs):
        return np.sum(args[0], dtype=dtype, **kwargs)

    return da.reduction(
        x,
        sum_drop_keepdims,
        aggregate_sum,
        axis=axis,
        dtype=dtype,
        meta=np.array([], dtype=dtype),
    )
