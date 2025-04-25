# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import partial, singledispatch
from typing import TYPE_CHECKING, cast

import numpy as np

from .. import types


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import DTypeLike, NDArray

    from ..typing import CpuArray, DiskArray, GpuArray


@singledispatch
def sum_(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
) -> NDArray[Any] | np.number[Any] | types.CupyArray | types.DaskArray:
    if TYPE_CHECKING:
        # these are never passed to this fallback function, but `singledispatch` wants them
        assert not isinstance(
            x, types.CSBase | types.DaskArray | types.CupyArray | types.CupyCSMatrix
        )
        # np.sum supports these, but doesn’t know it. (TODO: test cupy)
        assert not isinstance(x, types.ZarrArray | types.H5Dataset)
    return cast("NDArray[Any] | np.number[Any]", np.sum(x, axis=axis, dtype=dtype))


@sum_.register(types.CupyArray | types.CupyCSMatrix)  # type: ignore[call-overload,misc]
def _sum_cupy(
    x: GpuArray, /, *, axis: Literal[0, 1, None] = None, dtype: DTypeLike | None = None
) -> types.CupyArray | np.number[Any]:
    arr = cast("types.CupyArray", np.sum(x, axis=axis, dtype=dtype))
    return cast("np.number[Any]", arr.get()[()]) if axis is None else arr.squeeze()


@sum_.register(types.CSBase)  # type: ignore[call-overload,misc]
def _sum_cs(
    x: types.CSBase, /, *, axis: Literal[0, 1, None] = None, dtype: DTypeLike | None = None
) -> NDArray[Any] | np.number[Any]:
    import scipy.sparse as sp

    if isinstance(x, types.CSMatrix):
        x = sp.csr_array(x) if x.format == "csr" else sp.csc_array(x)

    if axis is None:
        return cast("np.number[Any]", x.data.sum(dtype=dtype))
    return cast("NDArray[Any] | np.number[Any]", x.sum(axis=axis, dtype=dtype))


@sum_.register(types.DaskArray)
def _sum_dask(
    x: types.DaskArray, /, *, axis: Literal[0, 1, None] = None, dtype: DTypeLike | None = None
) -> types.DaskArray:
    import dask.array as da

    from . import sum

    if isinstance(x._meta, np.matrix):  # pragma: no cover  # noqa: SLF001
        msg = "sum does not support numpy matrices"
        raise TypeError(msg)

    def sum_drop_keepdims(
        a: CpuArray,
        /,
        *,
        axis: tuple[Literal[0], Literal[1]] | Literal[0, 1, None] = None,
        dtype: DTypeLike | None = None,
        keepdims: bool = False,
    ) -> NDArray[Any] | types.CupyArray:
        del keepdims
        if a.ndim == 1:
            axis = None
        else:
            match axis:
                case (0, 1) | (1, 0):
                    axis = None
                case (0 | 1 as n,):
                    axis = n
                case tuple():  # pragma: no cover
                    msg = f"`sum` can only sum over `axis=0|1|(0,1)` but got {axis} instead"
                    raise ValueError(msg)
        rv = sum(a, axis=axis, dtype=dtype)
        shape = (1,) if a.ndim == 1 else (1, 1 if rv.shape == () else len(rv))  # type: ignore[arg-type]
        return np.reshape(rv, shape)

    if dtype is None:
        # Explicitly use numpy result dtype (e.g. `NDArray[bool].sum().dtype == int64`)
        dtype = np.zeros(1, dtype=x.dtype).sum().dtype

    return da.reduction(
        x,
        sum_drop_keepdims,  # type: ignore[arg-type]
        partial(np.sum, dtype=dtype),  # pyright: ignore[reportArgumentType]
        axis=axis,
        dtype=dtype,
        meta=np.array([], dtype=dtype),
    )
