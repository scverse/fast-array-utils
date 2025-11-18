# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, TypeVar, cast, get_args

import numpy as np
from numpy.exceptions import AxisError

from .. import types
from ..typing import GpuArray
from ._typing import DtypeOps


if TYPE_CHECKING:
    from typing import Any, Literal, TypeAlias

    from numpy.typing import DTypeLike, NDArray

    from ..typing import CpuArray
    from ._typing import DTypeKw, Ops

    ComplexAxis: TypeAlias = tuple[Literal[0], Literal[1]] | tuple[Literal[0, 1]] | Literal[0, 1] | None


__all__ = ["_dask_inner"]


def _dask_inner(x: types.DaskArray, op: Ops, /, *, axis: Literal[0, 1] | None, dtype: DTypeLike | None = None, keep_cupy_as_array: bool) -> types.DaskArray:
    import dask.array as da

    if isinstance(x._meta, np.matrix):  # pragma: no cover  # noqa: SLF001
        msg = "sum/max/min does not support numpy matrices"
        raise TypeError(msg)

    res_dtype = dtype if op in get_args(DtypeOps) else x.dtype

    rv = da.reduction(
        x,
        partial(_dask_block, op, dtype=dtype),
        partial(_dask_block, op, dtype=dtype),
        axis=axis,
        dtype=res_dtype,
        meta=np.array([], dtype=res_dtype),
    )

    if axis is not None or (
        isinstance(rv._meta, types.CupyArray)  # noqa: SLF001
        and keep_cupy_as_array
    ):
        return rv

    def to_scalar(a: types.CupyArray | NDArray[Any]) -> np.number[Any]:
        if isinstance(a, types.CupyArray):
            a = a.get()
        return a.reshape(())[()]  # type: ignore[return-value]

    return rv.map_blocks(to_scalar, meta=x.dtype.type(0))  # type: ignore[arg-type]


def _dask_block(
    op: Ops,
    a: CpuArray | GpuArray,
    /,
    *,
    axis: ComplexAxis = None,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    computing_meta: bool = False,
) -> NDArray[Any] | types.CupyArray:
    from . import max, min, sum

    if computing_meta:  # dask.blockwise doesn’t allow to pass `meta` in, and reductions below don’t handle a 0d matrix
        return (types.CupyArray if isinstance(a, GpuArray) else np.ndarray)((), dtype or a.dtype)

    fns = {fn.__name__: fn for fn in (min, max, sum)}

    axis = _normalize_axis(axis, a.ndim)
    rv = fns[op](a, axis=axis, keep_cupy_as_array=True, **_dtype_kw(dtype, op))  # type: ignore[call-overload]
    shape = _get_shape(rv, axis=axis, keepdims=keepdims)
    return cast("NDArray[Any] | types.CupyArray", rv.reshape(shape))


def _normalize_axis(axis: ComplexAxis, ndim: int) -> Literal[0, 1] | None:
    """Adapt `axis` parameter passed by Dask to what we support."""
    match axis:
        case int() | None:  # pragma: no cover
            pass
        case (0 | 1,):
            axis = axis[0]
        case (0, 1) | (1, 0):
            axis = None
        case _:  # pragma: no cover
            raise AxisError(axis, ndim)  # type: ignore[call-overload]
    if axis == 0 and ndim == 1:
        return None  # dask’s aggregate doesn’t know we don’t accept `axis=0` for 1D arrays
    return axis


def _get_shape(a: NDArray[Any] | np.number[Any] | types.CupyArray, *, axis: Literal[0, 1] | None, keepdims: bool) -> tuple[int] | tuple[int, int]:
    """Get the output shape of an axis-flattening operation."""
    match keepdims, a.ndim:
        case False, 0:
            return (1,)
        case True, 0:
            return (1, 1)
        case False, 1:
            return (a.size,)
        case True, 1:
            assert axis is not None
            return (1, a.size) if axis == 0 else (a.size, 1)
        case _:  # pragma: no cover
            msg = f"{keepdims=}, {a.ndim=}, {type(a)=}"
            raise AssertionError(msg)


DT = TypeVar("DT", bound="DTypeLike")


def _dtype_kw(dtype: DT | None, op: Ops) -> DTypeKw[DT]:
    return {"dtype": dtype} if dtype is not None and op in get_args(DtypeOps) else {}
