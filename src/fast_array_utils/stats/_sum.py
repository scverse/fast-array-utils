# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import partial, singledispatch
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.exceptions import AxisError

from .. import types


if TYPE_CHECKING:
    from typing import Any, Literal, TypeAlias

    from numpy.typing import DTypeLike, NDArray

    from ..typing import CpuArray, DiskArray, GpuArray

    ComplexAxis: TypeAlias = (
        tuple[Literal[0], Literal[1]] | tuple[Literal[0, 1]] | Literal[0, 1, None]
    )


@singledispatch
def sum_(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
    keep_cupy_as_array: bool = False,
) -> NDArray[Any] | np.number[Any] | types.CupyArray | types.DaskArray:
    del keep_cupy_as_array
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
    x: GpuArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
    keep_cupy_as_array: bool = False,
) -> types.CupyArray | np.number[Any]:
    arr = cast("types.CupyArray", np.sum(x, axis=axis, dtype=dtype))
    return (
        cast("np.number[Any]", arr.get()[()])
        if not keep_cupy_as_array and axis is None
        else arr.squeeze()
    )


@sum_.register(types.CSBase)  # type: ignore[call-overload,misc]
def _sum_cs(
    x: types.CSBase,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
    keep_cupy_as_array: bool = False,
) -> NDArray[Any] | np.number[Any]:
    del keep_cupy_as_array
    import scipy.sparse as sp

    if isinstance(x, types.CSMatrix):
        x = sp.csr_array(x) if x.format == "csr" else sp.csc_array(x)

    if axis is None:
        return cast("np.number[Any]", x.data.sum(dtype=dtype))
    return cast("NDArray[Any] | np.number[Any]", x.sum(axis=axis, dtype=dtype))


@sum_.register(types.DaskArray)
def _sum_dask(
    x: types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
    keep_cupy_as_array: bool = False,
) -> types.DaskArray:
    import dask.array as da

    if isinstance(x._meta, np.matrix):  # pragma: no cover  # noqa: SLF001
        msg = "sum does not support numpy matrices"
        raise TypeError(msg)

    if dtype is None:
        # Explicitly use numpy result dtype (e.g. `NDArray[bool].sum().dtype == int64`)
        dtype = np.zeros(1, dtype=x.dtype).sum().dtype

    rv = da.reduction(
        x,
        sum_dask_inner,  # type: ignore[arg-type]
        partial(sum_dask_inner, dtype=dtype),  # pyright: ignore[reportArgumentType]
        axis=axis,
        dtype=dtype,
        meta=np.array([], dtype=dtype),
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


def sum_dask_inner(
    a: CpuArray | GpuArray,
    /,
    *,
    axis: ComplexAxis = None,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | types.CupyArray:
    from . import sum

    axis = normalize_axis(axis, a.ndim)
    rv = sum(a, axis=axis, dtype=dtype, keep_cupy_as_array=True)  # type: ignore[misc,arg-type]
    shape = get_shape(rv, axis=axis, keepdims=keepdims)
    return cast("NDArray[Any] | types.CupyArray", rv.reshape(shape))


def normalize_axis(axis: ComplexAxis, ndim: int) -> Literal[0, 1, None]:
    """Adapt `axis` parameter passed by Dask to what we support."""
    match axis:
        case int() | None:
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


def get_shape(
    a: NDArray[Any] | np.number[Any] | types.CupyArray, *, axis: Literal[0, 1, None], keepdims: bool
) -> tuple[int] | tuple[int, int]:
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
    # pragma: no cover
    msg = f"{keepdims=}, {type(a)}"
    raise AssertionError(msg)
