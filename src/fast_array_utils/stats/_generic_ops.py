# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, cast, get_args

import numpy as np

from .. import types
from ._typing import DtypeOps
from ._utils import _dask_inner, _dtype_kw


if TYPE_CHECKING:
    from typing import Any, Literal, TypeAlias

    from numpy.typing import DTypeLike, NDArray

    from ..typing import CpuArray, DiskArray, GpuArray
    from ._typing import Ops

    ComplexAxis: TypeAlias = tuple[Literal[0], Literal[1]] | tuple[Literal[0, 1]] | Literal[0, 1] | None


def _run_numpy_op(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    op: Ops,
    *,
    axis: Literal[0, 1] | None = None,
    dtype: DTypeLike | None = None,
) -> NDArray[Any] | np.number[Any] | types.CupyArray | types.DaskArray:
    arr = cast("NDArray[Any] | np.number[Any] | types.CupyArray | types.CupyCOOMatrix | types.DaskArray", getattr(np, op)(x, axis=axis, **_dtype_kw(dtype, op)))
    return arr.toarray() if isinstance(arr, types.CupyCOOMatrix) else arr


@singledispatch
def generic_op(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    op: Ops,
    *,
    axis: Literal[0, 1] | None = None,
    dtype: DTypeLike | None = None,
    keep_cupy_as_array: bool = False,
) -> NDArray[Any] | np.number[Any] | types.CupyArray | types.DaskArray:
    del keep_cupy_as_array
    if TYPE_CHECKING:
        # these are never passed to this fallback function, but `singledispatch` wants them
        assert not isinstance(x, types.CSBase | types.DaskArray | types.CupyArray | types.CupyCSMatrix)
        # np supports these, but doesn’t know it. (TODO: test cupy)
        assert not isinstance(x, types.ZarrArray | types.H5Dataset)
    return cast("NDArray[Any] | np.number[Any]", _run_numpy_op(x, op, axis=axis, dtype=dtype))


@generic_op.register(types.CupyArray | types.CupyCSMatrix)
def _generic_op_cupy(
    x: GpuArray,
    /,
    op: Ops,
    *,
    axis: Literal[0, 1] | None = None,
    dtype: DTypeLike | None = None,
    keep_cupy_as_array: bool = False,
) -> types.CupyArray | np.number[Any]:
    arr = cast("types.CupyArray", _run_numpy_op(x, op, axis=axis, dtype=dtype))
    return cast("np.number[Any]", arr.get()[()]) if not keep_cupy_as_array and axis is None else arr.squeeze()


@generic_op.register(types.CSBase)
def _generic_op_cs(
    x: types.CSBase,
    /,
    op: Ops,
    *,
    axis: Literal[0, 1] | None = None,
    dtype: DTypeLike | None = None,
    keep_cupy_as_array: bool = False,
) -> NDArray[Any] | np.number[Any]:
    del keep_cupy_as_array
    import scipy.sparse as sp

    # TODO(flying-sheep): once scipy fixes this issue, instead of all this,
    # just convert to sparse array, then `return x.{op}(dtype=dtype)`
    # https://github.com/scipy/scipy/issues/23768

    if axis is None:
        return cast("np.number[Any]", getattr(x.data, op)(**_dtype_kw(dtype, op)))
    if TYPE_CHECKING:  # scipy-stubs thinks e.g. "int64" is invalid, which isn’t true
        assert isinstance(dtype, np.dtype | type | None)
    # convert to array so dimensions collapse as expected
    x = (sp.csr_array if x.format == "csr" else sp.csc_array)(x, **_dtype_kw(dtype, op))  # type: ignore[arg-type]
    rv = cast("NDArray[Any] | types.coo_array | np.number[Any]", getattr(x, op)(axis=axis))
    # old scipy versions’ sparray.{max,min}() return a 1×n/n×1 sparray here, so we squeeze
    return rv.toarray().squeeze() if isinstance(rv, types.coo_array) else rv


@generic_op.register(types.DaskArray)
def _generic_op_dask(
    x: types.DaskArray,
    /,
    op: Ops,
    *,
    axis: Literal[0, 1] | None = None,
    dtype: DTypeLike | None = None,
    keep_cupy_as_array: bool = False,
) -> types.DaskArray:
    if op in get_args(DtypeOps) and dtype is None:
        # Explicitly use numpy result dtype (e.g. `NDArray[bool].sum().dtype == int64`)
        dtype = getattr(np, op)(np.zeros(1, dtype=x.dtype)).dtype

    return _dask_inner(x, op, axis=axis, dtype=dtype, keep_cupy_as_array=keep_cupy_as_array)
