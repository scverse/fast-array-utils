# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from .. import types
from ._utils import _dask_inner


if TYPE_CHECKING:
    from typing import Any, Literal, TypeAlias

    from numpy.typing import DTypeLike, NDArray

    from ..typing import CpuArray, DiskArray, GpuArray

    ComplexAxis: TypeAlias = tuple[Literal[0], Literal[1]] | tuple[Literal[0, 1]] | Literal[0, 1, None]


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
        assert not isinstance(x, types.CSBase | types.DaskArray | types.CupyArray | types.CupyCSMatrix)
        # np.sum supports these, but doesn’t know it. (TODO: test cupy)
        assert not isinstance(x, types.ZarrArray | types.H5Dataset)
    return cast("NDArray[Any] | np.number[Any]", np.sum(x, axis=axis, dtype=dtype))


@sum_.register(types.CupyArray | types.CupyCSMatrix)
def _sum_cupy(
    x: GpuArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
    keep_cupy_as_array: bool = False,
) -> types.CupyArray | np.number[Any]:
    arr = cast("types.CupyArray", np.sum(x, axis=axis, dtype=dtype))
    return cast("np.number[Any]", arr.get()[()]) if not keep_cupy_as_array and axis is None else arr.squeeze()


@sum_.register(types.CSBase)
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
    from . import sum

    if dtype is None:
        # Explicitly use numpy result dtype (e.g. `NDArray[bool].sum().dtype == int64`)
        dtype = np.zeros(1, dtype=x.dtype).sum().dtype

    return _dask_inner(x, sum, axis=axis, dtype=dtype, keep_cupy_as_array=keep_cupy_as_array)
