# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, cast

import numpy as np

from .. import types
from ._utils import _dask_inner


if TYPE_CHECKING:
    from typing import Any, Literal, TypeAlias

    from numpy.typing import NDArray

    from ..typing import CpuArray, DiskArray, GpuArray

    ComplexAxis: TypeAlias = tuple[Literal[0], Literal[1]] | tuple[Literal[0, 1]] | Literal[0, 1, None]

    MinMaxOps = Literal["max", "min"]


@singledispatch
def min_max(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    op: MinMaxOps,
    *,
    axis: Literal[0, 1, None] = None,
    keep_cupy_as_array: bool = False,
) -> NDArray[Any] | np.number[Any] | types.CupyArray | types.DaskArray:
    del keep_cupy_as_array
    if TYPE_CHECKING:
        # these are never passed to this fallback function, but `singledispatch` wants them
        assert not isinstance(x, types.CSBase | types.DaskArray | types.CupyArray | types.CupyCSMatrix)
        # np supports these, but doesn’t know it. (TODO: test cupy)
        assert not isinstance(x, types.ZarrArray | types.H5Dataset)
    return cast("NDArray[Any] | np.number[Any]", getattr(np, op)(x, axis=axis))


@min_max.register(types.CupyArray | types.CupyCSMatrix)
def _min_max_cupy(
    x: GpuArray,
    /,
    op: MinMaxOps,
    *,
    axis: Literal[0, 1, None] = None,
    keep_cupy_as_array: bool = False,
) -> types.CupyArray | np.number[Any]:
    arr = cast("types.CupyArray", getattr(np, op)(x, axis=axis))
    return cast("np.number[Any]", arr.get()[()]) if not keep_cupy_as_array and axis is None else arr.squeeze()


@min_max.register(types.CSBase)
def _min_max_cs(
    x: types.CSBase,
    /,
    op: MinMaxOps,
    *,
    axis: Literal[0, 1, None] = None,
    keep_cupy_as_array: bool = False,
) -> NDArray[Any] | np.number[Any]:
    del keep_cupy_as_array
    import scipy.sparse as sp

    if isinstance(x, types.CSMatrix):
        x = sp.csr_array(x) if x.format == "csr" else sp.csc_array(x)

    if axis is None:
        return cast("np.number[Any]", getattr(sp, op)(x))
    return cast("NDArray[Any] | np.number[Any]", getattr(sp, op)(x, axis=axis))


@min_max.register(types.DaskArray)
def _min_max_dask(
    x: types.DaskArray,
    /,
    op: MinMaxOps,
    *,
    axis: Literal[0, 1, None] = None,
    keep_cupy_as_array: bool = False,
) -> types.DaskArray:
    from . import max, min

    fns = {fn.__name__: fn for fn in (min, max)}
    return _dask_inner(x, fns[op], axis=axis, keep_cupy_as_array=keep_cupy_as_array)
