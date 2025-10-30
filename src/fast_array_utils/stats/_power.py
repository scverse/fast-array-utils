# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np

from .. import types


if TYPE_CHECKING:
    from typing import TypeAlias, TypeVar

    from numpy.typing import DTypeLike

    from fast_array_utils.typing import CpuArray, GpuArray

    # All supported array types except for disk ones and CSDataset
    Array: TypeAlias = CpuArray | GpuArray | types.DaskArray

    _Arr = TypeVar("_Arr", bound=Array)
    _Mat = TypeVar("_Mat", bound=types.CSBase | types.CupyCSMatrix)


def power(x: _Arr, n: int, /, dtype: DTypeLike | None = None) -> _Arr:
    """Take array or matrix to a power."""
    # This wrapper is necessary because TypeVars can’t be used in `singledispatch` functions
    return _power(x, n, dtype=dtype)  # type: ignore[return-value]


@singledispatch
def _power(x: Array, n: int, /, dtype: DTypeLike | None = None) -> Array:
    if TYPE_CHECKING:
        assert not isinstance(x, types.DaskArray | types.CSBase | types.CupyCSMatrix)
    return x**n if dtype is None else np.power(x, n, dtype=dtype)  # type: ignore[operator]


@_power.register(types.CSBase | types.CupyCSMatrix)
def _power_cs(x: _Mat, n: int, /, dtype: DTypeLike | None = None) -> _Mat:
    new_data = power(x.data, n, dtype=dtype)
    return type(x)((new_data, x.indices, x.indptr), shape=x.shape, dtype=new_data.dtype)  # type: ignore[call-overload,return-value]


@_power.register(types.DaskArray)
def _power_dask(x: types.DaskArray, n: int, /, dtype: DTypeLike | None = None) -> types.DaskArray:
    meta = x._meta.astype(dtype or x.dtype)  # noqa: SLF001
    return x.map_blocks(lambda c: power(c, n, dtype=dtype), dtype=dtype, meta=meta)  # type: ignore[type-var,arg-type]
