# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

from .. import types


if TYPE_CHECKING:
    from typing import TypeAlias, TypeVar

    from numpy.typing import DTypeLike

    from fast_array_utils.typing import CpuArray, GpuArray

    # All supported array types except for disk ones and CSDataset
    Array: TypeAlias = CpuArray | GpuArray | types.DaskArray

    _Arr = TypeVar("_Arr", bound=Array)


def power(x: _Arr, n: int, /, dtype: DTypeLike | None = None) -> _Arr:
    """Take array or matrix to a power."""
    # This wrapper is necessary because TypeVars can’t be used in `singledispatch` functions
    return _power(x, n, dtype=dtype)  # type: ignore[return-value]


@singledispatch
def _power(x: Array, n: int, /, dtype: DTypeLike | None = None) -> Array:
    if TYPE_CHECKING:
        assert not isinstance(x, types.DaskArray | types.CSMatrix)
    if dtype is not None:
        x = x.astype(dtype, copy=False)  # type: ignore[assignment]
    return x**n  # type: ignore[operator]


@_power.register(types.CSMatrix | types.CupyCSMatrix)
def _power_cs(x: types.CSMatrix | types.CupyCSMatrix, n: int, /, dtype: DTypeLike | None = None) -> types.CSMatrix | types.CupyCSMatrix:
    if dtype is not None:
        x = x.astype(dtype, copy=False)  # type: ignore[assignment]
    return x.power(n)


@_power.register(types.DaskArray)
def _power_dask(x: types.DaskArray, n: int, /, dtype: DTypeLike | None = None) -> types.DaskArray:
    return x.map_blocks(lambda c: power(c, n, dtype=dtype), dtype=dtype)  # type: ignore[type-var]
