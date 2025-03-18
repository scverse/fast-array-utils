# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

from .. import types


if TYPE_CHECKING:
    from typing import TypeAlias, TypeVar

    from fast_array_utils.typing import CpuArray, GpuArray

    # All supported array types except for disk ones and CSDataset
    Array: TypeAlias = CpuArray | GpuArray | types.DaskArray

    _Arr = TypeVar("_Arr", bound=Array)


def power(x: _Arr, n: int, /) -> _Arr:
    """Take array or matrix to a power."""
    # This wrapper is necessary because TypeVars can’t be used in `singledispatch` functions
    return _power(x, n)  # type: ignore[return-value]


@singledispatch
def _power(x: Array, n: int, /) -> Array:
    if TYPE_CHECKING:
        assert not isinstance(x, types.DaskArray | types.CSMatrix)
    return x**n  # type: ignore[operator]


@_power.register(types.CSMatrix | types.CupyCSMatrix)  # type: ignore[call-overload,misc]
def _power_cs(
    x: types.CSMatrix | types.CupyCSMatrix, n: int, /
) -> types.CSMatrix | types.CupyCSMatrix:
    return x.power(n)


@_power.register(types.DaskArray)
def _power_dask(x: types.DaskArray, n: int, /) -> types.DaskArray:
    return x.map_blocks(lambda c: power(c, n))  # type: ignore[type-var]
