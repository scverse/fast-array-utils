# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, cast

from .. import types


if TYPE_CHECKING:
    from typing import Any, TypeVar

    from numpy.typing import NDArray

    # All supported array types except for disk ones and OutOfCoreDataset
    Array = NDArray[Any] | types.CSBase | types.CupyArray | types.CupySparseMatrix | types.DaskArray

    _Arr = TypeVar("_Arr", bound=Array)


def power(x: _Arr, n: int, /) -> _Arr:
    """Take array or matrix to a power."""
    return _power(x, n)  # type: ignore[return-value]


@singledispatch
def _power(x: Array, n: int, /) -> Array:
    if TYPE_CHECKING:
        assert not isinstance(x, types.DaskArray | types.CSMatrix)
    return x**n  # type: ignore[operator]


@_power.register(types.CSMatrix)  # type: ignore[call-overload,misc]
def _power_cs(x: types.CSMatrix, n: int, /) -> types.CSMatrix:
    return x.power(n)


@_power.register(types.DaskArray)
def _power_dask(x: types.DaskArray, n: int, /) -> types.DaskArray:
    import dask.array as da

    return cast(types.DaskArray, da.map_blocks(power, x, n))  # type: ignore[no-untyped-call]
