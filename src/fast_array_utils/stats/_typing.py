# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, overload

import numpy as np

from fast_array_utils import types

from ..typing import CpuArray, DiskArray, GpuArray


if TYPE_CHECKING:
    from typing import Any, TypeAlias

    from numpy.typing import NDArray


Array: TypeAlias = CpuArray | GpuArray | DiskArray | types.CSDataset | types.DaskArray

DTypeIn: TypeAlias = np.float32 | np.float64 | np.int32 | np.bool_
DTypeOut: TypeAlias = np.float32 | np.float64 | np.int64

NdAndAx: TypeAlias = tuple[Literal[1], Literal[None]] | tuple[Literal[2], Literal[0, 1, None]]


class StatFun(Protocol):
    __name__: str

    @overload
    def __call__(self, x: CpuArray | DiskArray, /, *, axis: None = None, keep_cupy_as_array: bool = False) -> np.number[Any]: ...
    @overload
    def __call__(self, x: CpuArray | DiskArray, /, *, axis: Literal[0, 1], keep_cupy_as_array: bool = False) -> NDArray[Any]: ...

    @overload
    def __call__(self, x: GpuArray, /, *, axis: None = None, keep_cupy_as_array: Literal[False] = False) -> np.number[Any]: ...
    @overload
    def __call__(self, x: GpuArray, /, *, axis: None, keep_cupy_as_array: Literal[True]) -> types.CupyArray: ...
    @overload
    def __call__(self, x: GpuArray, /, *, axis: Literal[0, 1], keep_cupy_as_array: bool = False) -> types.CupyArray: ...

    @overload
    def __call__(self, x: types.DaskArray, /, *, axis: Literal[0, 1, None] = None, keep_cupy_as_array: bool = False) -> types.DaskArray: ...
