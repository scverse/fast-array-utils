# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

import numpy as np

from fast_array_utils import types

from ..typing import CpuArray, DiskArray, GpuArray


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import DTypeLike, NDArray


type Array = CpuArray | GpuArray | DiskArray | types.CSDataset | types.DaskArray

type DTypeIn = np.float32 | np.float64 | np.int32 | np.bool_
type DTypeOut = np.float32 | np.float64 | np.int64

type NdAndAx = tuple[Literal[1], None] | tuple[Literal[2], Literal[0, 1] | None]


class StatFunNoDtype(Protocol):
    __name__: str

    def __call__(
        self, x: CpuArray | GpuArray | DiskArray | types.DaskArray, /, *, axis: Literal[0, 1] | None = None, keep_cupy_as_array: bool = False
    ) -> types.DaskArray: ...


class StatFunDtype(Protocol):
    __name__: str

    def __call__(
        self,
        x: CpuArray | GpuArray | DiskArray | types.DaskArray,
        /,
        *,
        axis: Literal[0, 1] | None = None,
        dtype: DTypeLike | None = None,
        keep_cupy_as_array: bool = False,
    ) -> NDArray[Any] | types.CupyArray | np.number[Any] | types.DaskArray: ...


NoDtypeOps = Literal["max", "min"]
DtypeOps = Literal["sum"]
type Ops = NoDtypeOps | DtypeOps


class DTypeKw[DT: DTypeLike](TypedDict, total=False):
    dtype: DT
