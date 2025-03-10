# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

import numpy as np

from ._sum import sum_


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import DTypeLike, NDArray

    from .. import types
    from ..typing import CpuArray, DiskArray, GpuArray


@no_type_check  # mypy is very confused
def mean_(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
) -> NDArray[np.number[Any]] | np.number[Any] | types.DaskArray:
    total = sum_(x, axis=axis, dtype=dtype)
    n = np.prod(x.shape) if axis is None else x.shape[axis]
    return total / n
