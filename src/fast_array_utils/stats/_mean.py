# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import sum


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import DTypeLike, NDArray

    from .. import types
    from ..typing import CpuArray, DiskArray, GpuArray


def mean_(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1] | None = None,
    dtype: DTypeLike | None = None,
) -> NDArray[np.number[Any]] | np.number[Any] | types.DaskArray:
    total = sum(x, axis=axis, dtype=dtype)  # type: ignore[misc,arg-type]
    n = np.prod(x.shape) if axis is None else x.shape[axis]
    return total / n  # type: ignore[no-any-return]
