# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._sum import sum as sum_


if TYPE_CHECKING:
    from typing import Any, Literal, TypeAlias

    from numpy._typing._array_like import _ArrayLikeFloat_co
    from numpy.typing import DTypeLike, NDArray

    from .. import types

    # all supported types except OutOfCoreDataset (TODO)
    Array: TypeAlias = (
        NDArray[Any]
        | types.CSBase
        | types.H5Dataset
        | types.ZarrArray
        | types.CupyArray
        | types.CupySparseMatrix
        | types.DaskArray
    )


def mean(
    x: _ArrayLikeFloat_co | Array,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
) -> NDArray[Any] | types.DaskArray:
    if not hasattr(x, "shape"):
        raise NotImplementedError  # TODO(flying-sheep): infer shape  # noqa: TD003
    if TYPE_CHECKING:
        assert isinstance(x, Array)
    total = sum_(x, axis=axis, dtype=dtype)
    n = np.prod(x.shape) if axis is None else x.shape[axis]
    return total / n
