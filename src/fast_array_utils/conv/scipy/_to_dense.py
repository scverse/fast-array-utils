# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numba


if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from numpy.typing import NDArray


__all__ = ["_to_dense_csc_numba", "_to_dense_csr_numba"]


@numba.njit(cache=True)
def _to_dense_csc_numba(
    indptr: NDArray[np.integer[Any]],
    indices: NDArray[np.integer[Any]],
    data: NDArray[np.number[Any]],
    x: NDArray[np.number[Any]],
) -> None:
    for c in numba.prange(x.shape[1]):
        for i in range(indptr[c], indptr[c + 1]):
            x[indices[i], c] = data[i]


@numba.njit(cache=True)
def _to_dense_csr_numba(
    indptr: NDArray[np.integer[Any]],
    indices: NDArray[np.integer[Any]],
    data: NDArray[np.number[Any]],
    x: NDArray[np.number[Any]],
) -> None:
    for r in numba.prange(x.shape[0]):
        for i in range(indptr[r], indptr[r + 1]):
            x[r, indices[i]] = data[i]
