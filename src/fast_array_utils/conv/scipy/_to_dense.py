# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numba


if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from numpy.typing import NDArray

    from fast_array_utils.types import CSBase


__all__ = ["_to_dense_csc_numba", "_to_dense_csr_numba"]


@numba.njit(cache=True)
def _to_dense_csc_numba(x: CSBase, out: NDArray[np.number[Any]]) -> None:
    for c in numba.prange(out.shape[1]):
        for i in range(x.indptr[c], x.indptr[c + 1]):
            out[x.indices[i], c] = x.data[i]


@numba.njit(cache=True)
def _to_dense_csr_numba(x: CSBase, out: NDArray[np.number[Any]]) -> None:
    for r in numba.prange(out.shape[0]):
        for i in range(x.indptr[r], x.indptr[r + 1]):
            out[r, x.indices[i]] = x.data[i]
