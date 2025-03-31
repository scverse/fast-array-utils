# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray
    from scipy.sparse import coo_array, coo_matrix

    from ... import types


__all__ = ["to_dense"]


def to_dense(x: types.spmatrix | types.sparray, order: Literal["C", "F"] = "C") -> NDArray[Any]:
    """Convert a sparse matrix to a dense matrix.

    Parameters
    ----------
    x
        Input matrix.
    order
        The order of the output matrix.

    Returns
    -------
    Dense matrix form of ``x``

    """
    if TYPE_CHECKING:
        assert isinstance(x, types.CSBase | coo_matrix | coo_array)

    out = np.zeros(x.shape, dtype=x.dtype, order=order)
    if x.format == "csr":
        _to_dense_csr_numba(x.indptr, x.indices, x.data, out)
    elif x.format == "csc":
        _to_dense_csc_numba(x.indptr, x.indices, x.data, out)
    else:  # pragma: no cover
        out = x.toarray(order=order)
    return out


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
