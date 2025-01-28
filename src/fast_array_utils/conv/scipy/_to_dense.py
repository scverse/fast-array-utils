# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np


if TYPE_CHECKING:
    from typing import Literal, TypeVar

    from numpy.typing import NDArray

    from ...types import CSBase

    DT_co = TypeVar("DT_co", bound=np.generic, covariant=True)
    DT = TypeVar("DT", bound=np.generic)


__all__ = ["to_dense"]


def to_dense(x: CSBase[DT_co], order: Literal["C", "F"] = "C") -> NDArray[DT_co]:
    """Convert a compressed sparse matrix to a dense matrix.

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
    indptr: NDArray[DT],
    indices: NDArray[DT],
    data: NDArray[DT_co],
    x: NDArray[DT_co],
) -> None:
    for c in numba.prange(x.shape[1]):
        for i in range(indptr[c], indptr[c + 1]):
            x[indices[i], c] = data[i]


@numba.njit(cache=True)
def _to_dense_csr_numba(
    indptr: NDArray[DT],
    indices: NDArray[DT],
    data: NDArray[DT_co],
    x: NDArray[DT_co],
) -> None:
    for r in numba.prange(x.shape[0]):
        for i in range(indptr[r], indptr[r + 1]):
            x[r, indices[i]] = data[i]
