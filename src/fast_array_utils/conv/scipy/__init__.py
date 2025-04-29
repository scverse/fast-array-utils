# SPDX-License-Identifier: MPL-2.0
"""Utilities only for sparse matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

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

    try:
        from ._to_dense import _to_dense_csc_numba, _to_dense_csr_numba
    except ImportError:
        warn("numba is not installed; falling back to slow conversion", RuntimeWarning, stacklevel=2)
        return x.toarray(order=order)

    out = np.zeros(x.shape, dtype=x.dtype, order=order)
    if x.format == "csr":
        _to_dense_csr_numba(x, out)
    elif x.format == "csc":
        _to_dense_csc_numba(x, out)
    else:  # pragma: no cover
        out = x.toarray(order=order)
    return out
