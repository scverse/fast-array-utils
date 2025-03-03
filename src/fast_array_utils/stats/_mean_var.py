# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numba
import numpy as np

from .. import types
from ._mean import mean
from ._power import power


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import NDArray

    MemArray = NDArray[Any] | types.CSBase | types.CupyArray | types.CupySparseMatrix

__all__ = ["mean_var"]


@overload
def mean_var(
    x: MemArray, /, *, axis: Literal[None] = None, correction: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
@overload
def mean_var(
    x: MemArray, /, *, axis: Literal[0, 1], correction: int = 0
) -> tuple[np.float64, np.float64]: ...
@overload
def mean_var(
    x: types.DaskArray, /, *, axis: Literal[0, 1, None] = None, correction: int = 0
) -> tuple[types.DaskArray, types.DaskArray]: ...


def mean_var(
    x: MemArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    correction: int = 0,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[np.float64, np.float64]
    | tuple[types.DaskArray, types.DaskArray]
):
    if axis is not None and isinstance(x, types.CSBase):
        mean_, var = _sparse_mean_var(x, axis=axis)
    else:
        mean_ = mean(x, axis=axis, dtype=np.float64)
        mean_sq = mean(power(x, x), axis=axis, dtype=np.float64)
        var = mean_sq - mean_**2
    if correction:  # R convention == 1 (unbiased estimator)
        n = np.prod(x.shape) if axis is None else x.shape[axis]
        if n != 1:
            var *= n / (n - correction)
    return mean_, var


def _sparse_mean_var(
    mtx: types.CSBase, /, *, axis: Literal[0, 1]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate means and variances for each row or column of a sparse matrix.

    This code and internal functions are based on sklearns `sparsefuncs.mean_variance_axis`.

    Modifications:
    - allow deciding on the output type,
      which can increase accuracy when calculating the mean and variance of 32bit floats.
    - Doesn't currently implement support for null values, but could.
    - Uses numba instead of Cython
    """
    assert axis in (0, 1)
    if mtx.format == "csr":
        ax_minor = 1
        shape = mtx.shape
    elif mtx.format == "csc":
        ax_minor = 0
        shape = mtx.shape[::-1]
    else:
        msg = "This function only works on sparse csr and csc matrices"
        raise TypeError(msg)
    if len(shape) == 1:
        msg = "array must have 2 dimensions"
        raise TypeError(msg)
    f = sparse_mean_var_major_axis if axis == ax_minor else sparse_mean_var_minor_axis
    return f(
        mtx.data,
        mtx.indptr,
        mtx.indices,
        major_len=shape[0],
        minor_len=shape[1],
        n_threads=numba.get_num_threads(),
    )


@numba.njit
def sparse_mean_var_minor_axis(
    data: NDArray[np.number[Any]],
    indptr: NDArray[np.integer[Any]],
    indices: NDArray[np.integer[Any]],
    *,
    major_len: int,
    minor_len: int,
    n_threads: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute mean and variance along the minor axis of a compressed sparse matrix."""
    rows = len(indptr) - 1
    sums = np.zeros((n_threads, minor_len))
    squared_sums = np.zeros((n_threads, minor_len))
    means = np.zeros(minor_len)
    variances = np.zeros(minor_len)
    for i in numba.prange(n_threads):
        for r in range(i, rows, n_threads):
            for j in range(indptr[r], indptr[r + 1]):
                minor_index = indices[j]
                if minor_index >= minor_len:
                    continue
                value = data[j]
                sums[i, minor_index] += value
                squared_sums[i, minor_index] += value * value
    for c in numba.prange(minor_len):
        sum = sums[:, c].sum()
        means[c] = sum / major_len
        variances[c] = squared_sums[:, c].sum() / major_len - (sum / major_len) ** 2
    return means, variances


@numba.njit
def sparse_mean_var_major_axis(
    data: NDArray[np.number[Any]],
    indptr: NDArray[np.integer[Any]],
    indices: NDArray[np.integer[Any]],  # noqa: ARG001
    *,
    major_len: int,
    minor_len: int,
    n_threads: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute means and variances along the major axis of a compressed sparse matrix."""
    rows = len(indptr) - 1
    means = np.zeros(major_len)
    variances = np.zeros_like(means)

    for i in numba.prange(n_threads):
        for r in range(i, rows, n_threads):
            sum_major = np.float64(0.0)
            squared_sum_minor = np.float64(0.0)
            for j in range(indptr[r], indptr[r + 1]):
                value = np.float64(data[j])
                sum_major += value
                squared_sum_minor += value * value
            means[r] = sum_major
            variances[r] = squared_sum_minor
    for c in numba.prange(major_len):
        mean = means[c] / minor_len
        means[c] = mean
        variances[c] = variances[c] / minor_len - mean * mean
    return means, variances
