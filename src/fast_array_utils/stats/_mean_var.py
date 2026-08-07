# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast, no_type_check

import numba
import numpy as np

from .. import types
from ..numba import njit
from ._power import power
from ._utils import _get_shape, _normalize_axis


if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Literal, TypedDict

    from dask.array.reductions import _Chunk
    from numpy.typing import NDArray

    from ..typing import CpuArray, GpuArray
    from ._utils import ComplexAxis

    class _Moments(TypedDict):
        """A (count, mean, M2) triple as tracked by Chan's parallel-variance algorithm.

        `mean`/`m2` are shaped like `_get_shape`'s `keepdims=True` convention.
        """

        n: int
        mean: NDArray[np.float64]
        m2: NDArray[np.float64]

    # what dask passes to `combine`/`aggregate`: our own chunk output, a (nested) list
    # thereof (`concatenate=False`), or a plain array while it’s computing `meta`
    type MomentsIn = _Moments | CpuArray | GpuArray | Sequence[Any]


@no_type_check  # mypy is extremely confused
def mean_var_(
    x: CpuArray | GpuArray | types.DaskArray | types.HasArrayNamespace,
    /,
    *,
    axis: Literal[0, 1] | None = None,
    correction: int = 0,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[types.CupyArray, types.CupyArray]
    | tuple[np.float64, np.float64]
    | tuple[types.DaskArray, types.DaskArray]
):
    if isinstance(x, types.DaskArray):
        return _dask_mean_var(x, axis=axis, correction=correction)

    from . import mean

    if isinstance(x, np.ndarray | types.CSBase) or not isinstance(x, types.HasArrayNamespace):
        xp = np
    else:
        import array_api_compat

        xp = array_api_compat.array_namespace(x)

    if axis is not None and isinstance(x, types.CSBase):
        mean_, var = _sparse_mean_var(x, axis=axis)
    else:
        mean_ = mean(x, axis=axis, dtype=xp.float64)
        mean_sq = mean(power(x, 2, dtype=xp.float64), axis=axis, dtype=xp.float64)
        var = mean_sq - mean_**2
    if correction:  # R convention == 1 (unbiased estimator)
        n = np.prod(x.shape) if axis is None else x.shape[axis]
        if n != 1:
            var *= n / (n - correction)
    return mean_, var


def _dask_mean_var(x: types.DaskArray, /, *, axis: Literal[0, 1] | None, correction: int) -> tuple[types.DaskArray, types.DaskArray]:
    """Mean and variance of a dask array.

    ``mean`` is a normal (associative) sum-based dask reduction.
    ``var`` is instead derived from per-chunk ``(count, mean, M2)`` triples
    (``M2 = sum((x - mean)**2)``, computed per chunk by recursing into :func:`mean_var_`)
    that get merged pairwise across chunks using Chan's parallel-variance algorithm.
    Separately dask-reducing ``mean(x)``/``mean(x**2)`` and subtracting at the end
    (the naive two-pass formula used before) loses precision once many chunks are
    combined, especially for float32 data on GPUs.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    import dask.array as da

    from . import mean

    n = np.prod(x.shape) if axis is None else x.shape[axis]
    mean_ = mean(x, axis=axis, dtype=np.float64)
    # mypy can’t infer `reduction`’s type parameter from the callbacks, so pin it here
    chunk: _Chunk[_Moments] = _moments_chunk
    m2 = da.reduction(x, chunk, _moments_aggregate, axis=axis, combine=_moments_combine, concatenate=False, dtype=np.float64)
    if axis is None:  # match `mean`/`sum`’s convention of reducing to a true scalar
        m2 = m2.map_blocks(lambda a: a.reshape(())[()], meta=m2.dtype.type(0))
    denom = n - correction if correction and n != 1 else n
    return mean_, m2 / denom


def _moments_chunk(
    a: CpuArray | GpuArray,
    /,
    *,
    axis: ComplexAxis = None,
    keepdims: bool = False,
    computing_meta: bool = False,
    **kwargs: object,  # noqa: ARG001  # `_Chunk`/`_CB` let dask pass arbitrary keywords
) -> _Moments | NDArray[np.float64]:
    if computing_meta:  # pragma: no cover
        return np.ndarray((), dtype=np.float64)
    axis_ = _normalize_axis(axis, a.ndim)
    mean_, var_ = mean_var_(a, axis=axis_, correction=0)
    n = int(np.prod(a.shape)) if axis_ is None else a.shape[axis_]
    shape = _get_shape(mean_, axis=axis_, keepdims=keepdims)
    moments: _Moments = {"n": n, "mean": mean_.reshape(shape), "m2": (var_ * n).reshape(shape)}
    return moments


def _moments_combine(
    pairs: MomentsIn,
    /,
    *,
    axis: ComplexAxis = None,  # noqa: ARG001
    keepdims: bool = False,  # noqa: ARG001
    computing_meta: bool = False,
    **kwargs: object,  # noqa: ARG001  # `_Chunk`/`_CB` let dask pass arbitrary keywords
) -> _Moments | NDArray[np.float64]:
    if computing_meta:  # pragma: no cover
        return np.ndarray((), dtype=np.float64)
    return _combine_all(pairs)


def _moments_aggregate(
    pairs: MomentsIn,
    /,
    *,
    axis: ComplexAxis = None,
    keepdims: bool = False,
    computing_meta: bool = False,
    **kwargs: object,  # noqa: ARG001  # `_Chunk`/`_CB` let dask pass arbitrary keywords
) -> NDArray[np.float64]:
    if computing_meta:  # pragma: no cover
        return np.ndarray((), dtype=np.float64)
    m2 = _combine_all(pairs)["m2"]
    axis_ = _normalize_axis(axis, 2)
    return m2.reshape(_final_moments_shape(m2.size, axis=axis_, keepdims=keepdims))


def _combine_all(pairs: MomentsIn) -> _Moments:
    """Merge every moment triple in a (possibly nested) list of `_moments_chunk` outputs."""
    combined = None
    for pair in _flatten_moments(pairs):
        combined = pair if combined is None else _chan_combine(combined, pair)
    assert combined is not None
    return combined


def _flatten_moments(pairs: MomentsIn) -> Iterator[_Moments]:
    match pairs:
        case {"n": _, "mean": _, "m2": _}:
            yield cast("_Moments", pairs)
        case Sequence():  # `concatenate=False` means dask hands us nested lists
            for pair in pairs:
                yield from _flatten_moments(pair)
        case _:  # pragma: no cover
            msg = f"Unexpected moments input: {type(pairs)}"
            raise TypeError(msg)


def _chan_combine(a: _Moments, b: _Moments) -> _Moments:
    """Pairwise-merge two ``(count, mean, M2)`` moment triples."""
    n_a, mean_a, m2_a = a["n"], a["mean"], a["m2"]
    n_b, mean_b, m2_b = b["n"], b["mean"], b["m2"]
    n = n_a + n_b
    delta = mean_b - mean_a
    mean_ = mean_a + delta * (n_b / n)
    m2 = m2_a + m2_b + delta**2 * (n_a * n_b / n)
    return {"n": n, "mean": mean_, "m2": m2}


def _final_moments_shape(size: int, *, axis: Literal[0, 1] | None, keepdims: bool) -> tuple[int, ...]:
    """Shape for a fully-combined moment array, mirroring `_get_shape`'s convention."""
    if axis is None:
        return (1, 1) if keepdims else (1,)
    if not keepdims:
        return (size,)
    return (1, size) if axis == 0 else (size, 1)


def _sparse_mean_var(mtx: types.CSBase, /, *, axis: Literal[0, 1]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
        mtx,
        major_len=shape[0],
        minor_len=shape[1],
        n_threads=numba.get_num_threads(),
    )


@njit
def sparse_mean_var_minor_axis(
    x: types.CSBase,
    *,
    major_len: int,
    minor_len: int,
    n_threads: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute mean and variance along the minor axis of a compressed sparse matrix."""
    rows = len(x.indptr) - 1
    sums = np.zeros((n_threads, minor_len))
    squared_sums = np.zeros((n_threads, minor_len))
    means = np.zeros(minor_len)
    variances = np.zeros(minor_len)
    for i in numba.prange(n_threads):
        for r in range(i, rows, n_threads):
            for j in range(x.indptr[r], x.indptr[r + 1]):
                minor_index = x.indices[j]
                if minor_index >= minor_len:
                    continue
                value = x.data[j]
                sums[i, minor_index] += value
                squared_sums[i, minor_index] += value * value
    for c in numba.prange(minor_len):
        sum = sums[:, c].sum()
        means[c] = sum / major_len
        variances[c] = squared_sums[:, c].sum() / major_len - (sum / major_len) ** 2
    return means, variances


@njit
def sparse_mean_var_major_axis(
    x: types.CSBase,
    *,
    major_len: int,
    minor_len: int,
    n_threads: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute means and variances along the major axis of a compressed sparse matrix."""
    rows = len(x.indptr) - 1
    means = np.zeros(major_len)
    variances = np.zeros_like(means)

    for i in numba.prange(n_threads):
        for r in range(i, rows, n_threads):
            sum_major = np.float64(0.0)
            squared_sum_minor = np.float64(0.0)
            for j in range(x.indptr[r], x.indptr[r + 1]):
                value = np.float64(x.data[j])
                sum_major += value
                squared_sum_minor += value * value
            means[r] = sum_major
            variances[r] = squared_sum_minor
    for c in numba.prange(major_len):
        mean = means[c] / minor_len
        means[c] = mean
        variances[c] = variances[c] / minor_len - mean * mean
    return means, variances
