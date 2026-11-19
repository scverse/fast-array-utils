# SPDX-License-Identifier: MPL-2.0
"""Entry points of the exact, deterministic k-means.

Guarantees:

- Exact: labels are those of plain float64 Lloyd from the same seeds.
  Approximate float32 distances only rank centers;
  a rigorous per-center rounding allowance decides when a label is certain,
  and every center that could still be nearest is rechecked in float64 (`_tiles._dot_allowance`, `_tiles._scan_queue_dot`).
- Deterministic: results depend only on the data, `k` and the seed, not on the thread count or threading layer.
  All reductions run over fixed row chunks combined in chunk order;
  work is claimed dynamically, but which thread computes a chunk never changes its result.

Modules: `_seed` (greedy k-means++), `_lloyd` (Lloyd with Hamerly bounds), `_ball` (ball scans for large k),
`_tiles` (FMA register tiles and their rounding bound), `_sync` (atomics and work claiming of the persistent parallel regions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from fast_array_utils.numba import njit

from ._ball import BALL_MAX_K, BALL_MIN_K, BALL_MIN_ROWS
from ._lloyd import _hamerly, _hamerly_bounds, _inertia
from ._seed import _plusplus
from ._sync import _prefetch_distance, _workers


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


NORM_CHUNK = 2048  # rows per chunk of the norms/variance pass: small enough to stay in L2 for the chunk’s second pass


def _validate(x: NDArray[np.number[Any]], k: int) -> NDArray[np.float32] | NDArray[np.float64]:
    """C-contiguous float32 or float64 (other dtypes become float64); 1 ≤ k ≤ n."""
    x = np.ascontiguousarray(x)
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float64)
    if not 1 <= k <= x.shape[0]:
        msg = f"k={k} must be between 1 and the number of rows ({x.shape[0]})"
        raise ValueError(msg)
    return x


def _check_finite(xnorm: NDArray[np.float64]) -> None:
    if not np.isfinite(xnorm.max()):  # xnorm ≥ 0: NaN and inf propagate through max; also trips if squares overflow
        msg = "x contains NaN or infinity"
        raise ValueError(msg)


def kmeans_plusplus_(x: NDArray[np.number[Any]], k: int, *, rng: np.random.Generator, n_local_trials: int | None) -> NDArray[np.float64]:
    x = _validate(x, k)
    _check_finite(_norms_var(x)[0])
    return _plusplus(x, k, rng, n_local_trials)


def kmeans_(
    x: NDArray[np.number[Any]],
    k: int,
    *,
    rng: np.random.Generator,
    init: NDArray[np.floating[Any]] | None,
    max_iter: int,
    tol: float,
    n_local_trials: int | None,
    seed_sample: int | None,
    max_moved: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], float, int]:
    x = _validate(x, k)
    n, d = x.shape
    xnorm, mean_var = _norms_var(x)
    _check_finite(xnorm)
    if init is not None:
        centers = np.array(init, dtype=np.float64)
    elif seed_sample is not None and seed_sample < n:
        rows = np.sort(rng.choice(n, seed_sample, replace=False))
        centers = _plusplus(x[rows], k, rng, n_local_trials)
    else:
        centers = _plusplus(x, k, rng, n_local_trials)
    labels = np.full(n, -1, dtype=np.int64)
    use_ball = BALL_MIN_K <= k <= BALL_MAX_K and n >= BALL_MIN_ROWS * k
    finfo = np.finfo(x.dtype)
    bounds = _hamerly_bounds(n, k, d)
    it = _hamerly(
        x,
        xnorm,
        centers,
        bounds,
        max_iter,
        tol * mean_var,
        max_moved,
        labels,
        _workers(n * d),
        float(finfo.eps) / 2,
        float(finfo.smallest_subnormal),
        _prefetch_distance(x),
        np.zeros((k, k) if use_ball else (1, 1)),
    )
    return labels, centers, _inertia(x, centers, labels, bounds), it


@njit
def _norms_var(x: NDArray[np.float32] | NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Row norms ‖x_i‖² and the mean column variance in one pass over x.

    Each fixed chunk takes its column means, then its squared deviations while it is still in cache;
    chunks are combined in chunk order with Chan’s parallel formula, so the result doesn’t depend on threads.
    """
    n, d = x.shape
    nc = (n + NORM_CHUNK - 1) // NORM_CHUNK
    xnorm = np.empty(n)
    csum, m2 = np.zeros((nc, d)), np.zeros((nc, d))
    for b in numba.prange(nc):
        lo, hi = b * NORM_CHUNK, min(n, (b + 1) * NORM_CHUNK)
        for i in range(lo, hi):
            s = 0.0
            for j in range(d):
                v = np.float64(x[i, j])
                s += v * v
                csum[b, j] += v
            xnorm[i] = s
        for j in range(d):
            csum[b, j] /= hi - lo  # now the chunk’s column means
        for i in range(lo, hi):
            for j in range(d):
                t = np.float64(x[i, j]) - csum[b, j]
                m2[b, j] += t * t
    var = 0.0
    for j in range(d):
        mean = 0.0
        for b in range(nc):
            mean += csum[b, j] * (min(n, (b + 1) * NORM_CHUNK) - b * NORM_CHUNK)
        mean /= n
        s = 0.0
        for b in range(nc):
            t = csum[b, j] - mean
            s += m2[b, j] + (min(n, (b + 1) * NORM_CHUNK) - b * NORM_CHUNK) * t * t
        var += s / n
    return xnorm, var / d
