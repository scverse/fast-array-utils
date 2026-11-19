# SPDX-License-Identifier: MPL-2.0
"""Clustering: exact, deterministic k-means (requires :doc:`numba <numba:index>`).

Labels equal those of plain float64 Lloyd from the same seeds,
and results don’t depend on the thread count, threading layer or CPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple


if TYPE_CHECKING:
    from typing import Any

    import numpy as np
    from numpy.typing import NDArray


__all__ = ["KMeansResult", "kmeans", "kmeans_plusplus"]


class KMeansResult(NamedTuple):
    """Result of :func:`kmeans`."""

    labels: NDArray[np.int64]
    """Index of each row’s center, shape (n,)."""
    centers: NDArray[np.float64]
    """Final centers, shape (k, d)."""
    inertia: float
    """Sum of squared distances of the rows to their centers."""
    n_iter: int
    """Lloyd iterations run."""


def kmeans_plusplus(x: NDArray[np.number[Any]], /, k: int, *, rng: np.random.Generator, n_local_trials: int | None = None) -> NDArray[np.float64]:
    """Greedy k-means++ seeding, deterministic for any thread count.

    Parameters
    ----------
    x
        Dense data, shape (n, d).
        float32 and float64 are used as they are; other dtypes become float64.
    k
        Number of centers, 1 ≤ k ≤ n.
    rng
        Source of the random draws.
    n_local_trials
        Candidates per step; default ``2 + int(ln k)``, as scikit-learn.

    Returns
    -------
    The k initial centers, shape (k, d).

    Example
    -------
    >>> import numpy as np
    >>> from fast_array_utils.cluster import kmeans_plusplus
    >>> x = np.array([[0.0], [0.1], [10.0], [10.1]])
    >>> kmeans_plusplus(x, 2, rng=np.random.default_rng(0)).shape
    (2, 1)

    """
    from ._kmeans import kmeans_plusplus_

    return kmeans_plusplus_(x, k, rng=rng, n_local_trials=n_local_trials)


def kmeans(
    x: NDArray[np.number[Any]],
    /,
    k: int,
    *,
    rng: np.random.Generator,
    init: NDArray[np.floating[Any]] | None = None,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_local_trials: int | None = None,
    seed_sample: int | None = None,
    max_moved: int = 0,
) -> KMeansResult:
    """Exact Lloyd’s k-means from greedy k-means++ seeds.

    The labels are those of plain float64 Lloyd from the same seeds,
    and results don’t depend on the thread count or threading layer.
    Hamerly’s bounds and exact rechecks of approximate distances skip most distance computations.
    Ties go to the lowest center index, as in the reference.

    Parameters
    ----------
    x
        Dense data, shape (n, d).
        float32 and float64 are used as they are; other dtypes become float64.
    k
        Number of clusters, 1 ≤ k ≤ n.
    rng
        Source of the random draws for seeding.
    init
        Initial centers, shape (k, d); seeding is skipped.
    max_iter
        Maximum number of Lloyd iterations.
    tol
        Stop once the squared center shift is at most ``tol`` times the mean feature variance (scikit-learn’s rule).
    n_local_trials
        Seeding candidates per step; default ``2 + int(ln k)``, as scikit-learn.
    seed_sample
        Seed on this many random rows only.
    max_moved
        Also stop once a pass changes at most this many labels (0: only when nothing moves, or by ``tol``).

    Returns
    -------
    Labels, centers, inertia and the number of iterations.

    Example
    -------
    >>> import numpy as np
    >>> from fast_array_utils.cluster import kmeans
    >>> x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [10.0, 10.0], [10.0, 11.0], [11.0, 10.0]])
    >>> res = kmeans(x, 2, rng=np.random.default_rng(0))
    >>> sorted(res.centers.round(2).tolist())
    [[0.33, 0.33], [10.33, 10.33]]
    >>> res.labels[:3].tolist() == [res.labels[0]] * 3
    True

    """
    from ._kmeans import kmeans_

    labels, centers, inertia, n_iter = kmeans_(
        x, k, rng=rng, init=init, max_iter=max_iter, tol=tol, n_local_trials=n_local_trials, seed_sample=seed_sample, max_moved=max_moved
    )
    return KMeansResult(labels, centers, inertia, n_iter)
