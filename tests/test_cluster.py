# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import numpy as np
import pytest


numba = pytest.importorskip("numba")

from fast_array_utils.cluster import kmeans, kmeans_plusplus  # noqa: E402


def _blobs(n: int, d: int, k: int, seed: int, dtype: type = np.float32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=8.0, size=(k, d))
    return (centers[rng.integers(k, size=n)] + rng.normal(size=(n, d))).astype(dtype)


def _lloyd(x: np.ndarray, init: np.ndarray, tol: float = 1e-4, max_iter: int = 300) -> np.ndarray:
    """Plain float64 Lloyd with sklearn's stopping rule; one final relabel against the last centers."""
    x64 = x.astype(np.float64)
    centers = init.astype(np.float64).copy()
    tol_abs = tol * np.mean(np.var(x64, axis=0))
    for _ in range(max_iter):
        labels = ((x64[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x64[labels == c].mean(0) if (labels == c).any() else centers[c] for c in range(len(centers))])
        shift = ((new - centers) ** 2).sum()
        centers = new
        if shift <= tol_abs:
            break
    return ((x64[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)


@pytest.mark.parametrize(
    ("n", "d", "k", "spread", "offset", "dtype"),
    [
        pytest.param(4000, 200, 30, 0.3, 1000.0, np.float32, id="f32-d200-far-from-origin"),
        pytest.param(4000, 50, 40, 0.2, 1e4, np.float32, id="f32-d50-very-far"),
        pytest.param(4000, 50, 60, 0.05, 50.0, np.float32, id="f32-near-duplicate-centers"),
        pytest.param(2000, 400, 70, 0.2, 100.0, np.float32, id="f32-d400-k70"),
    ],
)
def test_labels_exact_on_adversarial_data(n: int, d: int, k: int, spread: float, offset: float, dtype: type) -> None:
    """Approximate (dot-form) distances must never change a label: uncentered, high-d data (exp50b fix)."""
    g = np.random.default_rng(d + k)
    c = g.normal(size=(k, d)) * spread
    x = (c[g.integers(0, k, n)] + g.normal(size=(n, d)) + offset).astype(dtype)
    init = kmeans_plusplus(x, k, np.random.default_rng(0))
    res = kmeans(x, k, rng=np.random.default_rng(0), init=init, max_iter=30)
    np.testing.assert_array_equal(res.labels, _lloyd(x, init, max_iter=30))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [3, 20, 80])
def test_labels_equal_plain_lloyd(dtype: type, k: int) -> None:
    x = _blobs(3000, 10, k, seed=k, dtype=dtype)
    init = kmeans_plusplus(x, k, np.random.default_rng(0))
    res = kmeans(x, k, rng=np.random.default_rng(0), init=init)
    np.testing.assert_array_equal(res.labels, _lloyd(x, init))


def test_same_result_for_any_thread_count() -> None:
    x = _blobs(20_000, 16, 30, seed=1)
    n = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = kmeans(x, 30, rng=np.random.default_rng(3))
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        b = kmeans(x, 30, rng=np.random.default_rng(3))
    finally:
        numba.set_num_threads(n)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.centers, b.centers)
    assert a.inertia == b.inertia


def test_max_moved_stops_early() -> None:
    x = _blobs(20_000, 8, 50, seed=2, dtype=np.float64)
    full = kmeans(x, 50, rng=np.random.default_rng(0), tol=0.0)
    early = kmeans(x, 50, rng=np.random.default_rng(0), tol=0.0, max_moved=len(x) // 100)
    assert early.n_iter <= full.n_iter
    assert early.labels.shape == (len(x),)


def test_seed_sample_and_max_iter() -> None:
    x = _blobs(10_000, 12, 40, seed=4, dtype=np.float64)
    res = kmeans(x, 40, rng=np.random.default_rng(0), seed_sample=1200, max_iter=20)
    assert res.n_iter <= 20
    assert res.centers.shape == (40, 12)
    assert res.labels.min() >= 0 and res.labels.max() < 40


@pytest.mark.parametrize("k", [0, 101])
def test_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k="):
        kmeans(_blobs(100, 3, 2, seed=0), k, rng=np.random.default_rng(0))


def test_non_finite() -> None:
    x = _blobs(100, 3, 2, seed=0)
    x[5, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        kmeans(x, 2, rng=np.random.default_rng(0))
