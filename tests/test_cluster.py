# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest


numba = pytest.importorskip("numba")  # fast_array_utils.cluster imports numba at module level

from fast_array_utils.cluster import _seed, _sync, kmeans, kmeans_plusplus  # noqa: E402


if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture


def _data(n: int, d: int, k: int, seed: int, *, spread: float = 8.0, offset: float = 0.0, dtype: type = np.float32) -> np.ndarray:
    """Gaussian blobs around k random centers (spread: their scale), shifted by offset."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=spread, size=(k, d))
    return (centers[rng.integers(k, size=n)] + rng.normal(size=(n, d)) + offset).astype(dtype)


def _sqdist(x64: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d2 = np.empty((len(x64), len(centers)))
    for c in range(len(centers)):
        d2[:, c] = ((x64 - centers[c]) ** 2).sum(-1)
    return d2


def _lloyd(x: np.ndarray, init: np.ndarray, tol: float = 1e-4, max_iter: int = 300) -> np.ndarray:
    """Plain float64 Lloyd with sklearn's stopping rule; one final relabel against the last centers."""
    x64 = x.astype(np.float64)
    centers = init.astype(np.float64).copy()
    tol_abs = tol * np.mean(np.var(x64, axis=0))
    for _ in range(max_iter):
        labels = _sqdist(x64, centers).argmin(1)
        new = centers.copy()
        for c in range(len(centers)):
            if (mask := labels == c).any():
                new[c] = x64[mask].mean(0)
        shift = ((new - centers) ** 2).sum()
        centers = new
        if shift <= tol_abs:
            break
    return _sqdist(x64, centers).argmin(1)


@pytest.mark.parametrize(
    ("n", "d", "k", "kw", "dtype", "max_iter"),
    [
        pytest.param(3000, 10, 3, {}, np.float32, 300, id="f32-k3"),
        pytest.param(3000, 10, 20, {}, np.float32, 300, id="f32-k20"),
        pytest.param(3000, 10, 80, {}, np.float32, 300, id="f32-k80"),
        pytest.param(3000, 10, 20, {}, np.float64, 300, id="f64-k20"),
        pytest.param(3000, 10, 80, {}, np.float64, 300, id="f64-k80"),
        # approximate (dot-form) distances lose most precision far from the origin and in high dimensions
        pytest.param(4000, 200, 30, {"spread": 0.3, "offset": 1e3}, np.float32, 30, id="f32-d200-far-from-origin"),
        pytest.param(4000, 50, 40, {"spread": 0.2, "offset": 1e4}, np.float32, 30, id="f32-d50-very-far"),
        pytest.param(4000, 50, 60, {"spread": 0.05, "offset": 50.0}, np.float32, 30, id="f32-near-duplicate-centers"),
        pytest.param(2000, 400, 70, {"spread": 0.2, "offset": 100.0}, np.float32, 30, id="f32-d400-k70"),
        pytest.param(2000, 400, 70, {"spread": 0.2, "offset": 100.0}, np.float64, 30, id="f64-d400-k70"),
        # many centers on unclustered data: most rows fail the bounds, so the per-label (ball) scans can run
        pytest.param(4096, 32, 256, {"spread": 0.0}, np.float32, 10, id="f32-k256-ball-scans"),
    ],
)
def test_labels_equal_plain_lloyd(n: int, d: int, k: int, kw: dict[str, float], dtype: type, max_iter: int) -> None:  # noqa: PLR0917
    x = _data(n, d, k, seed=d + k, dtype=dtype, **kw)
    init = kmeans_plusplus(x, k, rng=np.random.default_rng(0))
    res = kmeans(x, k, rng=np.random.default_rng(0), init=init, max_iter=max_iter)
    np.testing.assert_array_equal(res.labels, _lloyd(x, init, max_iter=max_iter))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ties_go_to_the_lowest_center(dtype: type) -> None:
    """Integer data on a small grid: many rows are exactly as far from two centers, and the lowest index wins, as in the reference."""
    x = np.random.default_rng(0).integers(0, 3, size=(3000, 4)).astype(dtype)
    init = kmeans_plusplus(x, 12, rng=np.random.default_rng(0))
    res = kmeans(x, 12, rng=np.random.default_rng(0), init=init)
    np.testing.assert_array_equal(res.labels, _lloyd(x, init))


@pytest.mark.parametrize("copy", [True, False], ids=["transposed-copy", "no-copy"])
def test_same_result_for_any_thread_count(monkeypatch: pytest.MonkeyPatch, copy: bool) -> None:  # noqa: FBT001
    if not copy:  # the seeding path for x too large for a transposed copy
        monkeypatch.setattr(_seed, "SEED_COPY_BYTES", 0)
    x = _data(20_000, 16, 30, seed=1)
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


@pytest.mark.parametrize("copy", [True, False], ids=["transposed-copy", "no-copy"])
def test_seeds_are_rows(monkeypatch: pytest.MonkeyPatch, copy: bool) -> None:  # noqa: FBT001
    if not copy:
        monkeypatch.setattr(_seed, "SEED_COPY_BYTES", 0)
    x = _data(5000, 6, 12, seed=5, dtype=np.float64)
    centers = kmeans_plusplus(x, 12, rng=np.random.default_rng(0))
    assert all((x == c).all(1).any() for c in centers)
    assert len(np.unique(centers, axis=0)) == 12


def test_max_moved_stops_early() -> None:
    x = _data(10_000, 8, 50, seed=2, dtype=np.float64)
    full = kmeans(x, 50, rng=np.random.default_rng(0), tol=0.0, max_iter=50)
    early = kmeans(x, 50, rng=np.random.default_rng(0), tol=0.0, max_iter=50, max_moved=len(x) // 100)
    assert early.n_iter <= full.n_iter
    assert early.labels.shape == (len(x),)


def test_seed_sample_and_max_iter() -> None:
    x = _data(10_000, 12, 40, seed=4, dtype=np.float64)
    res = kmeans(x, 40, rng=np.random.default_rng(0), seed_sample=1200, max_iter=20)
    assert res.n_iter <= 20
    assert res.centers.shape == (40, 12)
    assert res.labels.min() >= 0
    assert res.labels.max() < 40


def test_integer_input_becomes_float64() -> None:
    x = (_data(2000, 5, 4, seed=9) * 10).astype(np.int64)
    a = kmeans(x, 4, rng=np.random.default_rng(0))
    b = kmeans(x.astype(np.float64), 4, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.centers, b.centers)


@pytest.mark.parametrize(
    ("layer", "step_terms", "expected"),
    [
        pytest.param(ValueError, 1, 0, id="no-region-yet"),
        pytest.param("workqueue", 1, -1, id="workqueue-small-step"),
        pytest.param("workqueue", 10**12, 0, id="workqueue-big-step"),
        pytest.param("tbb", 1, 0, id="tbb"),
    ],
)
def test_workers(monkeypatch: pytest.MonkeyPatch, layer: str | type[Exception], step_terms: int, expected: int) -> None:
    """Persistent regions use one worker fewer only for small steps on workqueue (speed only, results don't change)."""

    def threading_layer() -> str:
        if isinstance(layer, type):
            raise layer
        return layer

    monkeypatch.setattr(numba, "get_num_threads", lambda: 4)
    monkeypatch.setattr(numba, "threading_layer", threading_layer)
    assert _sync._workers(step_terms) == 4 + expected  # noqa: SLF001


@pytest.mark.parametrize("k", [0, 101])
def test_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k="):
        kmeans(_data(100, 3, 2, seed=0), k, rng=np.random.default_rng(0))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("func", [kmeans, kmeans_plusplus])
def test_non_finite(func: object, value: float) -> None:
    x = _data(100, 3, 2, seed=0)
    x[5, 1] = value
    with pytest.raises(ValueError, match="NaN"):
        func(x, 2, rng=np.random.default_rng(0))  # type: ignore[operator]


# Benchmarks stay below BALL_MIN_K: ball scans are chosen per pass by measured time, which would make instruction counts vary.
# n × d × k sized so each call takes about 5 to 60 ms natively on one thread;
# overlapping clusters (spread 1), so every one of Lloyd’s passes moves rows.
BENCH_N, BENCH_D, BENCH_SPREAD = 20_000, 32, 1.0


@pytest.mark.benchmark
@pytest.mark.parametrize("k", [8, 128])
def test_kmeans_plusplus_benchmark(benchmark: BenchmarkFixture, k: int) -> None:
    x = _data(BENCH_N, BENCH_D, k, seed=0, spread=BENCH_SPREAD)
    kmeans_plusplus(x[:2000], 5, rng=np.random.default_rng(0))  # warmup: numba compile

    @benchmark
    def call() -> None:
        kmeans_plusplus(x, k, rng=np.random.default_rng(0))


@pytest.mark.benchmark
@pytest.mark.parametrize("k", [8, 128])
def test_kmeans_benchmark(benchmark: BenchmarkFixture, k: int) -> None:
    """Lloyd from fixed seeds for a fixed number of passes (tol=0), so the work doesn’t depend on convergence."""
    x = _data(BENCH_N, BENCH_D, k, seed=0, spread=BENCH_SPREAD)
    init = kmeans_plusplus(x, k, rng=np.random.default_rng(0))
    kmeans(x[:2000], 5, rng=np.random.default_rng(0))  # warmup: numba compile

    @benchmark
    def call() -> None:
        kmeans(x, k, rng=np.random.default_rng(0), init=init, max_iter=5, tol=0.0)
