# SPDX-License-Identifier: MPL-2.0
"""Greedy k-means++ seeding in one persistent parallel region (`_seed`), deterministic for any thread count."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numba
import numpy as np

from fast_array_utils.numba import njit

from ._sync import SPIN_BEFORE_YIELD, _claim_range, _fetch_add, _load_acquire, _prefetch_distance, _prefetch_row, _store_release, _workers, _yield_cpu
from ._tiles import SEED_ROWS, _score


if TYPE_CHECKING:
    from ._typing import F64, I64, FArr


# Kernels take their arrays as positional arguments (one PLR0917 suppression per kernel, FBT001 for flags):
# numba can’t inline calls with keyword-only arguments.

CHUNK = 1 << 13  # rows per chunk of the grid-order counting sort; fixed, so the order doesn’t depend on threads
SEED_CHUNK = 1 << 10  # max rows per chunk of the seeding sums; chunks depend only on n, so seeds don’t depend on threads
SEED_CHUNKS_PER_N = 64  # aim for about n / this rows per chunk, so small n still gets many chunks
SEED_COPY_BYTES = 128 << 20  # x up to this size gets a block-transposed copy, so seeding steps don’t transpose
GRID_DIMS, GRID_BITS = 4, 3  # rows ordered by a grid key on the 4 highest-variance columns (8 levels each): nearby rows share blocks
GRID_SAMPLE = 8192  # the grid’s column statistics come from about this many rows
FIRST_STAGE = 4  # columns in the first stage of the candidate pass; later stages double

# `_seed`’s state slots, on separate cache lines
_DONE, _PUBLISHED, _WINNER = 8, 16, 24  # chunks done over all steps, last published step, last step’s winning candidate
STATE_SLOTS = 32


def _plusplus(x: np.ndarray, k: int, rng: np.random.Generator, n_local_trials: int | None) -> np.ndarray:
    trials = 2 + int(math.log(k)) if n_local_trials is None else n_local_trials
    first = int(rng.integers(x.shape[0]))
    u = rng.random((k - 1, trials))  # the same draws, in the same order, as one call per step
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    if x.nbytes <= SEED_COPY_BYTES:
        perm, cols = _grid_order(x)
        xt = np.empty((-(-x.shape[0] // SEED_ROWS), x.shape[1], SEED_ROWS), dtype=x.dtype)
        xt[-1] = 0.0  # rows past the end of the last block stay zero; the transpose writes the rest
        _block_transpose(x, perm, cols, xt, _prefetch_distance(x))
    else:  # large x transposes per step (near memory-bound there anyway), to keep peak memory low
        perm, cols = np.arange(x.shape[0]), np.arange(x.shape[1])
        xt = np.zeros((0, x.shape[1], SEED_ROWS), dtype=x.dtype)
    nthreads = _workers(x.shape[0] * trials * x.shape[1])  # passed in: get_num_threads inside a kernel makes it uncacheable
    _seed(x, xt, perm, cols, first, u, centers, nthreads, _prefetch_distance(x))
    return centers


@njit
def _seed(x: FArr, xt: FArr, perm: I64, cols: I64, first: int, u: F64, centers: F64, nthreads: int, pfd: int) -> None:  # noqa: PLR0917
    """Greedy k-means++ in one compiled loop: per step, draw the candidates and score all of them in one parallel pass.

    Row positions follow `perm` (rows of x in grid order, so nearby rows share a SEED_ROWS block).
    With the transposed copy, columns follow `cols` and a candidate’s block stops early once no row in it can improve
    (partial sums of squares only grow, so min(D², full) stays D² exactly).
    A candidate’s D² is written only for blocks it completed (`live`); `cur` holds everyone else’s.
    Block sums use a fixed order and chunk sums are added in chunk order, so the result doesn’t depend on the thread count.
    """
    # prepare memory
    n, d = x.shape
    k, trials = centers.shape[0], u.shape[1]
    ch = min(SEED_CHUNK, max(SEED_ROWS, (n // SEED_CHUNKS_PER_N + SEED_ROWS - 1) // SEED_ROWS * SEED_ROWS))
    nb = (n + ch - 1) // ch
    nblk = (n + SEED_ROWS - 1) // SEED_ROWS
    tot, cur = np.empty(nb), np.empty(n)
    new_d2, pots = np.empty((trials, n)), np.empty((trials, nb))
    live = np.zeros((nblk, trials), dtype=np.int64)  # per block, contiguous: no false sharing between threads’ chunks
    cand, idx = np.empty((trials, d), dtype=x.dtype), np.empty(trials, dtype=np.int64)  # in x’s dtype: float32 data, float32 lanes
    state = np.zeros(STATE_SLOTS, dtype=np.int64)
    rcnt = np.zeros(8 * nthreads, dtype=np.int64)  # claims so far from each worker’s chunk range, one cache line each
    # the first center and everyone’s D² to it
    for j in range(d):
        centers[0, j] = x[first, j]
    for b in numba.prange(nb):
        tot[b] = _first_d2(x, perm, centers, cur, b * ch, min(n, (b + 1) * ch), pfd)
    if k == 1:
        return
    _draw(x, perm, cols, new_d2, cur, live, 0, tot, u[0], ch, idx, cand)
    # the k - 1 steps in one region (per-region wake-ups cost more than small steps’ work);
    # nobody waits for a particular thread: any worker can claim any chunk, so it can’t deadlock
    for _w in numba.prange(nthreads):
        acc = np.empty(trials)
        xb = np.zeros((1, d if not xt.shape[0] else 0, SEED_ROWS), dtype=x.dtype)  # the no-copy branch’s transposed block
        dv = np.empty((trials, SEED_ROWS), dtype=x.dtype)
        cb, w, lq = np.empty(SEED_ROWS), np.empty(SEED_ROWS), np.empty(trials, dtype=np.int64)
        own, st, nv = _w % nthreads, 0, 0  # chunks of the own range first (they stay in this core’s caches), then steal
        while True:
            b, st, nv = _claim_step_chunk(rcnt, own, st, nv, k, nb, nthreads)
            if st >= k - 1:
                break
            _wait_published(state, st)
            prev = state[_WINNER]
            acc[:] = 0.0
            hi = min(n, (b + 1) * ch)
            for r0 in range(b * ch, hi, SEED_ROWS):
                _score_block(x, xt, xb, cand, new_d2, cur, live, prev, r0, min(SEED_ROWS, hi - r0), cb, w, dv, lq, acc)
            for q in range(trials):
                pots[q, b] = acc[q]
            if _fetch_add(state, _DONE) == (st + 1) * nb - 1:  # the last chunk of step st: pick its winner, draw the next candidates
                _finish_step(x, perm, cols, u, centers, new_d2, cur, live, tot, pots, cand, idx, ch, st, state)
                _store_release(state, _PUBLISHED, st + 1)


@numba.njit(inline="always", nogil=True, cache=True)
def _first_d2(x: FArr, perm: I64, centers: F64, cur: F64, lo: int, hi: int, pfd: int) -> float:  # noqa: PLR0917
    """cur[lo:hi] = the squared distances of row positions lo..hi to the first center; returns their sum."""
    n, d = x.shape
    s = 0.0
    for i in range(lo, hi):
        if pfd and i + pfd < n:  # rows in grid order are scattered in x
            _prefetch_row(x, perm[i + pfd])
        v = 0.0
        for j in range(d):
            t = np.float64(x[perm[i], j]) - centers[0, j]
            v += t * t
        cur[i] = v
        s += v
    return s


@numba.njit(inline="always", nogil=True, cache=True)
def _claim_step_chunk(rcnt: I64, own: int, st: int, nv: int, k: int, nb: int, nthreads: int) -> tuple[int, int, int]:  # noqa: PLR0917
    """Claim a chunk of the earliest step st with unclaimed chunks, from range own + nv; returns (chunk, st, nv)."""
    b = -1
    while st < k - 1:
        b = _claim_range(rcnt, (own + nv) % nthreads, st, nb, nthreads)
        if b >= 0:
            break
        nv += 1
        if nv == nthreads:  # every range of step st is claimed: on to the next step
            st, nv = st + 1, 0
    return b, st, nv


@numba.njit(inline="always", nogil=True, cache=True)
def _wait_published(state: I64, st: int) -> None:
    """Wait until step st’s candidates are drawn (by whoever finished step st - 1)."""
    spins = 0
    while _load_acquire(state, _PUBLISHED) < st:
        spins += 1
        if spins > SPIN_BEFORE_YIELD:
            _yield_cpu()


@numba.njit(inline="always", nogil=True, cache=True)
def _score_block(  # noqa: PLR0917
    x: FArr, xt: FArr, xb: FArr, cand: FArr, new_d2: F64, cur: F64, live: I64, prev: int, r0: int, m: int, cb: F64, w: F64, dv: FArr, lq: I64, acc: F64
) -> None:
    """Score every candidate on the m rows of the SEED_ROWS block at row position r0, adding its potential to acc."""
    d = x.shape[1]
    blk = r0 // SEED_ROWS
    fresh = live[blk, prev]  # the last winner changed this block: its values are current
    for r in range(SEED_ROWS):  # rows past m can’t improve and add nothing
        cb[r] = (new_d2[prev, r0 + r] if fresh else cur[r0 + r]) if r < m else 0.0
    sb = _bsum(cb)
    if not xt.shape[0]:
        for r in range(m):  # rows to columns, so the inner loop runs over rows: no horizontal sums
            for j in range(d):
                xb[0, j, r] = x[r0 + r, j]
    src, sblk = (xt, blk) if xt.shape[0] else (xb, 0)
    nl = _staged_scores(src, sblk, cand, lq, dv, cb, FIRST_STAGE if xt.shape[0] else d)
    _take_block_scores(new_d2, cur, live, blk, r0, m, fresh, nl, lq, cb, sb, w, dv, acc)


@numba.njit(inline="always", nogil=True, cache=True)
def _take_block_scores(  # noqa: PLR0917
    new_d2: F64, cur: F64, live: I64, blk: int, r0: int, m: int, fresh: int, nl: int, lq: I64, cb: F64, sb: float, w: F64, dv: FArr, acc: F64
) -> None:
    """Record which candidates completed the block (lq[:nl]) and their new D² there; add every candidate’s block potential."""
    trials = live.shape[1]
    for q in range(trials):
        live[blk, q] = 0
    for t in range(nl):
        live[blk, lq[t]] = 1
    if nl < trials and fresh:  # some candidate keeps this block’s D²: make it current
        for r in range(m):
            cur[r0 + r] = cb[r]
    for q in range(trials):
        if live[blk, q]:
            for r in range(SEED_ROWS):
                w[r] = min(cb[r], np.float64(dv[q, r]))
            for r in range(m):
                new_d2[q, r0 + r] = w[r]
            acc[q] += _bsum(w)
        else:
            acc[q] += sb


@numba.njit(inline="always", nogil=True, cache=True)
def _staged_scores(src: FArr, sblk: int, cand: FArr, lq: I64, dv: FArr, cb: F64, first_stage: int) -> int:  # noqa: PLR0917
    """Candidates’ partial D² over columns in stages of first_stage, 2·first_stage, …; returns how many stay live (lq[:nl]).

    A candidate leaves once no row’s partial sum is below its current D² (cb): the rest only adds,
    so min(D², full) stays D², exactly.
    """
    d = src.shape[1]
    trials = cand.shape[0]
    dv[:, :] = 0.0
    nl: int = trials  # candidates still live in this block, in order
    for q in range(trials):
        lq[q] = q
    j, e = 0, min(first_stage, d)
    while True:
        _score(src, sblk, cand, lq, nl, dv, j, e)
        j = e
        if e == d:
            return nl
        nn = 0
        for t in range(nl):
            q = lq[t]
            cnt = 0
            for r in range(SEED_ROWS):
                cnt += np.float64(dv[q, r]) < cb[r]
            if cnt:
                lq[nn] = q
                nn += 1
        nl = nn
        if nl == 0:
            return nl
        e = min(2 * e, d)


@numba.njit(inline="always", nogil=True, cache=True)
def _finish_step(  # noqa: PLR0917
    x: FArr,
    perm: I64,
    cols: I64,
    u: F64,
    centers: F64,
    new_d2: F64,
    cur: F64,
    live: I64,
    tot: F64,
    pots: F64,
    cand: FArr,
    idx: I64,
    ch: int,
    st: int,
    state: I64,
) -> None:
    """End of step st: the candidate with the lowest potential becomes center st + 1 (ties: lowest candidate); draw the next."""
    k, d = centers.shape
    trials, nb = pots.shape
    c = st + 1
    best, best_pot = 0, np.inf
    for q in range(trials):
        sq = 0.0
        for bb in range(nb):
            sq += pots[q, bb]
        if sq < best_pot:
            best, best_pot = q, sq
    for j in range(d):
        centers[c, cols[j]] = cand[best, j]
    for bb in range(nb):
        tot[bb] = pots[best, bb]
    state[_WINNER] = best
    if c + 1 < k:
        _draw(x, perm, cols, new_d2, cur, live, best, tot, u[c], ch, idx, cand)


@numba.njit(inline="always", nogil=True, cache=True)
def _draw(x: FArr, perm: I64, cols: I64, new_d2: F64, cur: F64, live: I64, prev: int, tot: F64, uc: F64, ch: int, idx: I64, cand: FArr) -> None:  # noqa: PLR0917
    """Draw the step’s candidates (D²-weighted, from the chunk totals and the last winner’s D²) and gather them in column order."""
    total = 0.0
    for b in range(tot.shape[0]):
        total += tot[b]
    for q in range(cand.shape[0]):
        idx[q] = perm[_pick(new_d2[prev], cur, live, prev, tot, uc[q] * total, ch)]
        for j in range(cand.shape[1]):
            cand[q, j] = x[idx[q], cols[j]]


@numba.njit(inline="always", nogil=True, cache=True)
def _pick(row: F64, cur: F64, live: I64, q: int, tot: F64, target: float, chunk: int) -> int:  # noqa: PLR0917
    """Row whose cumulative D² first exceeds `target`: the chunk from the chunk totals, then the row inside it."""
    nb = tot.shape[0]
    b, cum = 0, 0.0
    while b < nb - 1 and cum + tot[b] <= target:
        cum += tot[b]
        b += 1
    lo, hi = b * chunk, min(cur.shape[0], (b + 1) * chunk)
    last = lo
    for i in range(lo, hi):
        v = _d2(row, cur, live, q, i)
        if v > 0.0:
            last = i
        cum += v
        if cum > target:
            return i
    return last  # rounding between chunk totals and row sums: the last row that can be drawn


@numba.njit(inline="always", nogil=True, cache=True)
def _d2(row: F64, cur: F64, live: I64, q: int, i: int) -> float:
    """Return the current D² of row position i: the last winner q’s value where it was computed for i’s block, else `cur`."""
    v: float = row[i] if live[i // SEED_ROWS, q] else cur[i]
    return v


@numba.njit(inline="always", nogil=True, cache=True)
def _bsum(w: F64) -> float:
    """Sum of a SEED_ROWS block in a fixed order (8 interleaved partial sums, then a fixed tree)."""
    s0 = s1 = s2 = s3 = s4 = s5 = s6 = s7 = 0.0
    for g in range(0, SEED_ROWS, 8):
        s0 += w[g]
        s1 += w[g + 1]
        s2 += w[g + 2]
        s3 += w[g + 3]
        s4 += w[g + 4]
        s5 += w[g + 5]
        s6 += w[g + 6]
        s7 += w[g + 7]
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))


# ----------------------------------------------------------------------------- row and column order
@njit
def _grid_order(x: FArr) -> tuple[I64, I64]:
    """Row order (stable counting sort on the grid key) and column order (variance, descending), from the data alone.

    Variance and range come from a fixed subsample (every n // GRID_SAMPLE-th row), so the order doesn’t depend on threads.
    """
    n = x.shape[0]
    lo, hi, var = _sample_stats(x, max(1, n // GRID_SAMPLE))
    cols = np.argsort(-var, kind="mergesort")
    g = min(GRID_DIMS, x.shape[1])
    levels = 1 << GRID_BITS
    scale = np.zeros(g)
    for t in range(g):
        c = cols[t]
        if hi[c] > lo[c]:
            scale[t] = levels / (hi[c] - lo[c])
    nk = levels**g
    nc = (n + CHUNK - 1) // CHUNK
    key = np.empty(n, dtype=np.int64)
    hist = np.zeros((nc, nk), dtype=np.int64)
    for b in numba.prange(nc):
        for i in range(b * CHUNK, min(n, (b + 1) * CHUNK)):
            kk = _grid_key(x, i, cols, lo, scale, levels)
            key[i] = kk
            hist[b, kk] += 1
    pos = _bucket_starts(hist)
    perm = np.empty(n, dtype=np.int64)
    for b in numba.prange(nc):
        for i in range(b * CHUNK, min(n, (b + 1) * CHUNK)):
            kk = key[i]
            perm[pos[b, kk]] = i
            pos[b, kk] += 1
    return perm, cols


@numba.njit(inline="always", nogil=True, cache=True)
def _sample_stats(x: FArr, step: int) -> tuple[F64, F64, F64]:
    """Per-column min, max and variance over every step-th row."""
    n, d = x.shape
    ns = (n + step - 1) // step
    mean, var = np.zeros(d), np.zeros(d)
    lo, hi = np.full(d, np.inf), np.full(d, -np.inf)
    for i in range(0, n, step):
        for j in range(d):
            v = np.float64(x[i, j])
            mean[j] += v
            lo[j] = min(lo[j], v)
            hi[j] = max(hi[j], v)
    for j in range(d):
        mean[j] /= ns
    for i in range(0, n, step):
        for j in range(d):
            t = np.float64(x[i, j]) - mean[j]
            var[j] += t * t
    return lo, hi, var


@numba.njit(inline="always", nogil=True, cache=True)
def _grid_key(x: FArr, i: int, cols: I64, lo: F64, scale: F64, levels: int) -> int:  # noqa: PLR0917
    """Row i’s grid cell: levels per column on the first scale.shape[0] columns of `cols`."""
    kk = 0
    for t in range(scale.shape[0]):
        c = cols[t]
        q = int((np.float64(x[i, c]) - lo[c]) * scale[t])
        kk = kk * levels + min(max(q, 0), levels - 1)
    return kk


@numba.njit(inline="always", nogil=True, cache=True)
def _bucket_starts(hist: I64) -> I64:
    """First position of each (chunk, key) bucket in key order, chunks in order within a key: a stable counting sort."""
    nc, nk = hist.shape
    pos = np.empty((nc, nk), dtype=np.int64)
    s = 0
    for kk in range(nk):
        for b in range(nc):
            pos[b, kk] = s
            s += hist[b, kk]
    return pos


@njit
def _block_transpose(x: FArr, perm: I64, cols: I64, xt: FArr, pfd: int) -> None:
    """xt[b, j, r] = x[perm[b·SEED_ROWS + r], cols[j]]; rows past the end stay zero.

    Rows come in grid order, i.e. scattered in x: prefetching pfd rows ahead (0: off) turns the latency-bound reads into a stream.
    """
    n, d = x.shape
    for b in numba.prange(xt.shape[0]):
        for r in range(min(SEED_ROWS, n - b * SEED_ROWS)):
            p = b * SEED_ROWS + r
            if pfd and p + pfd < n:
                _prefetch_row(x, perm[p + pfd])
            i = perm[p]
            for j in range(d):
                xt[b, j, r] = x[i, cols[j]]
