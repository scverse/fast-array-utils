# SPDX-License-Identifier: MPL-2.0
"""Greedy k-means++ seeding in one persistent parallel region (`_seed`), deterministic for any thread count."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numba
import numpy as np

from fast_array_utils.numba import njit

from ._sync import LINE_SLOTS, _backoff, _claim_range, _fetch_add, _load_acquire, _prefetch_distance, _prefetch_row, _store_release, _workers
from ._tiles import BLOCK_ROWS, _score


if TYPE_CHECKING:
    from ._typing import F64, I64, FArr


CHUNK = 1 << 13  # rows per chunk of the grid-order counting sort; fixed, so the order doesn’t depend on threads
SEED_CHUNK = 1 << 10  # max rows per chunk of the seeding sums; chunks depend only on n, so seeds don’t depend on threads
SEED_CHUNKS_PER_N = 64  # aim for about n / this rows per chunk, so small n still gets many chunks
SEED_COPY_BYTES = 128 << 20  # x up to this size gets a block-transposed copy, so seeding steps don’t transpose
GRID_DIMS, GRID_BITS = 4, 3  # rows ordered by a grid key on the 4 highest-variance columns (8 levels each): nearby rows share blocks
GRID_LEVELS = 1 << GRID_BITS
GRID_SAMPLE = 8192  # the grid’s column statistics come from about this many rows
FIRST_STAGE = 4  # columns in the first stage of the candidate pass; later stages double
BSUM_WAYS = 8  # interleaved partial sums of `_bsum`
assert BLOCK_ROWS % BSUM_WAYS == 0

# state slots of `_seed`, on separate cache lines
STEP_DONE, PUBLISHED, WINNER = 8, 16, 24  # chunks done over all steps, last published step, last step’s winning candidate
STATE_SLOTS = 32


def _plusplus(x: FArr, k: int, rng: np.random.Generator, n_local_trials: int | None) -> F64:
    n, d = x.shape
    trials = 2 + int(math.log(k)) if n_local_trials is None else n_local_trials
    first = int(rng.integers(n))
    u = rng.random((k - 1, trials))  # the same draws, in the same order, as one call per step
    centers = np.empty((k, d), dtype=np.float64)
    if x.nbytes <= SEED_COPY_BYTES:
        perm, cols = _grid_order(x)
        xt = np.empty((-(-n // BLOCK_ROWS), d, BLOCK_ROWS), dtype=x.dtype)
        xt[-1] = 0.0  # rows past the end of the last block stay zero; the transpose writes the rest
        _block_transpose(x, perm, cols, xt, _prefetch_distance(x))
    else:  # large x transposes per step (near memory-bound there anyway), to keep peak memory low; xt stays empty
        perm, cols = np.arange(n), np.arange(d)
        xt = np.zeros((0, d, BLOCK_ROWS), dtype=x.dtype)
    nthreads = _workers(n * trials * d)  # passed in: get_num_threads inside a kernel makes it uncacheable
    _seed(x, xt, perm, cols, first, u, centers, nthreads, _prefetch_distance(x))
    return centers


@njit
def _seed(  # noqa: C901, PLR0912, PLR0915, PLR0917
    x: FArr, xt: FArr, perm: I64, cols: I64, first: int, u: F64, centers: F64, nthreads: int, pfd: int
) -> None:
    """Greedy k-means++ in one compiled loop: per step, draw the candidates and score all of them in one parallel pass.

    Row positions follow `perm` (rows of x in grid order, so nearby rows share a BLOCK_ROWS block).
    With the transposed copy (xt not empty), columns follow `cols` and a candidate’s block stops early once no row in it
    can improve (partial sums of squares only grow, so min(D², full) stays D² exactly).
    A candidate’s D² is written only for blocks it completed (`live`); `cur` holds everyone else’s.
    Block sums use a fixed order and chunk sums are added in chunk order, so the result doesn’t depend on the thread count.
    """
    # prepare memory
    n, d = x.shape
    k, trials = centers.shape[0], u.shape[1]
    ch = min(SEED_CHUNK, max(BLOCK_ROWS, (n // SEED_CHUNKS_PER_N + BLOCK_ROWS - 1) // BLOCK_ROWS * BLOCK_ROWS))
    nb = (n + ch - 1) // ch
    nblk = (n + BLOCK_ROWS - 1) // BLOCK_ROWS
    tot, cur = np.empty(nb), np.empty(n)
    new_d2, pots = np.empty((trials, n)), np.empty((trials, nb))
    live = np.zeros((nblk, trials), dtype=np.int64)  # per block, contiguous: no false sharing between threads’ chunks
    cand, idx = np.empty((trials, d), dtype=x.dtype), np.empty(trials, dtype=np.int64)  # in x’s dtype: float32 data, float32 lanes
    state = np.zeros(STATE_SLOTS, dtype=np.int64)
    rcnt = np.zeros(LINE_SLOTS * nthreads, dtype=np.int64)  # claims so far from each worker’s chunk range, one cache line each
    # the first center, and everyone’s D² to it
    for j in range(d):
        centers[0, j] = x[first, j]
    for b in numba.prange(nb):
        t = 0.0
        for i in range(b * ch, min(n, (b + 1) * ch)):
            if pfd and i + pfd < n:  # rows in grid order are scattered in x
                _prefetch_row(x, perm[i + pfd])
            v = 0.0
            for j in range(d):
                q = np.float64(x[perm[i], j]) - centers[0, j]
                v += q * q
            cur[i] = v
            t += v
        tot[b] = t
    if k == 1:
        return
    _draw(x, perm, cols, new_d2, cur, live, 0, tot, u[0], ch, idx, cand)
    # the k - 1 steps in one region: waking the workers for a region costs more than a small step’s work
    # (64 → 31 ms at k = 1000 on 5000 rows); any worker can claim any chunk, so it can’t deadlock
    for wi in numba.prange(nthreads):
        acc = np.empty(trials)
        xb = np.zeros((1, d if not xt.shape[0] else 0, BLOCK_ROWS), dtype=x.dtype)  # the no-copy branch’s transposed block
        dv = np.empty((trials, BLOCK_ROWS), dtype=x.dtype)
        cb, w, lq = np.empty(BLOCK_ROWS), np.empty(BLOCK_ROWS), np.empty(trials, dtype=np.int64)
        own, st, nv = wi % nthreads, 0, 0  # chunks of the own range first (they stay in this core’s caches), then steal
        while True:
            b = -1
            while st < k - 1:  # claim a chunk of the earliest step with unclaimed chunks
                b = _claim_range(rcnt, (own + nv) % nthreads, st, nb, nthreads)
                if b >= 0:
                    break
                nv += 1
                if nv == nthreads:  # every range of step st is claimed: on to the next step
                    st, nv = st + 1, 0
            if st >= k - 1:
                break
            spins = 0
            while _load_acquire(state, PUBLISHED) < st:  # step st’s candidates are drawn by whoever finished step st - 1
                spins = _backoff(spins)
            prev = state[WINNER]
            acc[:] = 0.0
            hi = min(n, (b + 1) * ch)
            for r0 in range(b * ch, hi, BLOCK_ROWS):
                _score_block(x, xt, xb, cand, new_d2, cur, live, prev, (r0, min(BLOCK_ROWS, hi - r0)), cb, w, dv, lq, acc)
            for q in range(trials):
                pots[q, b] = acc[q]
            if _fetch_add(state, STEP_DONE) == (st + 1) * nb - 1:  # the last chunk of step st: pick its winner, draw the next candidates
                c = st + 1
                best, best_pot = 0, np.inf
                for q in range(trials):  # ties: lowest candidate
                    pot = 0.0
                    for bb in range(nb):
                        pot += pots[q, bb]
                    if pot < best_pot:
                        best, best_pot = q, pot
                for j in range(d):
                    centers[c, cols[j]] = cand[best, j]
                for bb in range(nb):
                    tot[bb] = pots[best, bb]
                state[WINNER] = best
                if c + 1 < k:
                    _draw(x, perm, cols, new_d2, cur, live, best, tot, u[c], ch, idx, cand)
                _store_release(state, PUBLISHED, st + 1)


@numba.njit(inline="always", nogil=True, cache=True)
def _score_block(  # noqa: C901, PLR0912, PLR0917
    x: FArr,
    xt: FArr,
    xb: FArr,
    cand: FArr,
    new_d2: F64,
    cur: F64,
    live: I64,
    prev: int,
    block: tuple[int, int],
    cb: F64,
    w: F64,
    dv: FArr,
    lq: I64,
    acc: F64,
) -> None:
    """Score every candidate on one BLOCK_ROWS block (first row position r0, m rows), adding its potential to acc.

    The candidates’ partial D² grow over columns in stages of FIRST_STAGE, 2·FIRST_STAGE, …;
    a candidate leaves once no row’s partial sum is below its current D² (cb), since the rest only adds.
    """
    r0, m = block
    d = x.shape[1]
    trials = cand.shape[0]
    blk = r0 // BLOCK_ROWS
    fresh = live[blk, prev]  # the last winner changed this block: its values are current
    for r in range(BLOCK_ROWS):  # rows past m can’t improve and add nothing
        cb[r] = (new_d2[prev, r0 + r] if fresh else cur[r0 + r]) if r < m else 0.0
    sb = _bsum(cb)
    if not xt.shape[0]:
        for r in range(m):  # rows to columns, so the inner loop runs over rows: no horizontal sums
            for j in range(d):
                xb[0, j, r] = x[r0 + r, j]
    src, sblk = (xt, blk) if xt.shape[0] else (xb, 0)
    dv[:, :] = 0.0
    nl = trials  # candidates still live in this block, in order
    for q in range(trials):
        lq[q] = q
    j, e = 0, min(FIRST_STAGE, d) if xt.shape[0] else d
    while True:
        _score(src, sblk, cand, lq, nl, dv, j, e)
        j = e
        if e == d:
            break
        nn = 0
        for t in range(nl):
            q = lq[t]
            cnt = 0
            for r in range(BLOCK_ROWS):
                cnt += np.float64(dv[q, r]) < cb[r]
            if cnt:
                lq[nn] = q
                nn += 1
        nl = nn
        if nl == 0:  # the rest only adds: min(D², full) stays D², exactly
            break
        e = min(2 * e, d)
    # which candidates completed the block, their new D² there, and every candidate’s block potential
    for q in range(trials):
        live[blk, q] = 0
    for t in range(nl):
        live[blk, lq[t]] = 1
    if nl < trials and fresh:  # some candidate keeps this block’s D²: make it current
        for r in range(m):
            cur[r0 + r] = cb[r]
    for q in range(trials):
        if live[blk, q]:
            for r in range(BLOCK_ROWS):
                w[r] = min(cb[r], np.float64(dv[q, r]))
            for r in range(m):
                new_d2[q, r0 + r] = w[r]
            acc[q] += _bsum(w)
        else:
            acc[q] += sb


@numba.njit(inline="always", nogil=True, cache=True)
def _draw(  # noqa: PLR0917
    x: FArr, perm: I64, cols: I64, new_d2: F64, cur: F64, live: I64, prev: int, tot: F64, uc: F64, ch: int, idx: I64, cand: FArr
) -> None:
    """Draw the step’s candidates (D²-weighted, from the chunk totals and the last winner’s D²) and gather them in column order.

    Candidate q is the row whose cumulative D² first exceeds uc[q]·total: the chunk from the chunk totals, then the row inside it.
    """
    nb = tot.shape[0]
    row = new_d2[prev]
    total = 0.0
    for b in range(nb):
        total += tot[b]
    for q in range(cand.shape[0]):
        target = uc[q] * total
        b, cum = 0, 0.0
        while b < nb - 1 and cum + tot[b] <= target:
            cum += tot[b]
            b += 1
        lo, hi = b * ch, min(cur.shape[0], (b + 1) * ch)
        pick = lo  # if rounding between chunk totals and row sums leaves cum ≤ target: the last row that can be drawn
        for i in range(lo, hi):
            v = row[i] if live[i // BLOCK_ROWS, prev] else cur[i]  # the last winner’s value where it was computed for i’s block
            if v > 0.0:
                pick = i
            cum += v
            if cum > target:
                pick = i
                break
        idx[q] = perm[pick]
        for j in range(cand.shape[1]):
            cand[q, j] = x[idx[q], cols[j]]


@numba.njit(inline="always", nogil=True, cache=True)
def _bsum(w: F64) -> float:
    """Sum of a BLOCK_ROWS block in a fixed order (BSUM_WAYS interleaved partial sums, then a fixed tree)."""
    s0 = s1 = s2 = s3 = s4 = s5 = s6 = s7 = 0.0
    for g in range(0, BLOCK_ROWS, BSUM_WAYS):
        s0 += w[g]
        s1 += w[g + 1]
        s2 += w[g + 2]
        s3 += w[g + 3]
        s4 += w[g + 4]
        s5 += w[g + 5]
        s6 += w[g + 6]
        s7 += w[g + 7]
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))


@njit
def _grid_order(x: FArr) -> tuple[I64, I64]:
    """Row order (stable counting sort on the grid key) and column order (variance, descending), from the data alone.

    Variance and range come from a fixed subsample (every n // GRID_SAMPLE-th row), so the order doesn’t depend on threads.
    """
    n = x.shape[0]
    lo, hi, var = _sample_stats(x, max(1, n // GRID_SAMPLE))
    cols = np.argsort(-var, kind="mergesort")
    g = min(GRID_DIMS, x.shape[1])
    scale = np.zeros(g)
    for t in range(g):
        c = cols[t]
        if hi[c] > lo[c]:
            scale[t] = GRID_LEVELS / (hi[c] - lo[c])
    nk = GRID_LEVELS**g
    nc = (n + CHUNK - 1) // CHUNK
    key = np.empty(n, dtype=np.int64)
    hist = np.zeros((nc, nk), dtype=np.int64)
    for b in numba.prange(nc):
        for i in range(b * CHUNK, min(n, (b + 1) * CHUNK)):
            kk = _grid_key(x, i, cols, lo, scale)
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
def _grid_key(x: FArr, i: int, cols: I64, lo: F64, scale: F64) -> int:
    """Row i’s grid cell: GRID_LEVELS per column on the first scale.shape[0] columns of `cols`."""
    kk = 0
    for t in range(scale.shape[0]):
        c = cols[t]
        q = int((np.float64(x[i, c]) - lo[c]) * scale[t])
        kk = kk * GRID_LEVELS + min(max(q, 0), GRID_LEVELS - 1)
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
    """xt[b, j, r] = x[perm[b·BLOCK_ROWS + r], cols[j]]; rows past the end stay zero.

    Rows come in grid order, i.e. scattered in x: prefetching pfd rows ahead (0: off) turns the latency-bound reads into a stream.
    """
    n, d = x.shape
    for b in numba.prange(xt.shape[0]):
        for r in range(min(BLOCK_ROWS, n - b * BLOCK_ROWS)):
            p = b * BLOCK_ROWS + r
            if pfd and p + pfd < n:
                _prefetch_row(x, perm[p + pfd])
            i = perm[p]
            for j in range(d):
                xt[b, j, r] = x[i, cols[j]]
