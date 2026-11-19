# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import math

import numba
import numpy as np

from fast_array_utils.numba import njit

from ._sync import SPIN_BEFORE_YIELD, _claim_range, _fetch_add, _load_acquire, _prefetch_distance, _prefetch_row, _store_release, _workers, _yield_cpu
from ._tiles import SEED_ROWS, _score


CHUNK = 1 << 13  # rows per chunk of every reduction; fixed, so results don't depend on threads


SEED_CHUNK = 1 << 10  # max rows per chunk of the seeding sums; chunks depend only on n, so seeds don't depend on threads


SEED_COPY_BYTES = 128 << 20  # x up to this size gets a block-transposed copy, so seeding steps don't transpose


GRID_DIMS, GRID_BITS = 4, 3  # rows ordered by a grid key on the 4 highest-variance columns (8 levels each): nearby rows share blocks


@njit
def _grid_order(x):
    """Row order (stable counting sort on the grid key) and column order (variance, descending), from the data alone.

    Variance and range come from a fixed subsample (every n // 8192-th row), so the order doesn't depend on threads.
    """
    n, d = x.shape
    step = max(1, n // 8192)
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
    cols = np.argsort(-var, kind="mergesort")
    g = min(GRID_DIMS, d)
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
            kk = 0
            for t in range(g):
                c = cols[t]
                q = int((np.float64(x[i, c]) - lo[c]) * scale[t])
                kk = kk * levels + min(max(q, 0), levels - 1)
            key[i] = kk
            hist[b, kk] += 1
    pos = np.empty((nc, nk), dtype=np.int64)
    s = 0
    for kk in range(nk):
        for b in range(nc):
            pos[b, kk] = s
            s += hist[b, kk]
    perm = np.empty(n, dtype=np.int64)
    for b in numba.prange(nc):
        for i in range(b * CHUNK, min(n, (b + 1) * CHUNK)):
            kk = key[i]
            perm[pos[b, kk]] = i
            pos[b, kk] += 1
    return perm, cols


@njit
def _block_transpose(x, perm, cols, xt, pfd):
    """xt[b, j, r] = x[perm[b·SEED_ROWS + r], cols[j]]; rows past the end stay zero.

    Rows come in grid order, i.e. scattered in x: prefetching pfd rows ahead (0: off) turns the latency-bound reads into
    a stream.
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


@numba.njit(inline="always", nogil=True, cache=True)
def _d2(row, cur, live, q, i):
    """Return the current D² of row position i: the last winner q's value where it was computed for i's block, else `cur`."""
    return row[i] if live[i // SEED_ROWS, q] else cur[i]


@numba.njit(inline="always", nogil=True, cache=True)
def _pick(row, cur, live, q, tot, target, chunk):
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
def _bsum(w):
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


@numba.njit(inline="always", nogil=True, cache=True)
def _draw(x, perm, cols, new_d2, cur, live, prev, tot, uc, ch, idx, cand):
    """Draw the step's candidates (D²-weighted, from the chunk totals and the last winner's D²) and gather them in column order."""
    total = 0.0
    for b in range(tot.shape[0]):
        total += tot[b]
    for q in range(cand.shape[0]):
        idx[q] = perm[_pick(new_d2[prev], cur, live, prev, tot, uc[q] * total, ch)]
        for j in range(cand.shape[1]):
            cand[q, j] = x[idx[q], cols[j]]


@njit
def _seed(x, xt, perm, cols, first, u, centers, nthreads, pfd):
    """Greedy k-means++ in one compiled loop: per step, draw the candidates and score all of them in one parallel pass.

    Row positions follow `perm` (rows of x in grid order, so nearby rows share a 64-row block). With the
    transposed copy, columns follow `cols` and a candidate's block stops early once no row in it can
    improve (partial sums of squares only grow, so min(D², full) stays D² exactly). A candidate's D² is
    written only for blocks it completed (`live`); `cur` holds everyone else's. Block sums use a fixed
    order and chunk sums are added in chunk order, so the result doesn't depend on the thread count.
    """
    n, d = x.shape
    k, trials = centers.shape[0], u.shape[1]
    ch = min(SEED_CHUNK, max(SEED_ROWS, (n // 64 + SEED_ROWS - 1) // SEED_ROWS * SEED_ROWS))  # about n/64 rows: small n still gets many chunks
    nb = (n + ch - 1) // ch
    nblk = (n + SEED_ROWS - 1) // SEED_ROWS
    tot, cur = np.empty(nb), np.empty(n)
    new_d2, pots = np.empty((trials, n)), np.empty((trials, nb))
    live = np.zeros((nblk, trials), dtype=np.int64)  # per block, contiguous: no false sharing between threads' chunks
    cand, idx = np.empty((trials, d), dtype=x.dtype), np.empty(trials, dtype=np.int64)  # in x's dtype: float32 data, float32 lanes
    for j in range(d):
        centers[0, j] = x[first, j]
    for b in numba.prange(nb):
        s = 0.0
        for i in range(b * ch, min(n, (b + 1) * ch)):
            if pfd and i + pfd < n:  # rows in grid order are scattered in x
                _prefetch_row(x, perm[i + pfd])
            v = 0.0
            for j in range(d):
                t = np.float64(x[perm[i], j]) - centers[0, j]
                v += t * t
            cur[i] = v
            s += v
        tot[b] = s
    state = np.zeros(32, dtype=np.int64)  # [8] chunks done, [16] published step, [24] last winner (separate cache lines)
    rcnt = np.zeros(8 * nthreads, dtype=np.int64)  # claims so far from each worker's chunk range, one cache line each
    if k > 1:
        _draw(x, perm, cols, new_d2, cur, live, 0, tot, u[0], ch, idx, cand)
        for _w in numba.prange(nthreads):  # one region for all steps: per-region wake-ups cost more than small steps' work
            acc = np.empty(trials)
            xb = np.zeros((1, d if not xt.shape[0] else 0, SEED_ROWS), dtype=x.dtype)  # the no-copy branch's transposed block
            dv = np.empty((trials, SEED_ROWS), dtype=x.dtype)
            cb, w, lq = np.empty(SEED_ROWS), np.empty(SEED_ROWS), np.empty(trials, dtype=np.int64)
            own, st, nv = _w % nthreads, 0, 0  # chunks of the own range first (they stay in this core's caches), then steal
            while True:  # nobody waits for a particular thread: any worker can claim any chunk, so it can't deadlock
                b = -1
                while st < k - 1:
                    b = _claim_range(rcnt, (own + nv) % nthreads, st, nb, nthreads)
                    if b >= 0:
                        break
                    nv += 1
                    if nv == nthreads:  # every range of step st is claimed: on to the next step
                        st, nv = st + 1, 0
                if st >= k - 1:
                    break
                spins = 0
                while _load_acquire(state, 16) < st:  # step st's candidates are drawn by whoever finished step st - 1
                    spins += 1
                    if spins > SPIN_BEFORE_YIELD:
                        _yield_cpu()
                prev = state[24]
                acc[:] = 0.0
                hi = min(n, (b + 1) * ch)
                for r0 in range(b * ch, hi, SEED_ROWS):
                    m = min(SEED_ROWS, hi - r0)
                    blk = r0 // SEED_ROWS
                    fresh = live[blk, prev]  # the last winner changed this block: its values are current
                    for r in range(SEED_ROWS):  # rows past m can't improve and add nothing
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
                    j, e = 0, min(4, d) if xt.shape[0] else d
                    while True:  # columns in stages of 4, 8, 16, 32, ...; a candidate leaves once no row's partial sum is below its D²
                        _score(src, sblk, cand, lq, nl, dv, j, e)
                        j = e
                        if e == d:
                            break
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
                        if nl == 0:  # the rest only adds: min(D², full) stays D², exactly
                            break
                        e = min(2 * e, d)
                    for q in range(trials):
                        live[blk, q] = 0
                    for t in range(nl):
                        live[blk, lq[t]] = 1
                    if nl < trials and fresh:  # some candidate keeps this block's D²: make it current
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
                for q in range(trials):
                    pots[q, b] = acc[q]
                if _fetch_add(state, 8) == (st + 1) * nb - 1:  # the last chunk of step st: pick its winner, draw the next candidates
                    c = st + 1
                    best, best_pot = 0, np.inf
                    for q in range(trials):  # ties: lowest candidate
                        sq = 0.0
                        for bb in range(nb):
                            sq += pots[q, bb]
                        if sq < best_pot:
                            best, best_pot = q, sq
                    for j in range(d):
                        centers[c, cols[j]] = cand[best, j]
                    for bb in range(nb):
                        tot[bb] = pots[best, bb]
                    state[24] = best
                    if c + 1 < k:
                        _draw(x, perm, cols, new_d2, cur, live, best, tot, u[c], ch, idx, cand)
                    _store_release(state, 16, st + 1)


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
    _seed(
        x, xt, perm, cols, first, u, centers, _workers(x.shape[0] * trials * x.shape[1]), _prefetch_distance(x)
    )  # passed in: get_num_threads inside a kernel makes it uncacheable
    return centers
