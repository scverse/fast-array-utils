# SPDX-License-Identifier: MPL-2.0
"""Deterministic, exact numba k-means (greedy k-means++ seeding + Lloyd with Hamerly bounds).


Guarantees
- Exact: labels are those of plain float64 Lloyd from the same seeds. Approximate float32 distances only rank
  centers; a rigorous per-center rounding allowance decides when a label is certain, and every center that could
  still be nearest is rechecked in float64 (`_dot_allowance`, `_scan_queue_dot`).
- Deterministic: results depend only on the data, `k` and the seed, not on the thread count or threading layer.
  All reductions run over fixed row chunks combined in chunk order; work is claimed dynamically, but which thread
  computes a chunk never changes its result.

Structure
- `kmeans_plusplus` → `_seed`: rows in grid order with a block-transposed copy (x ≤ 128 MB), candidates scored by
  FMA register tiles (`_make_seed_tile`) with an exact staged early exit; all steps run in one persistent parallel
  region (`_claim_range`: own chunk range first for cache affinity, CAS stealing; the last finisher of a step draws
  the next candidates).
- `kmeans` → `_hamerly`: Hamerly bounds, the whole loop in one persistent region (phases: separations,
  bounds pass with queued FMA-tile scans, center update); incremental per-chunk center sums, drift-corrected stale
  separations, prefetched gathers for large x.
"""

from __future__ import annotations

import math
import sys
from typing import NamedTuple

import numba
import numpy as np
from fast_array_utils.numba import njit
from llvmlite import ir
from numba.core import cgutils, types
from numba.extending import intrinsic


CHUNK = 1 << 13  # rows per chunk of every reduction; fixed, so results don't depend on threads


@numba.njit(inline="always", nogil=True, cache=True)
def _dot_allowance(cn, unit, tiny, d):
    """(a, b) with a + b·‖x‖ ≥ the rounding error of v = fl(‖c‖²) − 2·fma_dot(x, fl(c)) for one center.

    ‖c‖² rounded to x's dtype: unit·‖c‖²; c rounded: 2·unit·‖x‖‖c‖; the FMA dot over d terms: 2·γ_d·‖x‖‖c‖ with
    γ_d ≈ d·unit (Higham, §3.1); the final subtraction (float64; for float64 data at the same unit): unit·(‖c‖² +
    2‖x‖‖c‖). The factor 1.01 absorbs second-order terms (d·unit ≪ 0.01); the tiny term covers underflow
    (tiny = the dtype's smallest subnormal). cn = ‖c‖² in float64.
    """
    return 1.01 * unit * 2.0 * cn + 4.0 * (d + 1) * tiny, 1.01 * unit * 2.0 * (d + 2) * math.sqrt(cn)


class KMeansResult(NamedTuple):
    """Result of :func:`kmeans`."""

    labels: np.ndarray
    """Index of each row's center, shape (n,), int64."""
    centers: np.ndarray
    """Final centers, shape (k, d), float64."""
    inertia: float
    """Sum of squared distances of the rows to their centers."""
    n_iter: int
    """Lloyd iterations run."""



WORKQUEUE_BIG_STEP = 50_000_000  # terms per step of a persistent region above which workqueue uses every thread


def _workers(step_terms: int) -> int:
    """Worker iterations for a persistent region whose steps (between barriers) do about `step_terms` terms of work.

    On numba's workqueue layer the main thread waits outside the pool, so with as many spinning workers as CPUs the
    barriers get slow: small steps use one worker fewer, big steps (barriers rare next to the work) all of
    them. TBB and OpenMP use all threads (their main thread works). Only speed depends on this: work
    is claimed, so results don't.
    """
    n = numba.get_num_threads()
    try:
        layer = numba.threading_layer()
    except ValueError:  # no parallel region has run yet in this process
        return n
    return max(1, n - 1) if layer == "workqueue" and step_terms < WORKQUEUE_BIG_STEP else n


def _validate(x: np.ndarray, k: int) -> np.ndarray:
    """C-contiguous float32 or float64 (other dtypes become float64); 1 ≤ k ≤ n."""
    x = np.ascontiguousarray(x)
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float64)
    if not 1 <= k <= x.shape[0]:
        msg = f"k={k} must be between 1 and the number of rows ({x.shape[0]})"
        raise ValueError(msg)
    return x


# ----------------------------------------------------------------------------- seeding
SEED_CHUNK = 1 << 10  # max rows per chunk of the seeding sums; chunks depend only on n, so seeds don't depend on threads
SEED_ROWS = 64  # rows per transposed block in the candidate pass; ≤ 32 gets unrolled into scalar code
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


PREFETCH_ROWS = 16  # rows of x prefetched ahead in gathers by row index...
PREFETCH_MIN_BYTES = 16 << 20  # ...when x is larger than this (smaller x stays in cache; prefetching it only costs)


def _prefetch_distance(x: np.ndarray) -> int:
    return PREFETCH_ROWS if x.nbytes > PREFETCH_MIN_BYTES else 0


@intrinsic
def _prefetch_row(typingctx, arr, i):
    """Prefetch the first cache lines of row i of a C-contiguous 2-D array (llvm.prefetch; a hint, no effect on results)."""
    sig = types.void(arr, types.intp)

    def codegen(context, builder, signature, args):
        aryty = signature.args[0]
        ary = context.make_array(aryty)(context, builder, args[0])
        ptr = cgutils.get_item_pointer(context, builder, aryty, ary, [args[1], context.get_constant(types.intp, 0)])
        i8p = ir.IntType(8).as_pointer()
        fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.VoidType(), [i8p, ir.IntType(32), ir.IntType(32), ir.IntType(32)]), "llvm.prefetch.p0")
        rowbytes = builder.mul(builder.extract_value(ary.shape, 1), context.get_constant(types.intp, context.get_abi_sizeof(context.get_data_type(aryty.dtype))))
        base = builder.bitcast(ptr, i8p)
        with cgutils.for_range_slice(builder, context.get_constant(types.intp, 0), rowbytes, context.get_constant(types.intp, 64), context.get_value_type(types.intp)) as (off, _):
            builder.call(fn, [builder.gep(base, [off]), ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 3), ir.Constant(ir.IntType(32), 1)])
        return context.get_dummy_value()

    return sig, codegen


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
    """Current D² of row position i: the last winner q's value where it was computed for i's block, else `cur`."""
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


def _vptr(builder, base, off, vt):
    """Pointer to the vector of type vt at element `off` of `base`."""
    return builder.bitcast(builder.gep(base, [off]), vt.as_pointer())


def _splat(builder, val, vt):
    """The scalar `val` in every lane of a vector of type vt."""
    undef = ir.Constant(vt, ir.Undefined)
    lane0 = builder.insert_element(undef, val, ir.Constant(ir.IntType(32), 0))
    return builder.shuffle_vector(lane0, undef, ir.Constant(ir.VectorType(ir.IntType(32), vt.count), [0] * vt.count))


def _fma(builder, vt):
    """llvm.fma for vectors of type vt (correctly rounded on every ISA)."""
    name = f"llvm.fma.v{vt.count}{'f32' if isinstance(vt.element, ir.FloatType) else 'f64'}"
    return cgutils.get_or_insert_function(builder.module, ir.FunctionType(vt, [vt, vt, vt]), name)


def _make_seed_tile(nq, rows):
    """An intrinsic scoring nq live candidates (lq[t0:t0+nq]) on `rows` rows (r0...) of block `blk` over columns [j0, j1).

    dv[q, r0+r] += (src[blk, j, r0+r] - cand[q, j])² for each column in order, as llvm.fma(t, t, acc): correctly rounded,
    so every value is the same on any FMA ISA and for any tile shape (each lane runs the same chain). Accumulators stay
    in 16-lane vector registers and each x vector feeds nq candidates (numba keeps array accumulators in L1).
    fma(t, t, acc) ≥ acc, so partial sums still only grow and the early exit stays exact.
    """
    lanes = 16
    nv = rows // lanes

    @intrinsic
    def tile(typingctx, src, blk, cand, lq, t0, dv, r0, j0, j1):
        sig = types.void(src, types.intp, cand, lq, types.intp, dv, types.intp, types.intp, types.intp)

        def codegen(context, builder, signature, args):
            sty, _, cty, lty, _, dty = signature.args[:6]
            sa = context.make_array(sty)(context, builder, args[0])
            ca = context.make_array(cty)(context, builder, args[2])
            la = context.make_array(lty)(context, builder, args[3])
            da = context.make_array(dty)(context, builder, args[5])
            blk_, t0_, r0_, j0_, j1_ = args[1], args[4], args[6], args[7], args[8]
            intp = context.get_value_type(types.intp)
            elt = context.get_value_type(sty.dtype)
            vt = ir.VectorType(elt, lanes)
            fma = _fma(builder, vt)
            undef = ir.Constant(vt, ir.Undefined)
            d, nr = builder.extract_value(sa.shape, 1), builder.extract_value(sa.shape, 2)
            cd, dr = builder.extract_value(ca.shape, 1), builder.extract_value(da.shape, 1)

            def k(v):
                return ir.Constant(intp, v)

            def vptr(base, off):
                return _vptr(builder, base, off, vt)

            qs = [builder.load(builder.gep(la.data, [builder.add(t0_, k(i))])) for i in range(nq)]
            accs = []
            for i in range(nq):
                for h in range(nv):
                    accs.append(cgutils.alloca_once_value(builder, undef))
                    off = builder.add(builder.mul(qs[i], dr), builder.add(r0_, k(h * lanes)))
                    builder.store(builder.load(vptr(da.data, off), align=4), accs[-1])
            base = builder.add(builder.mul(blk_, builder.mul(d, nr)), r0_)
            with cgutils.for_range_slice(builder, j0_, j1_, k(1), intp) as (j, _):
                row = builder.add(base, builder.mul(j, nr))
                xv = [builder.load(vptr(sa.data, builder.add(row, k(h * lanes))), align=4) for h in range(nv)]
                for i in range(nq):
                    cval = builder.load(builder.gep(ca.data, [builder.add(builder.mul(qs[i], cd), j)]))
                    cv = _splat(builder, cval, vt)
                    for h in range(nv):
                        t = builder.fsub(xv[h], cv)
                        a = accs[i * nv + h]
                        builder.store(builder.call(fma, [t, t, builder.load(a)]), a)
            for i in range(nq):
                for h in range(nv):
                    off = builder.add(builder.mul(qs[i], dr), builder.add(r0_, k(h * lanes)))
                    builder.store(builder.load(accs[i * nv + h]), vptr(da.data, off), align=4)
            return context.get_dummy_value()

        return sig, codegen

    return tile


_seed_tile4, _seed_tile2, _seed_tile1 = _make_seed_tile(4, 32), _make_seed_tile(2, 64), _make_seed_tile(1, 64)  # fastest shapes


@numba.njit(inline="always", nogil=True, cache=True)
def _score(src, blk, cand, lq, nl, dv, j0, j1):
    """Add columns [j0, j1) to the D² of the nl live candidates lq[:nl] for one SEED_ROWS block (rows 0..63)."""
    t = 0
    while t + 4 <= nl:
        _seed_tile4(src, blk, cand, lq, t, dv, 0, j0, j1)
        _seed_tile4(src, blk, cand, lq, t, dv, 32, j0, j1)
        t += 4
    if t + 2 <= nl:
        _seed_tile2(src, blk, cand, lq, t, dv, 0, j0, j1)
        t += 2
    if t < nl:
        _seed_tile1(src, blk, cand, lq, t, dv, 0, j0, j1)


@intrinsic
def _fetch_add(typingctx, arr, i):
    """Atomically arr[i] += 1, returning the old value (sequentially consistent)."""
    sig = types.int64(arr, types.intp)

    def codegen(context, builder, signature, args):
        aryty = signature.args[0]
        ptr = cgutils.get_item_pointer(context, builder, aryty, context.make_array(aryty)(context, builder, args[0]), [args[1]])
        return builder.atomic_rmw("add", ptr, context.get_constant(types.int64, 1), "seq_cst")

    return sig, codegen


@intrinsic
def _load_acquire(typingctx, arr, i):
    """arr[i], loaded with acquire ordering: writes published before the matching release store are visible after it."""
    sig = types.int64(arr, types.intp)

    def codegen(context, builder, signature, args):
        aryty = signature.args[0]
        ptr = cgutils.get_item_pointer(context, builder, aryty, context.make_array(aryty)(context, builder, args[0]), [args[1]])
        return builder.load_atomic(ptr, "acquire", 8)

    return sig, codegen


@intrinsic
def _store_release(typingctx, arr, i, v):
    """arr[i] = v with release ordering: publishes every earlier write to threads that acquire-load arr[i]."""
    sig = types.void(arr, types.intp, types.int64)

    def codegen(context, builder, signature, args):
        aryty = signature.args[0]
        ptr = cgutils.get_item_pointer(context, builder, aryty, context.make_array(aryty)(context, builder, args[0]), [args[1]])
        builder.store_atomic(args[2], ptr, "release", 8)
        return context.get_dummy_value()

    return sig, codegen


_YIELD_SYMBOL = "SwitchToThread" if sys.platform == "win32" else "sched_yield"
SPIN_BEFORE_YIELD = 64  # spin-wait iterations before a waiting worker yields its CPU


@intrinsic
def _yield_cpu(typingctx):
    """Give the CPU to another runnable thread (sched_yield; SwitchToThread on Windows), by symbol so kernels stay cacheable.

    Spinning workers must yield: with as many threads as CPUs plus a waiting main thread (numba's workqueue layer),
    a pure spin starves the thread doing the work for whole scheduler slices.
    """
    def codegen(context, builder, signature, args):
        fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.IntType(32), []), _YIELD_SYMBOL)
        builder.call(fn, [])
        return context.get_dummy_value()

    return types.void(), codegen


# monotonic clock by symbol (cacheable): clock_gettime(CLOCK_MONOTONIC) on Linux (id 1) and macOS (id 6), else QPC ticks
_CLOCK_SYMBOL, _CLOCK_ID = ("QueryPerformanceCounter", 0) if sys.platform == "win32" else ("clock_gettime", 6 if sys.platform == "darwin" else 1)


@intrinsic
def _clock_ticks(typingctx):
    """A monotonic clock reading (ns; QueryPerformanceCounter ticks on Windows). Only differences of readings are used.

    The ball scans are chosen per pass by measured cost; results don't depend on the choice."""
    def codegen(context, builder, signature, args):
        i64 = ir.IntType(64)
        if _CLOCK_SYMBOL == "QueryPerformanceCounter":
            slot = cgutils.alloca_once(builder, i64)
            fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.IntType(32), [i64.as_pointer()]), _CLOCK_SYMBOL)
            builder.call(fn, [slot])
            return builder.load(slot)
        ts = cgutils.alloca_once(builder, ir.ArrayType(i64, 2))
        fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.IntType(32), [ir.IntType(32), ts.type]), _CLOCK_SYMBOL)
        builder.call(fn, [ir.Constant(ir.IntType(32), _CLOCK_ID), ts])
        zero = ir.Constant(ir.IntType(32), 0)
        sec = builder.load(builder.gep(ts, [zero, zero]))
        nsec = builder.load(builder.gep(ts, [zero, ir.Constant(ir.IntType(32), 1)]))
        return builder.add(builder.mul(sec, ir.Constant(i64, 1_000_000_000)), nsec)

    return types.int64(), codegen


@intrinsic
def _cas(typingctx, arr, i, expected, new):
    """Atomically arr[i] = new if arr[i] == expected; returns the old value (sequentially consistent)."""
    sig = types.int64(arr, types.intp, types.int64, types.int64)

    def codegen(context, builder, signature, args):
        aryty = signature.args[0]
        ptr = cgutils.get_item_pointer(context, builder, aryty, context.make_array(aryty)(context, builder, args[0]), [args[1]])
        return builder.extract_value(builder.cmpxchg(ptr, args[2], args[3], "seq_cst", "seq_cst"), 0)

    return sig, codegen


@numba.njit(inline="always", nogil=True, cache=True)
def _claim_range(rcnt, v, st, nb, nr):
    """Claim the next chunk of step st from range v of nr (chunks v·nb/nr ...), or −1 if all are claimed.

    rcnt[8v] counts the range's claims over all steps; a claim for step st succeeds only while it's below (st + 1)·len,
    so a worker still looking at step st can never take a chunk of step st + 1.
    """
    lo, hi = (v * nb) // nr, ((v + 1) * nb) // nr
    n = hi - lo
    if n == 0:
        return -1
    c, lim = _load_acquire(rcnt, 8 * v), (st + 1) * n
    while c < lim:
        old = _cas(rcnt, 8 * v, c, c + 1)
        if old == c:
            return lo + c - st * n
        c = old
    return -1


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


def kmeans_plusplus(x: np.ndarray, k: int, *, rng: np.random.Generator, n_local_trials: int | None = None) -> np.ndarray:
    """Greedy k-means++ seeding, deterministic for any thread count.

    Parameters
    ----------
    x
        Dense data, shape (n, d). float32 and float64 are used as they are; other dtypes become float64.
    k
        Number of centers, 1 ≤ k ≤ n.
    rng
        Source of the random draws.
    n_local_trials
        Candidates per step; default ``2 + int(ln k)``, as scikit-learn.

    Returns
    -------
    The k initial centers, shape (k, d), float64.
    """
    x = _validate(x, k)
    _check_finite(_norms_var(x)[0])
    return _plusplus(x, k, rng, n_local_trials)


def _check_finite(xnorm: np.ndarray) -> None:
    if not np.isfinite(xnorm.max()):  # xnorm ≥ 0: NaN and inf propagate through max; also trips if squares overflow
        msg = "x contains NaN or infinity"
        raise ValueError(msg)


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
    _seed(x, xt, perm, cols, first, u, centers, _workers(x.shape[0] * trials * x.shape[1]), _prefetch_distance(x))  # passed in: get_num_threads inside a kernel makes it uncacheable
    return centers


# ----------------------------------------------------------------------------- Lloyd (Hamerly bounds)
def kmeans(
    x: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    init: np.ndarray | None = None,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_local_trials: int | None = None,
    seed_sample: int | None = None,
    max_moved: int = 0,
) -> KMeansResult:
    """Exact Lloyd's k-means from greedy k-means++ seeds.

    The labels are those of plain float64 Lloyd from the same seeds, and results don't depend on the thread count
    or threading layer. Hamerly's bounds and exact rechecks of approximate distances skip most distance computations.

    Parameters
    ----------
    x
        Dense data, shape (n, d). float32 and float64 are used as they are; other dtypes become float64.
    k
        Number of clusters, 1 ≤ k ≤ n.
    rng
        Source of the random draws for seeding.
    init
        Initial centers, shape (k, d); seeding is skipped.
    max_iter
        Maximum number of Lloyd iterations.
    tol
        Stop once the squared center shift is at most ``tol`` times the mean feature variance (scikit-learn's rule).
    n_local_trials
        Seeding candidates per step; default ``2 + int(ln k)``, as scikit-learn.
    seed_sample
        Seed on this many random rows only.
    max_moved
        Also stop once a pass changes at most this many labels (0: only when nothing moves, or by ``tol``).

    Returns
    -------
    Labels, centers, inertia and the number of iterations.
    """
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
    it, inertia = _hamerly(x, xnorm, centers, _hamerly_bounds(n, k, d), max_iter, tol * mean_var, max_moved, labels, _workers(n * d),
                          float(np.finfo(x.dtype).eps) / 2, float(np.finfo(x.dtype).smallest_subnormal), _prefetch_distance(x),
                          np.zeros((k, k) if BALL_MIN_K <= k <= BALL_MAX_K and n >= BALL_MIN_ROWS * k else (1, 1)))
    return KMeansResult(labels, centers, inertia, it)


SAFETY = 1e-5  # relative margin on every bound test, so float rounding can never skip a needed scan


def _dot_tile_body(context, builder, xbty, xb, dvty, dv, r0v, c0v, lanes, center):
    """Shared codegen of `_dot_tile` and `_dot_tile_idx`; center(j, i) loads center c0+i's column j."""
    xa = context.make_array(xbty)(context, builder, xb)
    da = context.make_array(dvty)(context, builder, dv)
    intp = context.get_value_type(types.intp)
    elt = context.get_value_type(xbty.dtype)
    vt = ir.VectorType(elt, lanes)
    fma = _fma(builder, vt)
    accs = [cgutils.alloca_once_value(builder, ir.Constant(vt, [ir.Constant(elt, 0.0)] * lanes)) for _ in range(8)]
    d = builder.extract_value(xa.shape, 0)
    nr, dvr = builder.extract_value(xa.shape, 1), builder.extract_value(da.shape, 1)
    with cgutils.for_range(builder, d) as loop:
        j = loop.index
        xrow = builder.add(builder.mul(j, nr), r0v)
        xv = (builder.load(_vptr(builder, xa.data, xrow, vt), align=4),
              builder.load(_vptr(builder, xa.data, builder.add(xrow, ir.Constant(intp, lanes)), vt), align=4))
        for i in range(4):
            cv = _splat(builder, center(j, i), vt)
            for h in range(2):
                builder.store(builder.call(fma, [xv[h], cv, builder.load(accs[2 * i + h])]), accs[2 * i + h])
    for i in range(4):
        for h in range(2):
            off = builder.add(builder.mul(builder.add(c0v, ir.Constant(intp, i)), dvr), builder.add(r0v, ir.Constant(intp, lanes * h)))
            builder.store(builder.load(accs[2 * i + h]), _vptr(builder, da.data, off, vt), align=4)
    return context.get_dummy_value()


@intrinsic
def _dot_tile(typingctx, xb, ct, dv, r0, c0):
    """dv[c0+i, r0+r] = Σ_j xb[j, r0+r]·ct[j, c0+i] for i < 4, r < 2·L, accumulated with llvm.fma in vector registers;
    L = 16 lanes for float32 (512-bit vectors: 32 rows per call), 8 for float64 (16 rows).

    numba keeps array accumulators in memory (L1 load and store per term); this tile keeps 8 vectors in
    registers: LLVM's mem2reg/SROA promotes only values that are never in addressable memory, so the
    accumulators are LLVM vector values built with llvmlite's IRBuilder. FMA is correctly rounded, so the values are
    the same on every FMA ISA. Rows r0..r0+2L-1 and centers c0..c0+3 must exist; arrays C-contiguous.
    """
    sig = types.void(xb, ct, dv, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        xbty, ctty, dvty = signature.args[:3]
        ca = context.make_array(ctty)(context, builder, args[1])
        intp = context.get_value_type(types.intp)
        kc = builder.extract_value(ca.shape, 1)
        lanes = 16 if isinstance(context.get_value_type(xbty.dtype), ir.FloatType) else 8

        def center(j, i):
            return builder.load(builder.gep(ca.data, [builder.add(builder.mul(j, kc), builder.add(args[4], ir.Constant(intp, i)))]))

        return _dot_tile_body(context, builder, xbty, args[0], dvty, args[2], args[3], args[4], lanes, center)

    return sig, codegen


@intrinsic
def _dot_tile_idx(typingctx, xb, crow, idx, dv, r0, c0):
    """`_dot_tile` with the centers read by index from row-major `crow`: dv[c0+i, r0+r] = Σ_j xb[j, r0+r]·crow[idx[c0+i], j],
    8 lanes (16 rows per call).

    The ball scans tile a different subset of centers for every label; reading rows by index saves gathering
    them into a transposed copy. Same FMA chain per value as `_dot_tile`, so the same values.
    """
    sig = types.void(xb, crow, idx, dv, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        xbty, crty, idty, dvty = signature.args[:4]
        xa = context.make_array(xbty)(context, builder, args[0])
        ca = context.make_array(crty)(context, builder, args[1])
        ia = context.make_array(idty)(context, builder, args[2])
        intp = context.get_value_type(types.intp)
        d = builder.extract_value(xa.shape, 0)
        rows = [builder.mul(builder.load(builder.gep(ia.data, [builder.add(args[5], ir.Constant(intp, i))])), d) for i in range(4)]

        def center(j, i):
            return builder.load(builder.gep(ca.data, [builder.add(rows[i], j)]))

        return _dot_tile_body(context, builder, xbty, args[0], dvty, args[3], args[4], args[5], 8, center)

    return sig, codegen


@intrinsic
def _transpose8(typingctx, x, queue, s, xb, r0, j0):
    """xb[j0 + j, r0 + r] = x[queue[s + r], j0 + j] for r, j < 8: eight row vectors, a 3-stage shuffle network, eight
    column vectors instead of storing one scalar per element.

    Copies values only, so results don't change. Rows queue[s..s+7] and columns j0..j0+7 must exist; arrays C-contiguous.
    """
    sig = types.void(x, queue, types.intp, xb, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        xty, qty, _, xbty = signature.args[:4]
        xa = context.make_array(xty)(context, builder, args[0])
        qa = context.make_array(qty)(context, builder, args[1])
        ba = context.make_array(xbty)(context, builder, args[3])
        sv, r0v, j0v = args[2], args[4], args[5]
        intp = context.get_value_type(types.intp)
        elt = context.get_value_type(xty.dtype)
        vt = ir.VectorType(elt, 8)
        d = builder.extract_value(xa.shape, 1)
        nr = builder.extract_value(ba.shape, 1)

        def vptr(base, off):
            return _vptr(builder, base, off, vt)

        v = []
        for r in range(8):
            i = builder.load(builder.gep(qa.data, [builder.add(sv, ir.Constant(intp, r))]))
            v.append(builder.load(vptr(xa.data, builder.add(builder.mul(i, d), j0v)), align=4))
        i32 = ir.IntType(32)
        for lo, hi, st in (([0, 8, 2, 10, 4, 12, 6, 14], [1, 9, 3, 11, 5, 13, 7, 15], 1),
                           ([0, 1, 8, 9, 4, 5, 12, 13], [2, 3, 10, 11, 6, 7, 14, 15], 2),
                           ([0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15], 4)):
            nv = list(v)
            for q in range(8):
                if (q // st) % 2 == 0:
                    a, b = v[q], v[q + st]
                    nv[q] = builder.shuffle_vector(a, b, ir.Constant(ir.VectorType(i32, 8), lo))
                    nv[q + st] = builder.shuffle_vector(a, b, ir.Constant(ir.VectorType(i32, 8), hi))
            v = nv
        for j in range(8):  # v[j] is column j0 + j of the eight rows
            off = builder.add(builder.mul(builder.add(j0v, ir.Constant(intp, j)), nr), r0v)
            builder.store(v[j], vptr(ba.data, off), align=4)
        return context.get_dummy_value()

    return sig, codegen


SMALL_K = 64  # up to this many centers, rows failing the bound test go straight to the queued scan, without tightening


@numba.njit(inline="always", nogil=True, cache=True)
def _scan_queue_dot(x, xnorm, ct, cnorm, err_a, err_b, centers, queue, nq, labels, upper, lower, pfd):
    """Full scans of the queued rows, SEED_ROWS at a time: dot products x·c by FMA register tiles (4 centers x 16 rows,
    `_dot_tile`), then per row, vectorized over rows, the nearest by v = ‖c‖² − 2x·c and the two smallest of
    w_c = v_c − δ_c, where δ_c = err_a[c] + err_b[c]·‖x‖ bounds v_c's rounding error (`_dot_allowance`).
    The label is certain if every other center's w exceeds v1 + δ1; then the bounds come from the dot form with the
    allowances. Otherwise the centers that could be nearest (w_c ≤ v1 + δ1) get exact float64 distances."""
    d, k = ct.shape
    xb = np.zeros((d, SEED_ROWS), dtype=x.dtype)
    dv = np.empty((k, SEED_ROWS), dtype=x.dtype)
    d1, l1, l2, xs = np.empty(SEED_ROWS), np.empty(SEED_ROWS), np.empty(SEED_ROWS), np.zeros(SEED_ROWS)
    b1, lb1 = np.empty(SEED_ROWS, dtype=np.int64), np.empty(SEED_ROWS, dtype=np.int64)
    k4 = k - k % 4
    moved = 0
    for s0 in range(0, nq, SEED_ROWS):
        m = min(SEED_ROWS, nq - s0)
        d8, m8 = d - d % 8, m - m % 8
        for r in range(m):
            if pfd and s0 + r + pfd < nq:  # queued rows are scattered in x (a few % of rows at large n)
                _prefetch_row(x, queue[s0 + r + pfd])
            i = queue[s0 + r]
            xs[r] = math.sqrt(xnorm[i])
            for j in range(d8 if r < m8 else 0, d):  # the rest of groups of 8 rows go through 8×8 register transposes
                xb[j, r] = x[i, j]
        for r0 in range(0, m8, 8):
            for j0 in range(0, d8, 8):
                _transpose8(x, queue, s0 + r0, xb, r0, j0)
        for r0 in range(0, SEED_ROWS, 32 if xb.itemsize == 4 else 16):  # rows per `_dot_tile` call
            for c0 in range(0, k4, 4):
                _dot_tile(xb, ct, dv, r0, c0)
        for c in range(k4, k):
            dv[c, :] = 0
            for j in range(d):
                cj = ct[j, c]
                for r in range(SEED_ROWS):
                    dv[c, r] += xb[j, r] * cj
        for r in range(SEED_ROWS):
            d1[r], l1[r], l2[r] = np.inf, np.inf, np.inf
            b1[r], lb1[r] = 0, -1
        for c in range(k):  # vectorized over rows (ties: lowest index)
            cn, ea, eb = np.float64(cnorm[c]), err_a[c], err_b[c]
            for r in range(SEED_ROWS):
                v = cn - 2.0 * np.float64(dv[c, r])
                w = v - (ea + eb * xs[r])
                lt1 = v < d1[r]
                d1[r] = v if lt1 else d1[r]
                b1[r] = c if lt1 else b1[r]
                lw1 = w < l1[r]
                lw2 = w < l2[r]
                l2[r] = l1[r] if lw1 else (w if lw2 else l2[r])
                l1[r] = w if lw1 else l1[r]
                lb1[r] = c if lw1 else lb1[r]
        for r in range(m):
            i = queue[s0 + r]
            best, v1 = b1[r], d1[r]
            reach = v1 + err_a[best] + err_b[best] * xs[r]  # ≥ the nearest center's true v
            wo = l1[r] if lb1[r] != best else l2[r]  # smallest w over the other centers (inf for k = 1)
            if wo > reach:  # certain; bounds from the dot form with the allowances
                e1 = xnorm[i] + reach
                e2 = max(xnorm[i] + wo, 0.0)
            else:  # exact float64 distances to the centers that could be nearest; the others' w bound the second
                best, e1, e2, rest = -1, np.inf, np.inf, np.inf
                for c in range(k):
                    w = np.float64(cnorm[c]) - 2.0 * np.float64(dv[c, r]) - (err_a[c] + err_b[c] * xs[r])
                    if w <= reach:
                        e = 0.0
                        for j in range(d):
                            t = np.float64(x[i, j]) - centers[c, j]
                            e += t * t
                        if e < e1:  # ties: lowest index, as the reference
                            best, e1, e2 = c, e, e1
                        elif e < e2:
                            e2 = e
                    else:
                        rest = min(rest, w)
                e2 = min(e2, max(xnorm[i] + rest, 0.0))
            moved += labels[i] != best
            labels[i] = best
            upper[i] = math.sqrt(e1)
            lower[i] = math.sqrt(e2)
    return moved


BALL_MIN_K, BALL_MAX_K = 256, 2048  # k range with ball scans (k×k center distances: ≤ 32 MB)
BALL_FRACTION = 0.8  # ball scans only while plain-pass bounds leave this fraction failing (high-dimensional data with many centers fails far more often than well-separated blobs)
BALL_MIN_ROWS = 8  # ...and only with this many failing rows per label on average: tiles take 16 rows
BALL_PROBE = 8  # ...and only while measured cheaper per failing row than the plain pass; the other kind is re-timed after this many passes, then at doubling intervals while confirmed


@numba.njit(inline="always", nogil=True, cache=True)
def _ball_label(x, xnorm, crow, centers, cnorm, err_a, err_b, cd, acc, a, rows, labels, upper, lower, keys, cand, sidx, dv, xb, sel):
    """Exact scans of the rows labelled a that failed the bound test, against the centers that could be nearer.

    Every center nearer to row i than c_a lies in the ball |c_o − c_a| ≤ 2u_i (u_i = the exact distance to c_a, from
    the bound pass). Candidates are ordered by a lower bound on |c_o − c_a| (center distances cd at the last full
    computation minus the drift since, acc), rows by u; each 16-row group is tiled (`_dot_tile_idx`) against the
    candidates inside its largest radius. Labels as in `_scan_queue_dot`: the dot form with per-center rounding
    allowances, exact float64 distances when another candidate could be nearest. Second bound: the smaller of the
    candidates' and of |c_o − c_a| − u over the centers not scanned.
    """
    k, d = crow.shape
    m = rows.shape[0]
    rows[:] = rows[np.argsort(upper[rows])]
    rad_all = 2.0 * upper[rows[m - 1]] * (1 + 4 * SAFETY)
    kc, oute = 0, np.inf
    for o in range(k):
        lb = (cd[a, o] - acc[a] - acc[o]) * (1 - SAFETY)
        if lb <= rad_all:
            cand[kc] = o
            keys[kc] = lb
            kc += 1
        else:
            oute = min(oute, lb)
    srt = cand[:kc][np.argsort(keys[:kc])]
    kc4 = (kc + 3) // 4 * 4
    for t in range(kc4):
        sidx[t] = srt[min(t, kc - 1)]  # padding columns repeat a real center (ignored)
    xs, d1, l1, l2 = sel[0], sel[1], sel[2], sel[3]
    bb, lb1 = np.zeros(16, dtype=np.int64), np.zeros(16, dtype=np.int64)
    for s0 in range(0, m, 16):
        nr = min(16, m - s0)
        rad = 2.0 * upper[rows[s0 + nr - 1]] * (1 + 4 * SAFETY)
        p = kc
        while p > 0 and (cd[a, srt[p - 1]] - acc[a] - acc[srt[p - 1]]) * (1 - SAFETY) > rad:
            p -= 1
        p4 = min((p + 3) // 4 * 4, kc4)
        pe = min(p4, kc)  # candidates scanned (the padded tile columns may add a few beyond the radius: harmless)
        out = (cd[a, srt[pe]] - acc[a] - acc[srt[pe]]) * (1 - SAFETY) if pe < kc else oute
        for r in range(16):
            if r < nr:
                i = rows[s0 + r]
                xs[r] = math.sqrt(xnorm[i])
                for j in range(d):
                    xb[j, r] = x[i, j]
            else:
                xs[r] = 0.0
                for j in range(d):
                    xb[j, r] = 0
        for cq in range(0, p4, 4):
            _dot_tile_idx(xb, crow, sidx, dv, 0, cq)
        for r in range(16):
            d1[r], l1[r], l2[r] = np.inf, np.inf, np.inf
            bb[r], lb1[r] = 0, -1
        for t in range(pe):  # vectorized over the rows; dot-form ties fall to the exact rescans (w_other ≤ v1 ≤ reach)
            o = srt[t]
            cn, ea, eb = np.float64(cnorm[o]), err_a[o], err_b[o]
            for r in range(16):
                v = cn - 2.0 * np.float64(dv[t, r])
                w = v - (ea + eb * xs[r])
                lt1 = v < d1[r]
                d1[r] = v if lt1 else d1[r]
                bb[r] = t if lt1 else bb[r]
                lw1 = w < l1[r]
                lw2 = w < l2[r]
                l2[r] = l1[r] if lw1 else (w if lw2 else l2[r])
                l1[r] = w if lw1 else l1[r]
                lb1[r] = t if lw1 else lb1[r]
        for r in range(nr):
            i = rows[s0 + r]
            best = srt[bb[r]]
            reach = d1[r] + err_a[best] + err_b[best] * xs[r]  # ≥ the nearest candidate's true v
            wo = l1[r] if lb1[r] != bb[r] else l2[r]
            rest = (out - upper[i]) * (1 - SAFETY)  # every center not scanned is at least this far from row i
            if wo > reach:
                e1 = xnorm[i] + reach
                e2 = max(xnorm[i] + wo, 0.0)
            else:
                best, e1, e2, wr = -1, np.inf, np.inf, np.inf
                for t in range(pe):
                    o = srt[t]
                    w = np.float64(cnorm[o]) - 2.0 * np.float64(dv[t, r]) - (err_a[o] + err_b[o] * xs[r])
                    if w <= reach:
                        e = 0.0
                        for j in range(d):
                            q = np.float64(x[i, j]) - centers[o, j]
                            e += q * q
                        if e < e1 or (e == e1 and o < best):  # ties: lowest index, as the reference
                            e2 = e1
                            best, e1 = o, e
                        elif e < e2:
                            e2 = e
                    else:
                        wr = min(wr, w)
                e2 = min(e2, max(xnorm[i] + wr, 0.0))
            labels[i] = best
            upper[i] = math.sqrt(e1)
            lower[i] = max(min(math.sqrt(e2), rest), 0.0)


@numba.njit(nogil=True, cache=True)
def _ball_scatter(bounds, b0, b1, qn, qbuf, qold, lcnt, order):
    """Phase C2: the recorded rows of chunks b0..b1 into label order (chunk order, then row order, per label).

    The ball phases are separate functions (not inlined) so the plain Hamerly loop stays as small as without them."""
    for b in range(b0, b1):
        lo = bounds[b]
        for t in range(qn[b]):
            a = qold[lo + t]
            if a >= 0:  # rows without a label were scanned in phase C
                order[lcnt[b, a]] = qbuf[lo + t]
                lcnt[b, a] += 1


@numba.njit(nogil=True, cache=True)
def _ball_group(x, xnorm, crow, centers, cnorm, err_a, err_b, cd, acc, a0, a1, lstart, order, labels, upper, lower):
    """Phase E: ball scans of the labels a0..a1 (`_ball_label`)."""
    k, d = crow.shape
    keys, cand, sidx = np.empty(k), np.empty(k, dtype=np.int64), np.empty(k + 4, dtype=np.int64)
    dv, xb, sel = np.empty((k + 4, 16), dtype=x.dtype), np.zeros((d, 16), dtype=x.dtype), np.empty((4, 16))
    for a in range(a0, a1):
        if lstart[a + 1] > lstart[a]:
            _ball_label(x, xnorm, crow, centers, cnorm, err_a, err_b, cd, acc, a, order[lstart[a]:lstart[a + 1]],
                        labels, upper, lower, keys, cand, sidx, dv, xb, sel)


@numba.njit(nogil=True, cache=True)
def _ball_sums(x, bounds, b0, b1, qn, qbuf, qold, labels, sums, counts, tch, dirty, moved):
    """Phase C3: chunk sums of chunks b0..b1 for their rows that moved in phases C and E, as phase C does
    without ball scans (many moves: the chunk's sums afresh in row order; few: those rows moved, in row order)."""
    for b in range(b0, b1):
        lo, hi = bounds[b], bounds[b + 1]
        nq = qn[b]
        mv = 0
        for t in range(nq):
            mv += qold[lo + t] != labels[qbuf[lo + t]]
        if mv:
            _chunk_sums(x, b, lo, hi, qbuf, qold, lo, nq, mv, labels, sums, counts, tch, dirty)
        moved[b] = mv


@numba.njit(nogil=True, cache=True)
def _chunk_sums(x, b, lo, hi, rows, old, r0, nq, mv, labels, sums, counts, tch, dirty):
    """Chunk b's center sums after mv > 0 of its nq recorded rows (rows[r0 + t], previous labels old[r0 + t]) moved.

    Many moves (or rows taken by the empty-center rule): the chunk's sums afresh, rows in order. Few: those rows move
    between their old and new centers' sums, in row order. Both orders depend only on the data, not on threads.
    """
    d = x.shape[1]
    if dirty[b] or mv * MOVED_RECOMPUTE > hi - lo:
        dirty[b] = 0
        sums[b] = 0.0
        counts[b] = 0
        for t in range(nq):  # centers rows left (their sums in this chunk change too)
            if old[r0 + t] >= 0:
                tch[old[r0 + t]] = 1
        for i in range(lo, hi):
            best = labels[i]
            tch[best] = 1
            counts[b, best] += 1
            for j in range(d):
                sums[b, best, j] += x[i, j]
    else:
        for t in range(nq):
            i, a0, a1 = rows[r0 + t], old[r0 + t], labels[rows[r0 + t]]
            if a0 != a1:
                tch[a0], tch[a1] = 1, 1
                counts[b, a0] -= 1
                counts[b, a1] += 1
                for j in range(d):
                    sums[b, a0, j] -= x[i, j]
                    sums[b, a1, j] += x[i, j]


@numba.njit(nogil=True, cache=True)
def _ball_offsets(lcnt, lstart, gsplit):
    """End of phase C with ball scans: per-chunk scatter offsets, label starts and label groups of about equal rows."""
    nb, k = lcnt.shape
    ne = gsplit.shape[0] - 1
    tot_rows = 0
    for c in range(k):
        lstart[c] = tot_rows
        for bb in range(nb):
            cnt = lcnt[bb, c]
            lcnt[bb, c] = tot_rows
            tot_rows += cnt
    lstart[k] = tot_rows
    c = 0
    for e in range(ne):
        gsplit[e] = c
        target = (tot_rows * (e + 1)) // ne
        while c < k - (ne - 1 - e) and lstart[c + 1] <= target:
            c += 1
    gsplit[ne] = k


SUM_BUDGET = 1 << 22  # max float64 entries of per-chunk center sums (32 MB)


def _hamerly_bounds(n: int, k: int, d: int) -> np.ndarray:
    """Fixed row chunks from the shape alone (≥ 256 rows each, ≤ 256 chunks, sums within SUM_BUDGET), so results don't depend on threads."""
    nb = max(1, min(-(-n // 256), 256, SUM_BUDGET // (k * d)))
    return (np.arange(nb + 1, dtype=np.int64) * n) // nb


@numba.njit(inline="always", nogil=True, cache=True)
def _top2(drift, fstate):
    """fstate[0], fstate[1] = largest and second-largest drift (equal on ties; second is 0 for k = 1)."""
    dmax, d2nd = -np.inf, -np.inf
    for c in range(drift.shape[0]):
        if drift[c] > dmax:
            dmax, d2nd = drift[c], dmax
        elif drift[c] > d2nd:
            d2nd = drift[c]
    if drift.shape[0] == 1:
        d2nd = 0.0
    fstate[0], fstate[1] = dmax, d2nd


MOVED_RECOMPUTE = 4  # a chunk with more than 1/4 of its rows moved recomputes its sums; fewer are moved incrementally
SEP_STALE = 0.05  # phase B skips the separations while the accumulated drift is ≤ this × the mean half-separation
CENTER_ITEMS = 16  # centers per work item in the per-center phases of the Lloyd loop (4 and 64 are slower: ticket cost vs balance)
ROW_ITEMS = 4  # rows of the transposed centers per work item


# `_hamerly`'s istate slots; counters and flags on separate cache lines, so spinning threads don't slow the atomics
_TICKET, _DONE, _PHASE, _STOP = 0, 8, 16, 17  # next ticket, items done, published phase, stop flag
_ITER, _FINAL = 24, 25  # iterations so far, 1 once this is the last iteration
_KEEP_SEP = 26  # 1 if the next phase B keeps the drift-adjusted separations instead of recomputing them
_BALL, _CD_OK = 27, 28  # 1 if this pass hands its failing rows to the ball scans; 1 if cd holds the last full separations
_T0, _RUN, _PREV = 29, 30, 31  # clock at the start of this bound pass; passes in a row of this kind; previous kind


@njit
def _hamerly(x, xnorm, centers, bounds, max_iter, tol_abs, max_moved, labels, nthreads, unit, tiny, pfd, cd):
    """Exact Lloyd with Hamerly's bounds, the whole loop compiled; `centers` is updated in place. Returns (n_iter, inertia).

    All iterations run in one parallel region: each iteration is a stream of tickets over four phases
    (A: transposed centers, B: center norms and separations, C: the bounds pass over row chunks, D: new centers),
    and the thread that completes a phase's last item does the serial work at its end and publishes the next
    phase. Work items and reduction orders are those of the per-phase regions, so results don't change.
    For BALL_MIN_K ≤ k ≤ BALL_MAX_K three phases follow C: C2 sorts the rows that failed the bound test by
    label, E scans them per label against the centers within reach (`_ball_label`), C3 updates the chunk sums. They
    work only in passes chosen at the end of the previous one (many failing rows, few centers within reach); labels
    are exact either way, so results don't depend on that choice.
    """
    n, d = x.shape
    k = centers.shape[0]
    nb = bounds.shape[0] - 1
    upper, lower = np.empty(n), np.empty(n)
    sums, counts = np.empty((nb, k, d)), np.empty((nb, k), dtype=np.int64)
    moved, scans = np.empty(nb, dtype=np.int64), np.empty(nb, dtype=np.int64)
    tch = np.ones(k, dtype=np.uint8)  # centers whose sums changed in some chunk since the last phase D (all at first)
    dirty = np.zeros(nb, dtype=np.int64)  # chunks whose sums must be recomputed (rows taken out by the empty-center rule)
    new, tot = np.empty((k, d)), np.empty(k, dtype=np.int64)
    ct, cnorm = np.empty((d, k), dtype=x.dtype), np.empty(k, dtype=x.dtype)
    err_a, err_b = np.empty(k), np.empty(k)  # per-center rounding allowance of the scans' dot form: a + b·‖x‖
    ctb = np.zeros(((k + SEED_ROWS - 1) // SEED_ROWS, d, SEED_ROWS))  # float64 centers, block-transposed like seeding's rows: separations by FMA tiles
    half_sep, drift = np.empty(k), np.zeros(k)
    hs_base, acc = np.zeros(k), np.zeros(k)  # separations at the last full computation, drift since then
    rcnt = np.zeros(8 * nthreads, dtype=np.int64)  # phase C claims so far from each worker's chunk range
    istate = np.zeros(32, dtype=np.int64)  # slots: _TICKET ... _PREV above
    istate[_FINAL] = max_iter == 0
    fstate = np.empty(2)  # largest and second-largest drift
    _top2(drift, fstate)
    # ball scans: rows failing the bound test are scanned per label against the centers within 2u of theirs
    # cd: k×k center distances at the last full computation of the separations if ball scans may run, else 1×1
    ball = cd.shape[0] > 1
    kb, nbl = (k, n) if ball else (1, 1)
    crow = np.empty((kb, d), dtype=x.dtype)  # centers in x's dtype, row-major, for `_dot_tile_idx`
    bprobe = np.array([BALL_PROBE, -1], dtype=np.int64)  # passes between re-timings (doubles while they confirm); kind before a re-timing
    qbuf, qold, order = np.empty(nbl, dtype=np.int64), np.empty(nbl, dtype=np.int64), np.empty(nbl, dtype=np.int64)
    qn = np.zeros(nb, dtype=np.int64)  # rows recorded per chunk (at qbuf[bounds[b]:]) with their old labels (qold)
    lcnt = np.zeros((nb if ball else 1, kb), dtype=np.int64)  # failing rows per chunk and label, then scatter offsets
    lstart = np.zeros(kb + 1, dtype=np.int64)  # label a's rows: order[lstart[a]:lstart[a + 1]]
    ne = min(k, 4 * nthreads) if ball else 0  # label groups (phase E items)
    ngc = min(nb, 2 * nthreads)  # chunk groups (phase C2 and C3 items)
    gsplit = np.zeros(ne + 1, dtype=np.int64)
    pcost = np.zeros(3)  # time per failing row of the last plain [0] and ball [1] pass (0: not yet); [2]: failing
    # fraction of the last pass that followed a plain pass (ball scans leave weaker lower bounds: |c_o − c_a| − u)
    na, nc = (d + ROW_ITEMS - 1) // ROW_ITEMS, (k + CENTER_ITEMS - 1) // CENTER_ITEMS
    # phase kinds: 0 A transposed centers, 1 B norms and separations, 2 C bound pass, 3 D new centers,
    # 4 C2 scatter failing rows into label order, 5 E ball scans per label group, 6 C3 chunk sums of moved rows
    kinds = np.array([0, 1, 2, 4, 5, 6, 3] if ball else [0, 1, 2, 3], dtype=np.int64)
    sizes = np.array([na, nc, nb, ngc, ne, ngc, nc] if ball else [na, nc, nb, nc], dtype=np.int64)
    nph = kinds.shape[0]
    offs = np.zeros(nph + 1, dtype=np.int64)  # first item of each phase; offs[nph] = items per iteration
    for p in range(nph):
        offs[p + 1] = offs[p] + sizes[p]
    per_it = offs[nph]
    pass_end = 6 if ball else 2  # the bound pass (incl. its scans and sums) is complete after this phase kind
    maxrows = 0
    for b in range(nb):
        maxrows = max(maxrows, bounds[b + 1] - bounds[b])
    for _w in numba.prange(nthreads):
        queue = np.empty(maxrows, dtype=np.int64)  # full scans go into a queue, then run SEED_ROWS rows at a time
        oldl = np.empty(maxrows, dtype=np.int64)  # the queued rows' labels before the scan
        dvs = np.empty((k, SEED_ROWS))  # squared distances of an item's centers to one block of centers
        lqc, sepmin = np.empty(CENTER_ITEMS, dtype=np.int64), np.empty(CENTER_ITEMS)
        while True:  # tickets in order: (iteration, phase, item); nobody waits for a particular thread, so it can't deadlock
            ticket = _fetch_add(istate, _TICKET)
            tit, r = ticket // per_it, ticket % per_it
            pp = 0
            while r >= offs[pp + 1]:
                pp += 1
            ph = kinds[pp]
            item = r - offs[pp]
            g = nph * tit + pp
            stop, spins = False, 0
            while True:  # wait until phase g may run
                if _load_acquire(istate, _STOP):
                    stop = True
                    break
                if _load_acquire(istate, _PHASE) >= g:
                    break
                spins += 1
                if spins > SPIN_BEFORE_YIELD:
                    _yield_cpu()
            if stop:
                break
            if ph == 0:
                for j in range(item * ROW_ITEMS, min(d, (item + 1) * ROW_ITEMS)):  # whole rows per item: no false sharing
                    for c in range(k):
                        ct[j, c] = centers[c, j]
                        ctb[c // SEED_ROWS, j, c % SEED_ROWS] = centers[c, j]
                        if ball:
                            if istate[_BALL]:
                                crow[c, j] = centers[c, j]
            elif ph == 1:  # norms, and separations: the item's centers against blocks of SEED_ROWS centers by FMA tiles (float64)
                c_lo = item * CENTER_ITEMS
                nl = min(k, c_lo + CENTER_ITEMS) - c_lo
                for t in range(nl):
                    c = c_lo + t
                    lqc[t] = c
                    sepmin[t] = np.inf
                    s = 0.0
                    for j in range(d):
                        s += centers[c, j] * centers[c, j]
                    cnorm[c] = s
                    err_a[c], err_b[c] = _dot_allowance(s, unit, tiny, d)
                if istate[_KEEP_SEP]:  # separations stay the drift-adjusted bounds set after phase D
                    nl = 0
                for bo in range(ctb.shape[0] if nl else 0):
                    for t in range(nl):
                        dvs[c_lo + t, :] = 0.0
                    _score(ctb, bo, centers, lqc, nl, dvs, 0, d)
                    for t in range(nl):
                        c = c_lo + t
                        for r in range(min(SEED_ROWS, k - bo * SEED_ROWS)):
                            if bo * SEED_ROWS + r != c:
                                sepmin[t] = min(sepmin[t], dvs[c, r])
                            if ball:
                                if istate[_BALL]:
                                    cd[c, bo * SEED_ROWS + r] = math.sqrt(dvs[c, r])
                for t in range(nl):
                    half_sep[c_lo + t] = 0.5 * math.sqrt(sepmin[t])
                    hs_base[c_lo + t] = half_sep[c_lo + t]
                    acc[c_lo + t] = 0.0
            elif ph == 2:  # one chunk per ticket, from the own range first: x rows and bounds stay in this core's caches
                v = _w % nthreads  # an int64 copy of the prange index: v is reassigned below, and (uint64 + 1) % int64 types as float64
                b = _claim_range(rcnt, v, tit, nb, nthreads)
                while b < 0:
                    v = (v + 1) % nthreads
                    b = _claim_range(rcnt, v, tit, nb, nthreads)
                dmax, d2nd = fstate[0], fstate[1]
                mv, sc = 0, 0
                lo, hi = bounds[b], bounds[b + 1]
                nq, nrec = 0, 0
                bl = 0
                if ball:
                    bl = istate[_BALL]
                if bl:
                    lcnt[b, :] = 0
                for i in range(lo, hi):
                    a = labels[i]
                    need = a < 0
                    if not need:
                        u = upper[i] + drift[a]
                        l = lower[i] - (d2nd if drift[a] == dmax else dmax)
                        m = max(half_sep[a], l)
                        if u * (1 + SAFETY) > m * (1 - SAFETY):
                            if k <= SMALL_K:  # small k: the queued scan is cheap and computes the exact distance anyway (most tightened rows got scanned)
                                need = True
                            else:
                                s = 0.0
                                for j in range(d):
                                    t = np.float64(x[i, j]) - centers[a, j]
                                    s += t * t
                                u = math.sqrt(s)
                                need = u * (1 + SAFETY) > m * (1 - SAFETY)
                        upper[i] = u
                        lower[i] = l
                    if need:
                        sc += 1
                        if bl and a >= 0:  # to the ball scans (phase E); its chunk sums follow in phase C3
                            qbuf[lo + nrec] = i
                            qold[lo + nrec] = a
                            nrec += 1
                            lcnt[b, a] += 1
                        else:
                            queue[nq] = i
                            nq += 1
                if nq:  # FMA dot tiles for every k: scans are most of the pass on real data
                    for t in range(nq):
                        oldl[t] = labels[queue[t]]
                    mv += _scan_queue_dot(x, xnorm, ct, cnorm, err_a, err_b, centers, queue, nq, labels, upper, lower, pfd)
                if bl:  # rows scanned here (no label yet) join the recorded rows; phase C3 updates the sums of all of them
                    for t in range(nq):
                        qbuf[lo + nrec] = queue[t]
                        qold[lo + nrec] = oldl[t]
                        nrec += 1
                    qn[b] = nrec
                    scans[b] = sc
                    mv = 0
                    nq = 0
                if mv:
                    _chunk_sums(x, b, lo, hi, queue, oldl, 0, nq, mv, labels, sums, counts, tch, dirty)
                if not bl:
                    moved[b] = mv
                    scans[b] = sc
            elif ph == 4:  # scatter the recorded rows of chunk group `item` into label order
                if ball and istate[_BALL]:
                    _ball_scatter(bounds, item * nb // ngc, (item + 1) * nb // ngc, qn, qbuf, qold, lcnt, order)
            elif ph == 5:  # ball scans of label group `item`
                if ball and istate[_BALL]:
                    _ball_group(x, xnorm, crow, centers, cnorm, err_a, err_b, cd, acc, gsplit[item], gsplit[item + 1], lstart, order,
                                labels, upper, lower)
            elif ph == 6:  # chunk sums for the rows that moved in phases C and E
                if ball and istate[_BALL]:
                    _ball_sums(x, bounds, item * nb // ngc, (item + 1) * nb // ngc, qn, qbuf, qold, labels, sums, counts, tch, dirty, moved)
            else:
                for c in range(item * CENTER_ITEMS, min(k, (item + 1) * CENTER_ITEMS)):
                    if not tch[c]:  # no chunk's sums for c changed: new[c] (= centers[c]) and tot[c] stand
                        continue
                    cnt = 0
                    for bb in range(nb):
                        cnt += counts[bb, c]
                    tot[c] = cnt
                    if cnt == 0:
                        for j in range(d):
                            new[c, j] = centers[c, j]
                    else:
                        for j in range(d):
                            s = 0.0
                            for bb in range(nb):
                                s += sums[bb, c, j]
                            new[c, j] = s / cnt
            if _fetch_add(istate, _DONE) != tit * per_it + offs[pp + 1] - 1:
                continue
            # the last item of phase g: its serial part, then publish phase g + 1
            if ball and ph == 1:
                istate[_T0] = _clock_ticks()
            if ball and ph == 6 and istate[_ITER] >= 1:  # a pass with labels: its time per failing row
                n_scan = 0
                for bb in range(nb):
                    n_scan += scans[bb]
                if n_scan:
                    pcost[istate[_BALL]] = (_clock_ticks() - istate[_T0]) / n_scan
            if ph == pass_end and istate[_FINAL]:
                _store_release(istate, _STOP, 1)  # the pass after the last update is done
                break
            if ball and ph == 2 and istate[_BALL]:  # ball scans: label offsets and label groups
                _ball_offsets(lcnt, lstart, gsplit)
            if ph == 3:
                istate[_ITER] += 1
                n_moved, n_scan = 0, 0
                for bb in range(nb):
                    n_moved += moved[bb]
                    n_scan += scans[bb]
                if ball:  # ball scans next pass? Only if the bound test failed for many rows in a pass with
                    # labels (the first pass scans every row), and by the measured cost per failing row of both kinds
                    if istate[_PREV] == 0:  # this pass's failures come from plain-pass bounds
                        pcost[2] = n_scan / n
                    istate[_PREV] = istate[_BALL]
                    nxt = 0
                    if istate[_ITER] >= 2 and pcost[2] >= BALL_FRACTION and n_scan >= BALL_MIN_ROWS * k:
                        nxt = 1 if pcost[1] == 0.0 else int(pcost[1] < pcost[0])
                        if bprobe[1] >= 0:  # the last pass re-timed the other kind: wait twice as long if it confirmed
                            bprobe[0] = 2 * bprobe[0] if nxt == bprobe[1] else BALL_PROBE
                            bprobe[1] = -1
                        if istate[_RUN] >= bprobe[0]:  # re-time the other kind now and then: the costs drift as Lloyd converges
                            bprobe[1] = nxt
                            nxt = 1 - nxt
                    istate[_RUN] = istate[_RUN] + 1 if nxt == istate[_BALL] else 1
                    istate[_BALL] = nxt
                n_empty = 0
                for c in range(k):
                    n_empty += tot[c] == 0
                if n_empty:  # as sklearn: empty centers move to the rows farthest from theirs (ties: lowest row); rare, so serial
                    dist = np.empty(n)  # exact squared distance to the assigned center, as the reference ranks them
                    for i in range(n):
                        s = 0.0
                        for j in range(d):
                            t = np.float64(x[i, j]) - centers[labels[i], j]
                            s += t * t
                        dist[i] = s
                    far = np.argsort(-dist, kind="mergesort")
                    e = 0
                    for c in range(k):
                        if tot[c] == 0:
                            for j in range(d):
                                new[c, j] = x[far[e], j]
                            if labels[far[e]] >= 0:
                                tch[labels[far[e]]] = 1  # the row leaves its center
                            labels[far[e]] = -1  # rescan next time
                            for bb in range(nb):  # its chunk's sums get recomputed (the row leaves its old center)
                                if bounds[bb] <= far[e] < bounds[bb + 1]:
                                    dirty[bb] = 1
                            e += 1
                shift = 0.0
                for c in range(k):
                    s = 0.0
                    for j in range(d):
                        t = new[c, j] - centers[c, j]
                        s += t * t
                    drift[c] = math.sqrt(s)
                    shift += drift[c] ** 2
                centers[:] = new
                tch[:] = 0
                istate[_FINAL] = n_moved <= max_moved or shift <= tol_abs or istate[_ITER] == max_iter
                _top2(drift, fstate)
                # separations: keep valid bounds hs_base[c] − (acc[c] + max acc)/2 (triangle inequality) while the drift
                # since the last full computation is small next to them; recompute otherwise
                amax, hmean = 0.0, 0.0
                for c in range(k):
                    acc[c] += drift[c]
                    amax = max(amax, acc[c])
                    hmean += hs_base[c]
                istate[_KEEP_SEP] = amax <= SEP_STALE * hmean / k
                if ball and istate[_BALL] and not istate[_CD_OK]:  # the ball scans need cd from the last full computation of the separations
                    istate[_KEEP_SEP] = 0
                if not istate[_KEEP_SEP]:
                    istate[_CD_OK] = istate[_BALL]  # phase B recomputes; it stores cd only for a ball pass
                if istate[_KEEP_SEP]:
                    for c in range(k):
                        half_sep[c] = max(hs_base[c] - 0.5 * (acc[c] + amax), 0.0)
            _store_release(istate, _PHASE, g + 1)
    it = istate[_ITER]
    part = np.empty(nb)
    for b in numba.prange(nb):  # exact inertia, rows in order per chunk, chunks added in order
        s = 0.0
        for i in range(bounds[b], bounds[b + 1]):
            for j in range(d):
                t = np.float64(x[i, j]) - centers[labels[i], j]
                s += t * t
        part[b] = s
    inertia = 0.0
    for b in range(nb):
        inertia += part[b]
    return it, inertia


NORM_CHUNK = 2048  # rows per chunk of the norms/variance pass: small enough to stay in L2 for the chunk's second pass


@njit
def _norms_var(x):
    """Row norms ‖x_i‖² and the mean column variance in one pass over x.

    Each fixed chunk takes its column means, then its squared deviations while it is still in cache; chunks are
    combined in chunk order with Chan's parallel formula, so the result doesn't depend on threads. Replaces np.einsum
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
            csum[b, j] /= hi - lo  # now the chunk's column means
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
