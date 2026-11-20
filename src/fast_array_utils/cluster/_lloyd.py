# SPDX-License-Identifier: MPL-2.0
"""Exact Lloyd with Hamerly’s bounds in one persistent parallel region (`_hamerly`)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numba
import numpy as np

from fast_array_utils.numba import njit

from ._ball import BALL_FRACTION, BALL_MIN_ROWS, BALL_PROBE, _ball_group, _ball_offsets, _ball_scatter, _ball_sums, _chunk_sums
from ._sync import LINE_SLOTS, SPIN_BEFORE_YIELD, _claim_range, _clock_ticks, _fetch_add, _load_acquire, _store_release, _yield_cpu
from ._tiles import SAFETY, SEED_ROWS, _dot_allowance, _scan_queue_dot, _score


if TYPE_CHECKING:
    from ._typing import F64, I64, U8, FArr


# Kernels take their arrays as positional arguments (one PLR0917 suppression per kernel, FBT001 for flags):
# numba can’t inline calls with keyword-only arguments.

BALL_FROM_ITER = 2  # ball scans from the second pass with labels on (the first pass scans every row)
SMALL_K = 64  # up to this many centers, rows failing the bound test go straight to the queued scan, without tightening
SUM_BUDGET = 1 << 22  # max float64 entries of per-chunk center sums (32 MB)
SEP_STALE = 0.05  # phase B skips the separations while the accumulated drift is ≤ this × the mean half-separation
CENTER_ITEMS = 16  # centers per work item in the per-center phases (4 and 64 are slower: ticket cost vs balance)
ROW_ITEMS = 4  # rows of the transposed centers per work item
LABEL_GROUPS_PER_THREAD, CHUNK_GROUPS_PER_THREAD = 4, 2  # work items per thread of phase E, and of phases C2 and C3

# phase kinds, in the order of one iteration; C2, E and C3 only with ball scans
PH_TRANSPOSE, PH_SEPARATIONS, PH_BOUNDS, PH_UPDATE = 0, 1, 2, 3  # A, B, C, D
PH_SCATTER, PH_BALL_SCAN, PH_BALL_SUMS = 4, 5, 6  # C2: failing rows into label order, E: ball scans, C3: their chunk sums

# `_hamerly`’s istate slots; counters and flags on separate cache lines, so spinning threads don’t slow the atomics
_TICKET, _DONE, _PHASE, _STOP = 0, 8, 16, 17  # next ticket, items done, published phase, stop flag
_ITER, _FINAL = 24, 25  # iterations so far, 1 once this is the last iteration
_KEEP_SEP = 26  # 1 if the next phase B keeps the drift-adjusted separations instead of recomputing them
_BALL, _CD_OK = 27, 28  # 1 if this pass hands its failing rows to the ball scans; 1 if cd holds the last full separations
_T0, _RUN, _PREV = 29, 30, 31  # clock at the start of this bound pass; passes in a row of this kind; previous kind
ISTATE_SLOTS = 32


def _hamerly_bounds(n: int, k: int, d: int) -> np.ndarray:
    """Split the rows into fixed chunks from the shape alone, so results don’t depend on threads.

    Chunks have ≥ 256 rows each, there are ≤ 256 of them, and their sums fit in SUM_BUDGET.
    """
    nb = max(1, min(-(-n // 256), 256, SUM_BUDGET // (k * d)))
    return (np.arange(nb + 1, dtype=np.int64) * n) // nb


@njit
def _hamerly(  # noqa: PLR0917
    x: FArr,
    xnorm: F64,
    centers: F64,
    bounds: I64,
    max_iter: int,
    tol_abs: float,
    max_moved: int,
    labels: I64,
    nthreads: int,
    unit: float,
    tiny: float,
    pfd: int,
    cd: F64,
) -> int:
    """Exact Lloyd with Hamerly’s bounds, the whole loop compiled; `centers` and `labels` are updated in place. Returns n_iter.

    All iterations run in one parallel region: each iteration is a stream of tickets over its phases
    (A: transposed centers, B: center norms and separations, C: the bounds pass over row chunks, D: new centers),
    and the thread that completes a phase’s last item does the serial work at its end and publishes the next phase.
    Work items and reduction orders are those of per-phase regions, so results don’t change.
    For BALL_MIN_K ≤ k ≤ BALL_MAX_K three phases follow C: C2 sorts the rows that failed the bound test by label,
    E scans them per label against the centers within reach (`_ball_label`), C3 updates the chunk sums.
    They work only in passes chosen at the end of the previous one (many failing rows, few centers within reach);
    labels are exact either way, so results don’t depend on that choice.
    """
    # prepare memory
    n, d = x.shape
    k = centers.shape[0]
    nb = bounds.shape[0] - 1
    upper, lower, sums, counts, moved, scans, tch, dirty, new, tot = _row_and_sum_state(n, nb, k, d)
    ct, cnorm, err_a, err_b, ctb, half_sep, drift, hs_base, acc, fstate = _center_state(x, k)
    rcnt = np.zeros(LINE_SLOTS * nthreads, dtype=np.int64)  # phase C claims so far from each worker’s chunk range
    istate = np.zeros(ISTATE_SLOTS, dtype=np.int64)
    istate[_FINAL] = max_iter == 0
    # ball scans: rows failing the bound test are scanned per label against the centers within 2u of theirs
    # cd: k×k center distances at the last full computation of the separations if ball scans may run, else 1×1
    ball = cd.shape[0] > 1
    crow, qbuf, qold, order, qn, lcnt, lstart, gsplit, bprobe, pcost = _ball_state(x, k, nb, nthreads, ball)
    ne, ngc = gsplit.shape[0] - 1, min(nb, CHUNK_GROUPS_PER_THREAD * nthreads)  # label groups (phase E items), chunk groups (phase C2 and C3 items)
    kinds, offs = _phase_plan(ball, d, k, nb, ngc, ne)
    nph, per_it = kinds.shape[0], offs[-1]
    maxrows = np.max(bounds[1:] - bounds[:-1])

    # run the update step until converged, all iterations in one region instead of one per phase (4 to 7 per iteration),
    # whose wake-ups dominated small passes (1.3× faster on 20 000 rows, k ≤ 40); tickets in order (iteration, phase, item),
    # nobody waits for a particular thread, so it can’t deadlock
    for _w in numba.prange(nthreads):
        queue = np.empty(maxrows, dtype=np.int64)  # full scans go into a queue, then run SEED_ROWS rows at a time
        oldl = np.empty(maxrows, dtype=np.int64)  # the queued rows’ labels before the scan
        dvs = np.empty((k, SEED_ROWS))  # squared distances of an item’s centers to one block of centers
        lqc, sepmin = np.empty(CENTER_ITEMS, dtype=np.int64), np.empty(CENTER_ITEMS)
        while True:
            tit, pp, item = _ticket(istate, per_it, offs)
            g = nph * tit + pp
            if _wait_phase(istate, g):
                break
            ph = kinds[pp]
            with_ball = istate[_BALL] != 0  # set only if ball scans may run (`_choose_pass_kind`)
            if ph == PH_TRANSPOSE:
                _transpose_centers(item, centers, ct, ctb, crow, with_ball)
            elif ph == PH_SEPARATIONS:
                _separations(item, centers, cnorm, err_a, err_b, unit, tiny, istate[_KEEP_SEP], ctb, lqc, sepmin, dvs, cd, with_ball, half_sep, hs_base, acc)
            elif ph == PH_BOUNDS:
                b = _claim_chunk(rcnt, _w, tit, nb, nthreads)
                _bound_pass(
                    x,
                    xnorm,
                    centers,
                    ct,
                    cnorm,
                    err_a,
                    err_b,
                    labels,
                    upper,
                    lower,
                    half_sep,
                    drift,
                    fstate,
                    bounds[b],
                    bounds[b + 1],
                    b,
                    with_ball,
                    queue,
                    oldl,
                    qbuf,
                    qold,
                    qn,
                    lcnt,
                    moved,
                    scans,
                    sums,
                    counts,
                    tch,
                    dirty,
                    pfd,
                )
            elif ph != PH_UPDATE:
                _ball_phase(
                    with_ball,
                    ph,
                    item,
                    x,
                    xnorm,
                    crow,
                    centers,
                    cnorm,
                    err_a,
                    err_b,
                    cd,
                    acc,
                    bounds,
                    ngc,
                    gsplit,
                    lstart,
                    order,
                    qn,
                    qbuf,
                    qold,
                    lcnt,
                    labels,
                    upper,
                    lower,
                    sums,
                    counts,
                    tch,
                    dirty,
                    moved,
                )
            else:
                _new_centers(item, centers, sums, counts, tch, new, tot)
            if _fetch_add(istate, _DONE) != tit * per_it + offs[pp + 1] - 1:
                continue
            # the last item of phase g: its serial part, then publish phase g + 1
            if _end_of_phase(
                ph,
                with_ball,
                x,
                centers,
                labels,
                bounds,
                new,
                tot,
                moved,
                scans,
                tch,
                dirty,
                drift,
                fstate,
                half_sep,
                hs_base,
                acc,
                istate,
                pcost,
                bprobe,
                lcnt,
                lstart,
                gsplit,
                ball,
                max_iter,
                max_moved,
                tol_abs,
            ):
                break
            _store_release(istate, _PHASE, g + 1)
    n_iter: int = istate[_ITER]
    return n_iter


@njit
def _inertia(x: FArr, centers: F64, labels: I64, bounds: I64) -> float:
    """Exact inertia: rows in order per chunk, chunks added in order."""
    d = x.shape[1]
    nb = bounds.shape[0] - 1
    part = np.empty(nb)
    for b in numba.prange(nb):
        s = 0.0
        for i in range(bounds[b], bounds[b + 1]):
            for j in range(d):
                t = np.float64(x[i, j]) - centers[labels[i], j]
                s += t * t
        part[b] = s
    inertia = 0.0
    for b in range(nb):
        inertia += part[b]
    return inertia


# ----------------------------------------------------------------------------- memory
@numba.njit(inline="always", nogil=True, cache=True)
def _row_and_sum_state(n: int, nb: int, k: int, d: int) -> tuple[F64, F64, F64, I64, I64, I64, U8, I64, F64, I64]:
    """Per-row bounds, per-chunk center sums and the bookkeeping of which sums changed."""
    upper, lower = np.empty(n), np.empty(n)
    sums, counts = np.empty((nb, k, d)), np.empty((nb, k), dtype=np.int64)
    moved, scans = np.empty(nb, dtype=np.int64), np.empty(nb, dtype=np.int64)
    tch = np.ones(k, dtype=np.uint8)  # centers whose sums changed in some chunk since the last phase D (all at first)
    dirty = np.zeros(nb, dtype=np.int64)  # chunks whose sums must be recomputed (rows taken out by the empty-center rule)
    new, tot = np.empty((k, d)), np.empty(k, dtype=np.int64)
    return upper, lower, sums, counts, moved, scans, tch, dirty, new, tot


@numba.njit(inline="always", nogil=True, cache=True)
def _center_state(x: FArr, k: int) -> tuple[FArr, FArr, F64, F64, F64, F64, F64, F64, F64, F64]:
    """Per-center copies, norms, rounding allowances, separations and drift."""
    d = x.shape[1]
    ct, cnorm = np.empty((d, k), dtype=x.dtype), np.empty(k, dtype=x.dtype)
    err_a, err_b = np.empty(k), np.empty(k)  # per-center rounding allowance of the scans’ dot form: a + b·‖x‖
    ctb = np.zeros(((k + SEED_ROWS - 1) // SEED_ROWS, d, SEED_ROWS))  # float64 centers, block-transposed like seeding’s rows
    half_sep, drift = np.empty(k), np.zeros(k)
    hs_base, acc = np.zeros(k), np.zeros(k)  # separations at the last full computation, drift since then
    fstate = np.empty(2)  # largest and second-largest drift
    _top2(drift, fstate)
    return ct, cnorm, err_a, err_b, ctb, half_sep, drift, hs_base, acc, fstate


@numba.njit(inline="always", nogil=True, cache=True)
def _ball_state(x: FArr, k: int, nb: int, nthreads: int, ball: bool) -> tuple[FArr, I64, I64, I64, I64, I64, I64, I64, I64, F64]:  # noqa: FBT001
    """Buffers of the ball scans (size 1 without them) and the bookkeeping of their pass-kind choice."""
    n, d = x.shape
    kb, nbl = (k, n) if ball else (1, 1)
    crow = np.empty((kb, d), dtype=x.dtype)  # centers in x’s dtype, row-major, for `_dot_tile_idx`
    qbuf, qold, order = np.empty(nbl, dtype=np.int64), np.empty(nbl, dtype=np.int64), np.empty(nbl, dtype=np.int64)
    qn = np.zeros(nb, dtype=np.int64)  # rows recorded per chunk (at qbuf[bounds[b]:]) with their old labels (qold)
    lcnt = np.zeros((nb if ball else 1, kb), dtype=np.int64)  # failing rows per chunk and label, then scatter offsets
    lstart = np.zeros(kb + 1, dtype=np.int64)  # label a’s rows: order[lstart[a]:lstart[a + 1]]
    gsplit = np.zeros((min(k, LABEL_GROUPS_PER_THREAD * nthreads) if ball else 0) + 1, dtype=np.int64)  # label groups of phase E: labels gsplit[e]...
    bprobe = np.array([BALL_PROBE, -1], dtype=np.int64)  # passes between re-timings (doubles while confirmed); kind before a re-timing
    pcost = np.zeros(3)  # time per failing row of the last plain [0] and ball [1] pass (0: not yet); [2]: failing
    # fraction of the last pass that followed a plain pass (ball scans leave weaker lower bounds: |c_o - c_a| - u)
    return crow, qbuf, qold, order, qn, lcnt, lstart, gsplit, bprobe, pcost


# ----------------------------------------------------------------------------- scheduling
@numba.njit(inline="always", nogil=True, cache=True)
def _phase_plan(ball: bool, d: int, k: int, nb: int, ngc: int, ne: int) -> tuple[I64, I64]:  # noqa: FBT001, PLR0917
    """Phase kinds of one iteration and the first item of each (offs[-1] = items per iteration)."""
    na, nc = (d + ROW_ITEMS - 1) // ROW_ITEMS, (k + CENTER_ITEMS - 1) // CENTER_ITEMS
    if ball:
        kinds = np.array([PH_TRANSPOSE, PH_SEPARATIONS, PH_BOUNDS, PH_SCATTER, PH_BALL_SCAN, PH_BALL_SUMS, PH_UPDATE], dtype=np.int64)
        sizes = np.array([na, nc, nb, ngc, ne, ngc, nc], dtype=np.int64)
    else:
        kinds = np.array([PH_TRANSPOSE, PH_SEPARATIONS, PH_BOUNDS, PH_UPDATE], dtype=np.int64)
        sizes = np.array([na, nc, nb, nc], dtype=np.int64)
    offs = np.zeros(kinds.shape[0] + 1, dtype=np.int64)
    for p in range(kinds.shape[0]):
        offs[p + 1] = offs[p] + sizes[p]
    return kinds, offs


@numba.njit(inline="always", nogil=True, cache=True)
def _ticket(istate: I64, per_it: int, offs: I64) -> tuple[int, int, int]:
    """Draw the next ticket: (iteration, phase index, item within the phase)."""
    ticket = _fetch_add(istate, _TICKET)
    tit, r = ticket // per_it, ticket % per_it
    pp = 0
    while r >= offs[pp + 1]:
        pp += 1
    return tit, pp, r - offs[pp]


@numba.njit(inline="always", nogil=True, cache=True)
def _wait_phase(istate: I64, g: int) -> bool:
    """Wait until phase g may run; True if the loop has stopped instead.

    Waiting workers spin, then yield: a pure spin starved the working threads when workers and CPUs are as many
    (numba’s workqueue layer: 18 → 66 ms on 20 000 rows, 8 threads).
    """
    spins = 0
    while True:
        if _load_acquire(istate, _STOP):
            return True
        if _load_acquire(istate, _PHASE) >= g:
            return False
        spins += 1
        if spins > SPIN_BEFORE_YIELD:
            _yield_cpu()


@numba.njit(inline="always", nogil=True, cache=True)
def _claim_chunk(rcnt: I64, w: int, tit: int, nb: int, nthreads: int) -> int:
    """Claim a row chunk of iteration tit’s bound pass, from worker w’s own range first (its rows stay in this core’s caches)."""
    v = w % nthreads  # an int64 copy of the prange index: v is reassigned below, and (uint64 + 1) % int64 types as float64
    b = _claim_range(rcnt, v, tit, nb, nthreads)
    while b < 0:
        v = (v + 1) % nthreads
        b = _claim_range(rcnt, v, tit, nb, nthreads)
    return b


# ----------------------------------------------------------------------------- phases
@numba.njit(inline="always", nogil=True, cache=True)
def _transpose_centers(item: int, centers: F64, ct: FArr, ctb: F64, crow: FArr, with_crow: bool) -> None:  # noqa: FBT001, PLR0917
    """Phase A: columns item·ROW_ITEMS... of the centers into ct, ctb (and crow for the ball scans); whole rows per item: no false sharing."""
    d, k = ct.shape
    for j in range(item * ROW_ITEMS, min(d, (item + 1) * ROW_ITEMS)):
        for c in range(k):
            ct[j, c] = centers[c, j]
            ctb[c // SEED_ROWS, j, c % SEED_ROWS] = centers[c, j]
            if with_crow:
                crow[c, j] = centers[c, j]


@numba.njit(inline="always", nogil=True, cache=True)
def _separations(  # noqa: PLR0917
    item: int,
    centers: F64,
    cnorm: FArr,
    err_a: F64,
    err_b: F64,
    unit: float,
    tiny: float,
    keep_sep: int,
    ctb: F64,
    lqc: I64,
    sepmin: F64,
    dvs: F64,
    cd: F64,
    with_cd: bool,  # noqa: FBT001
    half_sep: F64,
    hs_base: F64,
    acc: F64,
) -> None:
    """Phase B: norms and rounding allowances of the item’s centers, and their half-separations.

    Separations: the item’s centers against blocks of SEED_ROWS centers by FMA tiles (float64);
    skipped while they stay the drift-adjusted bounds set after phase D (keep_sep).
    """
    k, d = centers.shape
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
    if keep_sep:
        return
    for bo in range(ctb.shape[0]):
        _separation_block(bo, c_lo, nl, centers, ctb, lqc, sepmin, dvs, cd, with_cd)
    for t in range(nl):
        half_sep[c_lo + t] = 0.5 * math.sqrt(sepmin[t])
        hs_base[c_lo + t] = half_sep[c_lo + t]
        acc[c_lo + t] = 0.0


@numba.njit(inline="always", nogil=True, cache=True)
def _separation_block(bo: int, c_lo: int, nl: int, centers: F64, ctb: F64, lqc: I64, sepmin: F64, dvs: F64, cd: F64, with_cd: bool) -> None:  # noqa: FBT001, PLR0917
    """Squared distances of centers c_lo..c_lo+nl to block bo of SEED_ROWS centers; their minimum over the others (and cd)."""
    k, d = centers.shape
    for t in range(nl):
        dvs[c_lo + t, :] = 0.0
    _score(ctb, bo, centers, lqc, nl, dvs, 0, d)
    for t in range(nl):
        c = c_lo + t
        for r in range(min(SEED_ROWS, k - bo * SEED_ROWS)):
            if bo * SEED_ROWS + r != c:
                sepmin[t] = min(sepmin[t], dvs[c, r])
            if with_cd:
                cd[c, bo * SEED_ROWS + r] = math.sqrt(dvs[c, r])


@numba.njit(inline="always", nogil=True, cache=True)
def _bound_pass(  # noqa: PLR0917
    x: FArr,
    xnorm: F64,
    centers: F64,
    ct: FArr,
    cnorm: FArr,
    err_a: F64,
    err_b: F64,
    labels: I64,
    upper: F64,
    lower: F64,
    half_sep: F64,
    drift: F64,
    fstate: F64,
    lo: int,
    hi: int,
    b: int,
    with_ball: bool,  # noqa: FBT001
    queue: I64,
    oldl: I64,
    qbuf: I64,
    qold: I64,
    qn: I64,
    lcnt: I64,
    moved: I64,
    scans: I64,
    sums: F64,
    counts: I64,
    tch: U8,
    dirty: I64,
    pfd: int,
) -> None:
    """Phase C on rows lo..hi (chunk b): Hamerly’s bound test, then exact scans of the rows that fail it.

    Without ball scans, the failing rows are scanned here (`_scan_queue_dot`) and the chunk sums updated;
    with them, rows that have a label are recorded for phases C2, E and C3.
    """
    dmax, d2nd = fstate[0], fstate[1]
    mv, sc = 0, 0
    nq, nrec = 0, 0
    if with_ball:
        lcnt[b, :] = 0
    for i in range(lo, hi):
        a = labels[i]
        need = a < 0 or _fails_bound_test(x, i, a, centers, upper, lower, half_sep, drift, dmax, d2nd)
        if need:
            sc += 1
            if with_ball and a >= 0:  # to the ball scans (phase E); its chunk sums follow in phase C3
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
    if with_ball:  # rows scanned here (no label yet) join the recorded rows; phase C3 updates the sums of all of them
        for t in range(nq):
            qbuf[lo + nrec] = queue[t]
            qold[lo + nrec] = oldl[t]
            nrec += 1
        qn[b] = nrec
        scans[b] = sc
        return
    if mv:
        _chunk_sums(x, b, lo, hi, queue, oldl, 0, nq, mv, labels, sums, counts, tch, dirty)
    moved[b] = mv
    scans[b] = sc


@numba.njit(inline="always", nogil=True, cache=True)
def _fails_bound_test(x: FArr, i: int, a: int, centers: F64, upper: F64, lower: F64, half_sep: F64, drift: F64, dmax: float, d2nd: float) -> bool:  # noqa: PLR0917
    """Hamerly’s test for labelled row i, with drift-updated bounds (stored back); True if it needs a full scan."""
    k, d = centers.shape
    u = upper[i] + drift[a]
    lb = lower[i] - (d2nd if drift[a] == dmax else dmax)
    m = max(half_sep[a], lb)
    need = False
    if u * (1 + SAFETY) > m * (1 - SAFETY):
        if k <= SMALL_K:  # small k: the queued scan is cheap and computes the exact distance anyway
            need = True
        else:
            s = 0.0
            for j in range(d):
                t = np.float64(x[i, j]) - centers[a, j]
                s += t * t
            u = math.sqrt(s)
            need = u * (1 + SAFETY) > m * (1 - SAFETY)
    upper[i] = u
    lower[i] = lb
    return need


@numba.njit(inline="always", nogil=True, cache=True)
def _new_centers(item: int, centers: F64, sums: F64, counts: I64, tch: U8, new: F64, tot: I64) -> None:  # noqa: PLR0917
    """Phase D: new means of the item’s centers from the chunk sums (centers whose sums didn’t change keep theirs)."""
    k, d = centers.shape
    nb = sums.shape[0]
    for c in range(item * CENTER_ITEMS, min(k, (item + 1) * CENTER_ITEMS)):
        if not tch[c]:  # no chunk’s sums for c changed: new[c] (= centers[c]) and tot[c] stand
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


@numba.njit(inline="always", nogil=True, cache=True)
def _ball_phase(  # noqa: PLR0917
    with_ball: bool,  # noqa: FBT001
    ph: int,
    item: int,
    x: FArr,
    xnorm: F64,
    crow: FArr,
    centers: F64,
    cnorm: FArr,
    err_a: F64,
    err_b: F64,
    cd: F64,
    acc: F64,
    bounds: I64,
    ngc: int,
    gsplit: I64,
    lstart: I64,
    order: I64,
    qn: I64,
    qbuf: I64,
    qold: I64,
    lcnt: I64,
    labels: I64,
    upper: F64,
    lower: F64,
    sums: F64,
    counts: I64,
    tch: U8,
    dirty: I64,
    moved: I64,
) -> None:
    """Item `item` of ball phase ph: C2 scatter (chunk group), E ball scans (label group) or C3 chunk sums (chunk group).

    Nothing to do in passes without ball scans.
    """
    nb = bounds.shape[0] - 1
    if not with_ball:
        return
    if ph == PH_SCATTER:
        _ball_scatter(bounds, item * nb // ngc, (item + 1) * nb // ngc, qn, qbuf, qold, lcnt, order)
    elif ph == PH_BALL_SCAN:
        _ball_group(x, xnorm, crow, centers, cnorm, err_a, err_b, cd, acc, gsplit[item], gsplit[item + 1], lstart, order, labels, upper, lower)
    else:
        _ball_sums(x, bounds, item * nb // ngc, (item + 1) * nb // ngc, qn, qbuf, qold, labels, sums, counts, tch, dirty, moved)


# ----------------------------------------------------------------------------- serial work at the end of a phase
@numba.njit(inline="always", nogil=True, cache=True)
def _end_of_phase(  # noqa: PLR0917
    ph: int,
    with_ball: bool,  # noqa: FBT001
    x: FArr,
    centers: F64,
    labels: I64,
    bounds: I64,
    new: F64,
    tot: I64,
    moved: I64,
    scans: I64,
    tch: U8,
    dirty: I64,
    drift: F64,
    fstate: F64,
    half_sep: F64,
    hs_base: F64,
    acc: F64,
    istate: I64,
    pcost: F64,
    bprobe: I64,
    lcnt: I64,
    lstart: I64,
    gsplit: I64,
    ball: bool,  # noqa: FBT001
    max_iter: int,
    max_moved: int,
    tol_abs: float,
) -> bool:
    """Do the serial work once phase kind ph’s last item is done; True if the loop stops (the pass after the last update is done)."""
    if ball and ph == PH_SEPARATIONS:
        istate[_T0] = _clock_ticks()
    if ball and ph == PH_BALL_SUMS and istate[_ITER] >= 1:
        _record_pass_cost(istate, scans, pcost)
    if ph == (PH_BALL_SUMS if ball else PH_BOUNDS) and istate[_FINAL]:
        _store_release(istate, _STOP, 1)
        return True
    if with_ball and ph == PH_BOUNDS:
        _ball_offsets(lcnt, lstart, gsplit)
    if ph == PH_UPDATE:
        _end_of_iteration(
            x,
            centers,
            labels,
            bounds,
            new,
            tot,
            moved,
            scans,
            tch,
            dirty,
            drift,
            fstate,
            half_sep,
            hs_base,
            acc,
            istate,
            pcost,
            bprobe,
            ball,
            centers.shape[0],
            max_iter,
            max_moved,
            tol_abs,
        )
    return False


@numba.njit(inline="always", nogil=True, cache=True)
def _record_pass_cost(istate: I64, scans: I64, pcost: F64) -> None:
    """End of a pass with labels: its time per failing row, by kind (plain or ball)."""
    n_scan = 0
    for bb in range(scans.shape[0]):
        n_scan += scans[bb]
    if n_scan:
        pcost[istate[_BALL]] = (_clock_ticks() - istate[_T0]) / n_scan


@numba.njit(inline="always", nogil=True, cache=True)
def _end_of_iteration(  # noqa: PLR0917
    x: FArr,
    centers: F64,
    labels: I64,
    bounds: I64,
    new: F64,
    tot: I64,
    moved: I64,
    scans: I64,
    tch: U8,
    dirty: I64,
    drift: F64,
    fstate: F64,
    half_sep: F64,
    hs_base: F64,
    acc: F64,
    istate: I64,
    pcost: F64,
    bprobe: I64,
    ball: bool,  # noqa: FBT001
    k: int,
    max_iter: int,
    max_moved: int,
    tol_abs: float,
) -> None:
    """End of phase D: pick the next pass’s kind, move empty centers, take the new centers, test for convergence."""
    istate[_ITER] += 1
    n_moved, n_scan = 0, 0
    for bb in range(moved.shape[0]):
        n_moved += moved[bb]
        n_scan += scans[bb]
    if ball:
        _choose_pass_kind(istate, pcost, bprobe, n_scan, x.shape[0], k)
    _move_empty_centers(x, centers, labels, bounds, new, tot, tch, dirty)
    shift = 0.0
    for c in range(k):
        s = 0.0
        for j in range(centers.shape[1]):
            t = new[c, j] - centers[c, j]
            s += t * t
        drift[c] = math.sqrt(s)
        shift += drift[c] ** 2
    centers[:] = new
    tch[:] = 0
    istate[_FINAL] = n_moved <= max_moved or shift <= tol_abs or istate[_ITER] == max_iter
    _top2(drift, fstate)
    _stale_separations(istate, half_sep, hs_base, acc, drift, ball)


@numba.njit(inline="always", nogil=True, cache=True)
def _choose_pass_kind(istate: I64, pcost: F64, bprobe: I64, n_scan: int, n: int, k: int) -> None:  # noqa: PLR0917
    """Choose whether the next pass uses ball scans; labels are exact either way.

    Only if the bound test failed for many rows in a pass with labels (the first pass scans every row),
    and by the measured cost per failing row of both kinds.
    """
    if istate[_PREV] == 0:  # this pass’s failures come from plain-pass bounds
        pcost[2] = n_scan / n
    istate[_PREV] = istate[_BALL]
    nxt = 0
    if istate[_ITER] >= BALL_FROM_ITER and pcost[2] >= BALL_FRACTION and n_scan >= BALL_MIN_ROWS * k:
        nxt = 1 if pcost[1] == 0.0 else int(pcost[1] < pcost[0])
        if bprobe[1] >= 0:  # the last pass re-timed the other kind: wait twice as long if it confirmed
            bprobe[0] = 2 * bprobe[0] if nxt == bprobe[1] else BALL_PROBE
            bprobe[1] = -1
        if istate[_RUN] >= bprobe[0]:  # re-time the other kind now and then: the costs drift as Lloyd converges
            bprobe[1] = nxt
            nxt = 1 - nxt
    istate[_RUN] = istate[_RUN] + 1 if nxt == istate[_BALL] else 1
    istate[_BALL] = nxt


@numba.njit(inline="always", nogil=True, cache=True)
def _move_empty_centers(x: FArr, centers: F64, labels: I64, bounds: I64, new: F64, tot: I64, tch: U8, dirty: I64) -> None:  # noqa: PLR0917
    """As sklearn: empty centers move to the rows farthest from theirs (ties: lowest row); rare, so serial."""
    d = x.shape[1]
    k = centers.shape[0]
    n_empty = 0
    for c in range(k):
        n_empty += tot[c] == 0
    if not n_empty:
        return
    far = _farthest_rows(x, centers, labels)
    e = 0
    for c in range(k):
        if tot[c] == 0:
            for j in range(d):
                new[c, j] = x[far[e], j]
            if labels[far[e]] >= 0:
                tch[labels[far[e]]] = 1  # the row leaves its center
            labels[far[e]] = -1  # rescan next time
            for bb in range(bounds.shape[0] - 1):  # its chunk’s sums get recomputed (the row leaves its old center)
                if bounds[bb] <= far[e] < bounds[bb + 1]:
                    dirty[bb] = 1
            e += 1


@numba.njit(inline="always", nogil=True, cache=True)
def _farthest_rows(x: FArr, centers: F64, labels: I64) -> I64:
    """Rows by exact squared distance to their center, farthest first (ties: lowest row), as the reference ranks them."""
    n, d = x.shape
    dist = np.empty(n)
    for i in range(n):
        s = 0.0
        for j in range(d):
            t = np.float64(x[i, j]) - centers[labels[i], j]
            s += t * t
        dist[i] = s
    return np.argsort(-dist, kind="mergesort")


@numba.njit(inline="always", nogil=True, cache=True)
def _stale_separations(istate: I64, half_sep: F64, hs_base: F64, acc: F64, drift: F64, ball: bool) -> None:  # noqa: FBT001, PLR0917
    """Decide whether the next phase B keeps drift-adjusted separations or recomputes them.

    Kept bounds hs_base[c] - (acc[c] + max acc)/2 are valid by the triangle inequality;
    they are used while the drift since the last full computation is small next to them.
    """
    k = half_sep.shape[0]
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


@numba.njit(inline="always", nogil=True, cache=True)
def _top2(drift: F64, fstate: F64) -> None:
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
