# SPDX-License-Identifier: MPL-2.0
"""Exact Lloyd with Hamerly’s bounds in one persistent parallel region (`_hamerly`)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numba
import numpy as np

from fast_array_utils.numba import njit

from ._ball import BALL_FRACTION, BALL_MIN_ROWS, BALL_PROBE, _ball_group, _ball_offsets, _ball_scatter, _ball_sums, _chunk_sums
from ._sync import (
    _BALL,
    _CD_OK,
    _DONE,
    _FINAL,
    _ITER,
    _KEEP_SEP,
    _PHASE,
    _PREV,
    _RUN,
    _STOP,
    _T0,
    _TICKET,
    ISTATE_SLOTS,
    LINE_SLOTS,
    _backoff,
    _claim_range,
    _clock_ticks,
    _fetch_add,
    _load_acquire,
    _store_release,
)
from ._tiles import BLOCK_ROWS, SAFETY, _dot_allowance, _scan_queue_dot, _score


if TYPE_CHECKING:
    from ._typing import F64, I64, U8, FArr


BALL_FROM_ITER = 2  # ball scans from the second pass with labels on (the first pass scans every row)
SMALL_K = 64  # up to this many centers, rows failing the bound test go straight to the queued scan, without tightening
SUM_BUDGET = 1 << 22  # max float64 entries of per-chunk center sums (32 MB)
CHUNK_MIN_ROWS, MAX_CHUNKS = 256, 256  # row chunks have at least this many rows, and there are at most this many
SEP_STALE = 0.05  # phase B skips the separations while the accumulated drift is ≤ this × the mean half-separation
CENTER_ITEMS = 16  # centers per work item in the per-center phases (4 and 64 are slower: ticket cost vs balance)
ROW_ITEMS = 4  # rows of the transposed centers per work item
LABEL_GROUPS_PER_THREAD = 4  # work items per thread of phase E
CHUNK_GROUPS_PER_THREAD = 2  # work items per thread of phases C2 and C3

# phase kinds, in the order of one iteration; C2, E and C3 only with ball scans
PH_TRANSPOSE, PH_SEPARATIONS, PH_BOUNDS, PH_UPDATE = 0, 1, 2, 3  # A, B, C, D
PH_SCATTER, PH_BALL_SCAN, PH_BALL_SUMS = 4, 5, 6  # C2: failing rows into label order, E: ball scans, C3: their chunk sums

TOP_MAX, TOP_SECOND = 0, 1  # slots of fstate: largest and second-largest drift
COST_PLAIN, COST_BALL, FAIL_FRACTION = 0, 1, 2  # slots of pcost (pcost[istate[_BALL]] is this pass kind’s cost)
PROBE_WAIT, PROBE_KIND = 0, 1  # slots of probe: passes between re-timings of the other kind, kind before a re-timing


def _hamerly_bounds(n: int, k: int, d: int) -> I64:
    """Split the rows into fixed chunks from the shape alone, so results don’t depend on threads.

    Chunks have ≥ CHUNK_MIN_ROWS rows each, there are ≤ MAX_CHUNKS of them, and their sums fit in SUM_BUDGET.
    """
    nb = max(1, min(-(-n // CHUNK_MIN_ROWS), MAX_CHUNKS, SUM_BUDGET // (k * d)))
    return (np.arange(nb + 1, dtype=np.int64) * n) // nb


@njit
def _hamerly(  # noqa: C901, PLR0912, PLR0915, PLR0917
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
    If ball scans may run (cd is k×k, else 1×1), three phases follow C: C2 sorts the rows that failed the bound test by label,
    E scans them per label against the centers within reach (`_ball._ball_label`), C3 updates the chunk sums.
    They work only in passes chosen at the end of the previous one (many failing rows, few centers within reach);
    labels are exact either way, so results don’t depend on that choice.
    """
    # prepare memory
    n, d = x.shape
    k = centers.shape[0]
    nb = bounds.shape[0] - 1
    ball = cd.shape[0] > 1
    upper, lower, sums, counts, moved, scans, tch, dirty, new, tot = _row_and_sum_state(n, nb, k, d)
    ct, cnorm, err_a, err_b, ctb, half_sep, drift, hs_base, acc, fstate = _center_state(x, k)
    kb, nbl = (k, n) if ball else (1, 1)
    ne = min(k, LABEL_GROUPS_PER_THREAD * nthreads) if ball else 0  # label groups (phase E items)
    ngc = min(nb, CHUNK_GROUPS_PER_THREAD * nthreads)  # chunk groups (phase C2 and C3 items)
    crow, qbuf, qold, order, qn, lcnt, lstart, gsplit, probe, pcost = _ball_state(x, kb, nbl, nb, ne)
    rcnt = np.zeros(LINE_SLOTS * nthreads, dtype=np.int64)  # phase C claims so far from each worker’s chunk range
    istate = np.zeros(ISTATE_SLOTS, dtype=np.int64)
    istate[_FINAL] = max_iter == 0
    kinds, offs = _phase_plan(d, k, nb, ngc, ne)
    nph, per_it = kinds.shape[0], offs[-1]
    pass_end = PH_BALL_SUMS if ball else PH_BOUNDS  # the bound pass (incl. its scans and sums) is complete after this phase kind
    maxrows = np.max(bounds[1:] - bounds[:-1])

    # run the update step until converged, in one region instead of one per phase (their wake-ups dominated small passes:
    # 1.3× on 20 000 rows); tickets in order (iteration, phase, item), nobody waits for a particular thread, so it can’t deadlock
    for wi in numba.prange(nthreads):
        queue = np.empty(maxrows, dtype=np.int64)  # full scans go into a queue, then run BLOCK_ROWS rows at a time
        oldl = np.empty(maxrows, dtype=np.int64)  # the queued rows’ labels before the scan
        dvs = np.empty((k, BLOCK_ROWS))  # squared distances of an item’s centers to one block of centers
        lqc, sepmin = np.empty(CENTER_ITEMS, dtype=np.int64), np.empty(CENTER_ITEMS)
        while True:
            tit, pp, item = _ticket(istate, offs)
            g = nph * tit + pp
            if _wait_phase(istate, g):
                break
            ph = kinds[pp]
            with_ball = istate[_BALL] != 0  # set only if ball scans may run
            if ph == PH_TRANSPOSE:  # whole rows of ct per item: no false sharing
                for j in range(item * ROW_ITEMS, min(d, (item + 1) * ROW_ITEMS)):
                    for c in range(k):
                        ct[j, c] = centers[c, j]
                        ctb[c // BLOCK_ROWS, j, c % BLOCK_ROWS] = centers[c, j]
                        if with_ball:
                            crow[c, j] = centers[c, j]
            elif ph == PH_SEPARATIONS:
                _separations(item, centers, cnorm, err_a, err_b, unit, tiny, istate, ctb, lqc, sepmin, dvs, cd, half_sep, hs_base, acc)
            elif ph == PH_BOUNDS:
                b = _claim_chunk(rcnt, wi, tit, nb, nthreads)
                _bound_pass(
                    x, xnorm, centers, ct, cnorm, err_a, err_b, labels, upper, lower, half_sep, drift, fstate, bounds, b, istate,
                    queue, oldl, qbuf, qold, qn, lcnt, moved, scans, sums, counts, tch, dirty, pfd,
                )  # fmt: skip
            elif ph == PH_UPDATE:  # new means of the item’s centers (centers whose sums didn’t change keep theirs)
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
                            q = 0.0
                            for bb in range(nb):
                                q += sums[bb, c, j]
                            new[c, j] = q / cnt
            elif with_ball and ph == PH_SCATTER:
                _ball_scatter(bounds, item * nb // ngc, (item + 1) * nb // ngc, qn, qbuf, qold, lcnt, order)
            elif with_ball and ph == PH_BALL_SCAN:
                _ball_group(x, xnorm, crow, centers, cnorm, err_a, err_b, cd, acc, gsplit[item], gsplit[item + 1], lstart, order, labels, upper, lower)
            elif with_ball:
                _ball_sums(x, bounds, item * nb // ngc, (item + 1) * nb // ngc, qn, qbuf, qold, labels, sums, counts, tch, dirty, moved)
            if _fetch_add(istate, _DONE) != tit * per_it + offs[pp + 1] - 1:
                continue
            # the last item of phase g: its serial part, then publish phase g + 1
            if ball and ph == PH_SEPARATIONS:
                istate[_T0] = _clock_ticks()
            if ball and ph == PH_BALL_SUMS and istate[_ITER] >= 1:
                _record_pass_cost(istate, scans, pcost)
            if ph == pass_end and istate[_FINAL]:
                _store_release(istate, _STOP, 1)  # the pass after the last update is done
                break
            if with_ball and ph == PH_BOUNDS:
                _ball_offsets(lcnt, lstart, gsplit)
            if ph == PH_UPDATE:
                _end_of_iteration(
                    x, centers, labels, bounds, new, tot, moved, scans, tch, dirty, drift, fstate, half_sep, hs_base, acc,
                    istate, pcost, probe, cd, max_iter, max_moved, tol_abs,
                )  # fmt: skip
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
        t = 0.0
        for i in range(bounds[b], bounds[b + 1]):
            for j in range(d):
                q = np.float64(x[i, j]) - centers[labels[i], j]
                t += q * q
        part[b] = t
    inertia = 0.0
    for b in range(nb):
        inertia += part[b]
    return inertia


# ----------------------------------------------------------------------------- memory and schedule
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
    ctb = np.zeros(((k + BLOCK_ROWS - 1) // BLOCK_ROWS, d, BLOCK_ROWS))  # float64 centers, block-transposed like seeding’s rows
    half_sep, drift = np.empty(k), np.zeros(k)
    hs_base, acc = np.zeros(k), np.zeros(k)  # separations at the last full computation, drift since then
    fstate = np.empty(2)
    _top2(drift, fstate)
    return ct, cnorm, err_a, err_b, ctb, half_sep, drift, hs_base, acc, fstate


@numba.njit(inline="always", nogil=True, cache=True)
def _ball_state(x: FArr, kb: int, nbl: int, nb: int, ne: int) -> tuple[FArr, I64, I64, I64, I64, I64, I64, I64, I64, F64]:
    """Buffers of the ball scans (kb = k and nbl = n with them, else 1) and the bookkeeping of their pass-kind choice."""
    crow = np.empty((kb, x.shape[1]), dtype=x.dtype)  # centers in x’s dtype, row-major, for `_dot_tile_idx`
    qbuf, qold, order = np.empty(nbl, dtype=np.int64), np.empty(nbl, dtype=np.int64), np.empty(nbl, dtype=np.int64)
    qn = np.zeros(nb, dtype=np.int64)  # rows recorded per chunk (at qbuf[bounds[b]:]) with their old labels (qold)
    lcnt = np.zeros((nb if ne else 1, kb), dtype=np.int64)  # failing rows per chunk and label, then scatter offsets
    lstart = np.zeros(kb + 1, dtype=np.int64)  # label a’s rows: order[lstart[a]:lstart[a + 1]]
    gsplit = np.zeros(ne + 1, dtype=np.int64)  # label groups of phase E: labels gsplit[e]..gsplit[e + 1]
    probe = np.array([BALL_PROBE, -1], dtype=np.int64)
    pcost = np.zeros(3)
    return crow, qbuf, qold, order, qn, lcnt, lstart, gsplit, probe, pcost


@numba.njit(inline="always", nogil=True, cache=True)
def _phase_plan(d: int, k: int, nb: int, ngc: int, ne: int) -> tuple[I64, I64]:
    """Phase kinds of one iteration (with the ball phases if there are ne > 0 label groups) and each one’s first item.

    offs[-1] = items per iteration.
    """
    na, nc = (d + ROW_ITEMS - 1) // ROW_ITEMS, (k + CENTER_ITEMS - 1) // CENTER_ITEMS
    if ne:
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
def _ticket(istate: I64, offs: I64) -> tuple[int, int, int]:
    """Draw the next ticket: (iteration, phase index, item within the phase)."""
    per_it = offs[-1]
    ticket = _fetch_add(istate, _TICKET)
    tit, r = ticket // per_it, ticket % per_it
    pp = 0
    while r >= offs[pp + 1]:
        pp += 1
    return tit, pp, r - offs[pp]


@numba.njit(inline="always", nogil=True, cache=True)
def _wait_phase(istate: I64, g: int) -> bool:
    """Wait until phase g may run; True if the loop has stopped instead."""
    spins = 0
    while True:
        if _load_acquire(istate, _STOP):
            return True
        if _load_acquire(istate, _PHASE) >= g:
            return False
        spins = _backoff(spins)


@numba.njit(inline="always", nogil=True, cache=True)
def _claim_chunk(rcnt: I64, wi: int, tit: int, nb: int, nthreads: int) -> int:
    """Claim a row chunk of iteration tit’s bound pass, from worker wi’s own range first (its rows stay in this core’s caches)."""
    v = wi % nthreads  # an int64 copy of the prange index: v is reassigned below, and (uint64 + 1) % int64 types as float64
    b = _claim_range(rcnt, v, tit, nb, nthreads)
    while b < 0:
        v = (v + 1) % nthreads
        b = _claim_range(rcnt, v, tit, nb, nthreads)
    return b


# ----------------------------------------------------------------------------- phases B and C
@numba.njit(inline="always", nogil=True, cache=True)
def _separations(  # noqa: C901, PLR0917
    item: int,
    centers: F64,
    cnorm: FArr,
    err_a: F64,
    err_b: F64,
    unit: float,
    tiny: float,
    istate: I64,
    ctb: F64,
    lqc: I64,
    sepmin: F64,
    dvs: F64,
    cd: F64,
    half_sep: F64,
    hs_base: F64,
    acc: F64,
) -> None:
    """Phase B: norms and rounding allowances of the item’s centers, and their half-separations.

    Separations: the item’s centers against blocks of BLOCK_ROWS centers by FMA tiles (float64),
    also kept in cd for a ball pass; skipped while they stay the drift-adjusted bounds set after phase D.
    """
    k, d = centers.shape
    c_lo = item * CENTER_ITEMS
    nl = min(k, c_lo + CENTER_ITEMS) - c_lo
    for t in range(nl):
        c = c_lo + t
        lqc[t] = c
        sepmin[t] = np.inf
        q = 0.0
        for j in range(d):
            q += centers[c, j] * centers[c, j]
        cnorm[c] = q
        err_a[c], err_b[c] = _dot_allowance(q, unit, tiny, d)
    if istate[_KEEP_SEP]:
        return
    with_cd = istate[_BALL] != 0
    for bo in range(ctb.shape[0]):
        for t in range(nl):
            dvs[c_lo + t, :] = 0.0
        _score(ctb, bo, centers, lqc, nl, dvs, 0, d)
        for t in range(nl):
            c = c_lo + t
            for r in range(min(BLOCK_ROWS, k - bo * BLOCK_ROWS)):
                if bo * BLOCK_ROWS + r != c:
                    sepmin[t] = min(sepmin[t], dvs[c, r])
                if with_cd:
                    cd[c, bo * BLOCK_ROWS + r] = math.sqrt(dvs[c, r])
    for t in range(nl):
        half_sep[c_lo + t] = 0.5 * math.sqrt(sepmin[t])
        hs_base[c_lo + t] = half_sep[c_lo + t]
        acc[c_lo + t] = 0.0


@numba.njit(inline="always", nogil=True, cache=True)
def _bound_pass(  # noqa: C901, PLR0912, PLR0917
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
    bounds: I64,
    b: int,
    istate: I64,
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
    """Phase C on row chunk b: Hamerly’s bound test, then exact scans of the rows that fail it.

    Without ball scans, the failing rows are scanned here (`_tiles._scan_queue_dot`) and the chunk sums updated;
    in a ball pass, rows that have a label are recorded for phases C2, E and C3.
    """
    k, d = centers.shape
    lo, hi = bounds[b], bounds[b + 1]
    with_ball = istate[_BALL] != 0
    dmax, d2nd = fstate[TOP_MAX], fstate[TOP_SECOND]
    mv, sc = 0, 0
    nq, nrec = 0, 0
    if with_ball:
        lcnt[b, :] = 0
    for i in range(lo, hi):
        a = labels[i]
        need = a < 0
        if not need:  # Hamerly’s test, with drift-updated bounds
            u = upper[i] + drift[a]
            lb = lower[i] - (d2nd if drift[a] == dmax else dmax)
            m = max(half_sep[a], lb)
            if u * (1 + SAFETY) > m * (1 - SAFETY):
                if k <= SMALL_K:  # small k: the queued scan is cheap and computes the exact distance anyway
                    need = True
                else:
                    q = 0.0
                    for j in range(d):
                        t = np.float64(x[i, j]) - centers[a, j]
                        q += t * t
                    u = math.sqrt(q)
                    need = u * (1 + SAFETY) > m * (1 - SAFETY)
            upper[i] = u
            lower[i] = lb
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


# ----------------------------------------------------------------------------- serial work at the end of a phase
@numba.njit(inline="always", nogil=True, cache=True)
def _record_pass_cost(istate: I64, scans: I64, pcost: F64) -> None:
    """End of a pass with labels: its time per failing row, by kind (plain or ball)."""
    n_scan = 0
    for bb in range(scans.shape[0]):
        n_scan += scans[bb]
    if n_scan:
        pcost[istate[_BALL]] = (_clock_ticks() - istate[_T0]) / n_scan


@numba.njit(inline="always", nogil=True, cache=True)
def _end_of_iteration(  # noqa: C901, PLR0912, PLR0915, PLR0917
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
    probe: I64,
    cd: F64,
    max_iter: int,
    max_moved: int,
    tol_abs: float,
) -> None:
    """End of phase D: pick the next pass’s kind, move empty centers, take the new centers, test for convergence, update the separations."""
    n, d = x.shape
    k = centers.shape[0]
    ball = cd.shape[0] > 1
    istate[_ITER] += 1
    n_moved, n_scan = 0, 0
    for bb in range(moved.shape[0]):
        n_moved += moved[bb]
        n_scan += scans[bb]
    if ball:  # ball scans next pass? Only if the bound test failed for many rows in a pass with labels (the first pass
        # scans every row), and by the measured cost per failing row of both kinds; labels are exact either way
        if istate[_PREV] == 0:  # this pass’s failures come from plain-pass bounds
            pcost[FAIL_FRACTION] = n_scan / n
        istate[_PREV] = istate[_BALL]
        nxt = 0
        if istate[_ITER] >= BALL_FROM_ITER and pcost[FAIL_FRACTION] >= BALL_FRACTION and n_scan >= BALL_MIN_ROWS * k:
            nxt = 1 if pcost[COST_BALL] == 0.0 else int(pcost[COST_BALL] < pcost[COST_PLAIN])
            if probe[PROBE_KIND] >= 0:  # the last pass re-timed the other kind: wait twice as long if it confirmed
                probe[PROBE_WAIT] = 2 * probe[PROBE_WAIT] if nxt == probe[PROBE_KIND] else BALL_PROBE
                probe[PROBE_KIND] = -1
            if istate[_RUN] >= probe[PROBE_WAIT]:  # re-time the other kind now and then: the costs drift as Lloyd converges
                probe[PROBE_KIND] = nxt
                nxt = 1 - nxt
        istate[_RUN] = istate[_RUN] + 1 if nxt == istate[_BALL] else 1
        istate[_BALL] = nxt
    n_empty = 0
    for c in range(k):
        n_empty += tot[c] == 0
    if n_empty:  # as sklearn: empty centers move to the rows farthest from theirs (ties: lowest row); rare, so serial
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
    shift = 0.0
    for c in range(k):
        q = 0.0
        for j in range(d):
            t = new[c, j] - centers[c, j]
            q += t * t
        drift[c] = math.sqrt(q)
        shift += drift[c] ** 2
    centers[:] = new
    tch[:] = 0
    istate[_FINAL] = n_moved <= max_moved or shift <= tol_abs or istate[_ITER] == max_iter
    _top2(drift, fstate)
    # separations: keep valid bounds hs_base[c] - (acc[c] + max acc)/2 (triangle inequality) while the drift
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


@numba.njit(inline="always", nogil=True, cache=True)
def _farthest_rows(x: FArr, centers: F64, labels: I64) -> I64:
    """Rows by exact squared distance to their center, farthest first (ties: lowest row), as the reference ranks them."""
    n, d = x.shape
    dist = np.empty(n)
    for i in range(n):
        q = 0.0
        for j in range(d):
            t = np.float64(x[i, j]) - centers[labels[i], j]
            q += t * t
        dist[i] = q
    return np.argsort(-dist, kind="mergesort")


@numba.njit(inline="always", nogil=True, cache=True)
def _top2(drift: F64, fstate: F64) -> None:
    """Set fstate to the largest and second-largest drift (equal on ties; second is 0 for k = 1)."""
    dmax, d2nd = -np.inf, -np.inf
    for c in range(drift.shape[0]):
        if drift[c] > dmax:
            dmax, d2nd = drift[c], dmax
        elif drift[c] > d2nd:
            d2nd = drift[c]
    if drift.shape[0] == 1:
        d2nd = 0.0
    fstate[TOP_MAX], fstate[TOP_SECOND] = dmax, d2nd
