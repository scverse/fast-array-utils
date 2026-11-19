# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import math

import numba
import numpy as np

from fast_array_utils.numba import njit

from ._ball import BALL_FRACTION, BALL_MIN_ROWS, BALL_PROBE, _ball_group, _ball_offsets, _ball_scatter, _ball_sums, _chunk_sums
from ._sync import SPIN_BEFORE_YIELD, _claim_range, _clock_ticks, _fetch_add, _load_acquire, _store_release, _yield_cpu
from ._tiles import SAFETY, SEED_ROWS, _dot_allowance, _scan_queue_dot, _score


SMALL_K = 64  # up to this many centers, rows failing the bound test go straight to the queued scan, without tightening


SUM_BUDGET = 1 << 22  # max float64 entries of per-chunk center sums (32 MB)


def _hamerly_bounds(n: int, k: int, d: int) -> np.ndarray:
    """Split the rows into fixed chunks from the shape alone, so results don't depend on threads.

    Chunks have ≥ 256 rows each, there are ≤ 256 of them, and their sums fit in SUM_BUDGET.
    """
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
    # fraction of the last pass that followed a plain pass (ball scans leave weaker lower bounds: |c_o - c_a| - u)
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
                        if ball and istate[_BALL]:
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
                            if ball and istate[_BALL]:
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
                        lb = lower[i] - (d2nd if drift[a] == dmax else dmax)
                        m = max(half_sep[a], lb)
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
                        lower[i] = lb
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
                    _ball_group(x, xnorm, crow, centers, cnorm, err_a, err_b, cd, acc, gsplit[item], gsplit[item + 1], lstart, order, labels, upper, lower)
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
