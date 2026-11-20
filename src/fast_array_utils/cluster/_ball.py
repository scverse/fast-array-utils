# SPDX-License-Identifier: MPL-2.0
"""Ball scans for large k: rows that failed the bound test are scanned per label, against the centers within reach."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numba
import numpy as np

from ._tiles import SAFETY, TILE_CENTERS, _dot_tile_idx


if TYPE_CHECKING:
    from ._typing import F64, I64, U8, FArr


BALL_MIN_K, BALL_MAX_K = 256, 2048  # k range with ball scans (k×k center distances: ≤ 32 MB)
BALL_FRACTION = 0.8  # ball scans only while plain-pass bounds leave this fraction of rows failing
BALL_MIN_ROWS = 8  # and only with this many failing rows per label on average (tiles take 16 rows)
BALL_PROBE = 8  # and only while cheaper per failing row; the other kind is re-timed after this many passes, then doubling
BALL_TILE_ROWS = 16  # rows per `_dot_tile_idx` call (2 vectors of 8 lanes)
SEL_XS, SEL_D1, SEL_L1, SEL_L2 = 0, 1, 2, 3  # rows of `_ball_label`’s sel: ‖x‖, smallest v, two smallest w
SEL_ROWS = 4
MOVED_RECOMPUTE = 4  # a chunk with more than 1/4 of its rows moved recomputes its sums; fewer are moved incrementally
RADIUS_MARGIN = 1 + 4 * SAFETY  # margin on the ball radius 2u, so rounding can never leave a nearer center out


@numba.njit(inline="always", nogil=True, cache=True)
def _ball_label(  # noqa: C901, PLR0912, PLR0915, PLR0917
    x: FArr,
    xnorm: F64,
    crow: FArr,
    centers: F64,
    cnorm: FArr,
    err_a: F64,
    err_b: F64,
    cd: F64,
    acc: F64,
    a: int,
    rows: I64,
    labels: I64,
    upper: F64,
    lower: F64,
    keys: F64,
    cand: I64,
    sidx: I64,
    dv: FArr,
    xb: FArr,
    sel: F64,
) -> None:
    """Exact scans of the rows labelled a that failed the bound test, against the centers that could be nearer.

    Every center nearer to row i than c_a lies in the ball |c_o - c_a| ≤ 2u_i (u_i = the exact distance to c_a, from the bound pass).
    Candidates are ordered by a lower bound on |c_o - c_a| (center distances cd at the last full computation minus the drift since, acc),
    rows by u; each 16-row group is tiled (`_dot_tile_idx`) against the candidates inside its largest radius.
    Labels as in `_tiles._scan_queue_dot`: the dot form with per-center rounding allowances,
    exact float64 distances when another candidate could be nearest.
    Second bound: the smaller of the candidates’ and of |c_o - c_a| - u over the centers not scanned.
    The selection loops mirror `_scan_queue_dot`’s over the candidates srt[:pe];
    they stay separate because the plain scan reads the centers in order, which keeps its hot loop free of the index load.
    """
    k, d = crow.shape
    m = rows.shape[0]
    rows[:] = rows[np.argsort(upper[rows])]
    # candidates: centers whose lower bound on |c_o - c_a| is within the largest radius, sorted by it
    rad_all = 2.0 * upper[rows[m - 1]] * RADIUS_MARGIN
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
    kc4 = (kc + TILE_CENTERS - 1) // TILE_CENTERS * TILE_CENTERS
    for t in range(kc4):
        sidx[t] = srt[min(t, kc - 1)]  # padding columns repeat a real center (ignored)
    xs, d1, l1, l2 = sel[SEL_XS], sel[SEL_D1], sel[SEL_L1], sel[SEL_L2]
    bb, lb1 = np.zeros(BALL_TILE_ROWS, dtype=np.int64), np.zeros(BALL_TILE_ROWS, dtype=np.int64)
    for s0 in range(0, m, BALL_TILE_ROWS):
        nr = min(BALL_TILE_ROWS, m - s0)
        # the candidates within this group’s radius (padded tile columns may add a few beyond it: harmless)
        rad = 2.0 * upper[rows[s0 + nr - 1]] * RADIUS_MARGIN
        p = kc
        while p > 0 and (cd[a, srt[p - 1]] - acc[a] - acc[srt[p - 1]]) * (1 - SAFETY) > rad:
            p -= 1
        p4 = min((p + TILE_CENTERS - 1) // TILE_CENTERS * TILE_CENTERS, kc4)
        pe = min(p4, kc)  # candidates scanned
        out = (cd[a, srt[pe]] - acc[a] - acc[srt[pe]]) * (1 - SAFETY) if pe < kc else oute
        for r in range(BALL_TILE_ROWS):
            if r < nr:
                i = rows[s0 + r]
                xs[r] = math.sqrt(xnorm[i])
                for j in range(d):
                    xb[j, r] = x[i, j]
            else:
                xs[r] = 0.0
                for j in range(d):
                    xb[j, r] = 0
        for cq in range(0, p4, TILE_CENTERS):
            _dot_tile_idx(xb, crow, sidx, dv, 0, cq)
        for r in range(BALL_TILE_ROWS):
            d1[r], l1[r], l2[r] = np.inf, np.inf, np.inf
            bb[r], lb1[r] = 0, -1
        for t in range(pe):  # vectorized over the rows; dot-form ties fall to the exact rescans (w_other ≤ v1 ≤ reach)
            o = srt[t]
            cn, ea, eb = np.float64(cnorm[o]), err_a[o], err_b[o]
            for r in range(BALL_TILE_ROWS):
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
            reach = d1[r] + err_a[best] + err_b[best] * xs[r]  # ≥ the nearest candidate’s true v
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
def _ball_scatter(bounds: I64, b0: int, b1: int, qn: I64, qbuf: I64, qold: I64, lcnt: I64, order: I64) -> None:  # noqa: PLR0917
    """Phase C2: the recorded rows of chunks b0..b1 into label order (chunk order, then row order, per label).

    The ball phases are separate functions (not inlined) so the plain Hamerly loop stays as small as without them.
    """
    for b in range(b0, b1):
        lo = bounds[b]
        for t in range(qn[b]):
            a = qold[lo + t]
            if a >= 0:  # rows without a label were scanned in phase C
                order[lcnt[b, a]] = qbuf[lo + t]
                lcnt[b, a] += 1


@numba.njit(nogil=True, cache=True)
def _ball_group(  # noqa: PLR0917
    x: FArr,
    xnorm: F64,
    crow: FArr,
    centers: F64,
    cnorm: FArr,
    err_a: F64,
    err_b: F64,
    cd: F64,
    acc: F64,
    a0: int,
    a1: int,
    lstart: I64,
    order: I64,
    labels: I64,
    upper: F64,
    lower: F64,
) -> None:
    """Phase E: ball scans of the labels a0..a1 (`_ball_label`)."""
    k, d = crow.shape
    keys, cand, sidx = np.empty(k), np.empty(k, dtype=np.int64), np.empty(k + TILE_CENTERS, dtype=np.int64)
    dv, xb = np.empty((k + TILE_CENTERS, BALL_TILE_ROWS), dtype=x.dtype), np.zeros((d, BALL_TILE_ROWS), dtype=x.dtype)
    sel = np.empty((SEL_ROWS, BALL_TILE_ROWS))
    for a in range(a0, a1):
        if lstart[a + 1] > lstart[a]:
            _ball_label(
                x, xnorm, crow, centers, cnorm, err_a, err_b, cd, acc, a, order[lstart[a] : lstart[a + 1]], labels, upper, lower, keys, cand, sidx, dv, xb, sel
            )


@numba.njit(nogil=True, cache=True)
def _ball_sums(  # noqa: PLR0917
    x: FArr, bounds: I64, b0: int, b1: int, qn: I64, qbuf: I64, qold: I64, labels: I64, sums: F64, counts: I64, tch: U8, dirty: I64, moved: I64
) -> None:
    """Phase C3: chunk sums of chunks b0..b1 for their rows that moved in phases C and E.

    As phase C does without ball scans: many moves, the chunk’s sums afresh in row order; few, those rows moved, in
    row order.
    """
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
def _chunk_sums(  # noqa: PLR0917
    x: FArr, b: int, lo: int, hi: int, rows: I64, old: I64, r0: int, nq: int, mv: int, labels: I64, sums: F64, counts: I64, tch: U8, dirty: I64
) -> None:
    """Chunk b’s center sums after mv > 0 of its nq recorded rows (rows[r0 + t], previous labels old[r0 + t]) moved.

    Many moves (or rows taken by the empty-center rule): the chunk’s sums afresh, rows in order. Few: those rows move
    between their old and new centers’ sums, in row order. Both orders depend only on the data, not on threads.
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
def _ball_offsets(lcnt: I64, lstart: I64, gsplit: I64) -> None:
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
