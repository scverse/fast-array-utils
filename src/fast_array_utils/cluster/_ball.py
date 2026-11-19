# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numba
import numpy as np

from ._tiles import SAFETY, TILE_CENTERS, _dot_tile_idx


if TYPE_CHECKING:
    from ._typing import F64, I64, U8, FArr


# Kernels take their arrays as positional arguments (one PLR0917 suppression per kernel, FBT001 for flags):
# numba can’t inline calls with keyword-only arguments.

BALL_MIN_K, BALL_MAX_K = 256, 2048  # k range with ball scans (k×k center distances: ≤ 32 MB)


# ball scans only while plain-pass bounds leave this fraction failing (high-dimensional data with many centers fails
# far more often than well-separated blobs)
BALL_FRACTION = 0.8


BALL_MIN_ROWS = 8  # ...and only with this many failing rows per label on average: tiles take 16 rows


# ...and only while measured cheaper per failing row than the plain pass; the other kind is re-timed after this many
# passes, then at doubling intervals while confirmed
BALL_PROBE = 8


BALL_TILE_ROWS = 16  # rows per `_dot_tile_idx` call (8 float64 or float32 lanes, 2 vectors)


@numba.njit(inline="always", nogil=True, cache=True)
def _ball_label(  # noqa: PLR0917
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
    Labels as in `_scan_queue_dot`: the dot form with per-center rounding allowances,
    exact float64 distances when another candidate could be nearest.
    Second bound: the smaller of the candidates’ and of |c_o - c_a| - u over the centers not scanned.
    """
    m = rows.shape[0]
    rows[:] = rows[np.argsort(upper[rows])]
    kc, oute = _ball_candidates(cd, acc, a, 2.0 * upper[rows[m - 1]] * (1 + 4 * SAFETY), cand, keys)
    srt = cand[:kc][np.argsort(keys[:kc])]
    kc4 = (kc + TILE_CENTERS - 1) // TILE_CENTERS * TILE_CENTERS
    for t in range(kc4):
        sidx[t] = srt[min(t, kc - 1)]  # padding columns repeat a real center (ignored)
    xs, d1, l1, l2 = sel[0], sel[1], sel[2], sel[3]
    bb, lb1 = np.zeros(BALL_TILE_ROWS, dtype=np.int64), np.zeros(BALL_TILE_ROWS, dtype=np.int64)
    for s0 in range(0, m, BALL_TILE_ROWS):
        nr = min(BALL_TILE_ROWS, m - s0)
        pe, p4, out = _ball_reach(cd, acc, a, srt, kc, kc4, 2.0 * upper[rows[s0 + nr - 1]] * (1 + 4 * SAFETY), oute)
        _gather_group(x, xnorm, rows, s0, nr, xb, xs)
        for cq in range(0, p4, TILE_CENTERS):
            _dot_tile_idx(xb, crow, sidx, dv, 0, cq)
        _nearest_two_of(dv, srt, pe, cnorm, err_a, err_b, xs, d1, l1, l2, bb, lb1)
        for r in range(nr):
            i = rows[s0 + r]
            rest = (out - upper[i]) * (1 - SAFETY)  # every center not scanned is at least this far from row i
            best, e1, e2 = _label_ball_row(x, xnorm[i], i, r, centers, dv, srt, pe, cnorm, err_a, err_b, xs[r], d1[r], l1[r], l2[r], bb[r], lb1[r])
            labels[i] = best
            upper[i] = math.sqrt(e1)
            lower[i] = max(min(math.sqrt(e2), rest), 0.0)


@numba.njit(inline="always", nogil=True, cache=True)
def _ball_candidates(cd: F64, acc: F64, a: int, rad_all: float, cand: I64, keys: F64) -> tuple[int, float]:  # noqa: PLR0917
    """Centers o whose lower bound on |c_o - c_a| is within rad_all (cand[:kc], keyed by it); returns kc and the smallest bound outside."""
    kc, oute = 0, np.inf
    for o in range(cd.shape[0]):
        lb = (cd[a, o] - acc[a] - acc[o]) * (1 - SAFETY)
        if lb <= rad_all:
            cand[kc] = o
            keys[kc] = lb
            kc += 1
        else:
            oute = min(oute, lb)
    return kc, oute


@numba.njit(inline="always", nogil=True, cache=True)
def _ball_reach(cd: F64, acc: F64, a: int, srt: I64, kc: int, kc4: int, rad: float, oute: float) -> tuple[int, int, float]:  # noqa: PLR0917
    """Candidates to scan for a row group of radius rad: (scanned, tiled columns, smallest bound of the rest).

    The padded tile columns may add a few beyond the radius: harmless.
    """
    p = kc
    while p > 0 and (cd[a, srt[p - 1]] - acc[a] - acc[srt[p - 1]]) * (1 - SAFETY) > rad:
        p -= 1
    p4 = min((p + TILE_CENTERS - 1) // TILE_CENTERS * TILE_CENTERS, kc4)
    pe = min(p4, kc)
    out = (cd[a, srt[pe]] - acc[a] - acc[srt[pe]]) * (1 - SAFETY) if pe < kc else oute
    return pe, p4, out


@numba.njit(inline="always", nogil=True, cache=True)
def _gather_group(x: FArr, xnorm: F64, rows: I64, s0: int, nr: int, xb: FArr, xs: F64) -> None:  # noqa: PLR0917
    """xb[:, r] = x[rows[s0 + r]] and xs[r] = ‖x‖ for r < nr; the rest of the BALL_TILE_ROWS columns zero."""
    d = x.shape[1]
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


@numba.njit(inline="always", nogil=True, cache=True)
def _nearest_two_of(dv: FArr, srt: I64, pe: int, cnorm: FArr, err_a: F64, err_b: F64, xs: F64, d1: F64, l1: F64, l2: F64, bb: I64, lb1: I64) -> None:  # noqa: PLR0917
    """`_tiles._nearest_two` over the candidates srt[:pe] (tile column t is center srt[t]); bb and lb1 hold columns."""
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


@numba.njit(inline="always", nogil=True, cache=True)
def _label_ball_row(  # noqa: PLR0917
    x: FArr,
    xn: float,
    i: int,
    r: int,
    centers: F64,
    dv: FArr,
    srt: I64,
    pe: int,
    cnorm: FArr,
    err_a: F64,
    err_b: F64,
    xsr: float,
    v1: float,
    l1r: float,
    l2r: float,
    bbr: int,
    lb1r: int,
) -> tuple[int, float, float]:
    """`_tiles._label_row` over the candidates srt[:pe]: row i’s label and squared distances to its nearest two."""
    best = srt[bbr]
    reach = v1 + err_a[best] + err_b[best] * xsr  # ≥ the nearest candidate’s true v
    wo = l1r if lb1r != bbr else l2r
    if wo > reach:
        return best, xn + reach, max(xn + wo, 0.0)
    best, e1, e2, wr = -1, np.inf, np.inf, np.inf
    for t in range(pe):
        o = srt[t]
        w = np.float64(cnorm[o]) - 2.0 * np.float64(dv[t, r]) - (err_a[o] + err_b[o] * xsr)
        if w <= reach:
            e = 0.0
            for j in range(x.shape[1]):
                q = np.float64(x[i, j]) - centers[o, j]
                e += q * q
            if e < e1 or (e == e1 and o < best):  # ties: lowest index, as the reference
                e2 = e1
                best, e1 = o, e
            elif e < e2:
                e2 = e
        else:
            wr = min(wr, w)
    return best, e1, min(e2, max(xn + wr, 0.0))


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
    keys, cand, sidx = np.empty(k), np.empty(k, dtype=np.int64), np.empty(k + 4, dtype=np.int64)
    dv, xb, sel = np.empty((k + 4, 16), dtype=x.dtype), np.zeros((d, 16), dtype=x.dtype), np.empty((4, 16))
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

    As phase C does without ball scans: many moves, the chunk's sums afresh in row order; few, those rows moved, in
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


MOVED_RECOMPUTE = 4  # a chunk with more than 1/4 of its rows moved recomputes its sums; fewer are moved incrementally
