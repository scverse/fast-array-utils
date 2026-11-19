# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import math

import numba
import numpy as np
from llvmlite import ir
from numba.core import cgutils, types
from numba.extending import intrinsic

from ._sync import _prefetch_row


@numba.njit(inline="always", nogil=True, cache=True)
def _dot_allowance(cn, unit, tiny, d):
    """Return (a, b) with a + b·‖x‖ ≥ the rounding error of v = fl(‖c‖²) - 2·fma_dot(x, fl(c)) for one center.

    ‖c‖² rounded to x's dtype: unit·‖c‖²; c rounded: 2·unit·‖x‖‖c‖; the FMA dot over d terms: 2·gamma_d·‖x‖‖c‖ with
    gamma_d ≈ d·unit (Higham, §3.1); the final subtraction (float64; for float64 data at the same unit): unit·(‖c‖² +
    2‖x‖‖c‖). The factor 1.01 absorbs second-order terms (d·unit ≪ 0.01); the tiny term covers underflow
    (tiny = the dtype's smallest subnormal). cn = ‖c‖² in float64.
    """
    return 1.01 * unit * 2.0 * cn + 4.0 * (d + 1) * tiny, 1.01 * unit * 2.0 * (d + 2) * math.sqrt(cn)


SEED_ROWS = 64  # rows per transposed block in the candidate pass; ≤ 32 gets unrolled into scalar code


def _vptr(builder, base, off, vt):
    """Pointer to the vector of type vt at element `off` of `base`."""
    return builder.bitcast(builder.gep(base, [off]), vt.as_pointer())


def _splat(builder, val, vt):
    """Broadcast the scalar `val` to every lane of a vector of type vt."""
    undef = ir.Constant(vt, ir.Undefined)
    lane0 = builder.insert_element(undef, val, ir.Constant(ir.IntType(32), 0))
    return builder.shuffle_vector(lane0, undef, ir.Constant(ir.VectorType(ir.IntType(32), vt.count), [0] * vt.count))


def _fma(builder, vt):
    """llvm.fma for vectors of type vt (correctly rounded on every ISA)."""
    name = f"llvm.fma.v{vt.count}{'f32' if isinstance(vt.element, ir.FloatType) else 'f64'}"
    return cgutils.get_or_insert_function(builder.module, ir.FunctionType(vt, [vt, vt, vt]), name)


def _make_seed_tile(nq, rows):
    """Build an intrinsic scoring nq live candidates (lq[t0:t0+nq]) on `rows` rows (r0...) of block `blk` over columns [j0, j1).

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
        xv = (
            builder.load(_vptr(builder, xa.data, xrow, vt), align=4),
            builder.load(_vptr(builder, xa.data, builder.add(xrow, ir.Constant(intp, lanes)), vt), align=4),
        )
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
    """Compute dv[c0+i, r0+r] = Σ_j xb[j, r0+r]·ct[j, c0+i] for i < 4, r < 2·L with llvm.fma in vector registers.

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
    """Compute `_dot_tile` with the centers read by index from row-major `crow`.

    dv[c0+i, r0+r] = Σ_j xb[j, r0+r]·crow[idx[c0+i], j], 8 lanes (16 rows per call).

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
    """Transpose eight gathered rows into column vectors: xb[j0 + j, r0 + r] = x[queue[s + r], j0 + j] for r, j < 8.

    Eight row vectors go through a 3-stage shuffle network into eight column vectors, instead of storing one scalar
    per element.

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
        for lo, hi, st in (
            ([0, 8, 2, 10, 4, 12, 6, 14], [1, 9, 3, 11, 5, 13, 7, 15], 1),
            ([0, 1, 8, 9, 4, 5, 12, 13], [2, 3, 10, 11, 6, 7, 14, 15], 2),
            ([0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15], 4),
        ):
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


@numba.njit(inline="always", nogil=True, cache=True)
def _scan_queue_dot(x, xnorm, ct, cnorm, err_a, err_b, centers, queue, nq, labels, upper, lower, pfd):
    """Scan the queued rows fully, SEED_ROWS at a time.

    Dot products x·c by FMA register tiles (4 centers x 16 rows,
    `_dot_tile`), then per row, vectorized over rows, the nearest by v = ‖c‖² - 2x·c and the two smallest of
    w_c = v_c - δ_c, where δ_c = err_a[c] + err_b[c]·‖x‖ bounds v_c's rounding error (`_dot_allowance`).
    The label is certain if every other center's w exceeds v1 + δ1; then the bounds come from the dot form with the
    allowances. Otherwise the centers that could be nearest (w_c ≤ v1 + δ1) get exact float64 distances.
    """
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
