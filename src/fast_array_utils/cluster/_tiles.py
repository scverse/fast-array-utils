# SPDX-License-Identifier: MPL-2.0
"""FMA register tiles, the rounding bound of their dot form, and the full scans of queued rows."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numba
import numpy as np
from llvmlite import ir
from numba.core import cgutils, types
from numba.extending import intrinsic

from ._sync import _prefetch_row


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from llvmlite.ir import Constant, Function, Instruction, IRBuilder, Value
    from numba.core.base import BaseContext
    from numba.core.extending import TypingContext
    from numba.core.typing.templates import Signature

    from ._typing import F64, I64, FArr


BLOCK_ROWS = 64  # rows per transposed block of the seeding pass and of the full scans; ≤ 32 gets unrolled into scalar code
SAFETY = 1e-5  # relative margin on every bound test, so float rounding can never skip a needed scan
ALLOWANCE_SLACK = 1.01  # absorbs the second-order terms of the rounding bound (d·unit ≪ 0.01)
SEED_LANES = 16  # vector lanes of the seeding tiles
DOT_LANES_F32, DOT_LANES_F64 = 16, 8  # vector lanes of `_dot_tile` (512-bit vectors)
IDX_LANES = 8  # vector lanes of `_dot_tile_idx`
TILE_CENTERS, TILE_VECTORS = 4, 2  # centers and row vectors per dot tile
TILE_ROWS_F32, TILE_ROWS_F64 = TILE_VECTORS * DOT_LANES_F32, TILE_VECTORS * DOT_LANES_F64  # rows per `_dot_tile` call
F32_BYTES = 4
ELEM_ALIGN = 4  # alignment of vector loads and stores: a float32 element’s
TRANSPOSE_W = 8  # rows and columns per `_transpose8` call


@numba.njit(inline="always", nogil=True, cache=True)
def _dot_allowance(cn: float, unit: float, tiny: float, d: int) -> tuple[float, float]:
    """Return (a, b) with a + b·‖x‖ ≥ the rounding error of v = fl(‖c‖²) - 2·fma_dot(x, fl(c)) for one center.

    ‖c‖² rounded to x’s dtype: unit·‖c‖²; c rounded: 2·unit·‖x‖‖c‖;
    the FMA dot over d terms: 2·gamma_d·‖x‖‖c‖ with gamma_d ≈ d·unit (Higham, §3.1);
    the final subtraction (float64; for float64 data at the same unit): unit·(‖c‖² + 2‖x‖‖c‖).
    ALLOWANCE_SLACK absorbs second-order terms; the tiny term covers underflow (tiny = the dtype’s smallest subnormal).
    cn = ‖c‖² in float64.
    """
    return ALLOWANCE_SLACK * unit * 2.0 * cn + 4.0 * (d + 1) * tiny, ALLOWANCE_SLACK * unit * 2.0 * (d + 2) * math.sqrt(cn)


# ----------------------------------------------------------------------------- codegen helpers
def _vptr(builder: IRBuilder, base: Value, off: Value, vt: ir.VectorType) -> Instruction:
    """Pointer to the vector of type vt at element `off` of `base`."""
    return builder.bitcast(builder.gep(base, [off]), vt.as_pointer())


def _splat(builder: IRBuilder, val: Value, vt: ir.VectorType) -> Instruction:
    """Broadcast the scalar `val` to every lane of a vector of type vt."""
    undef = ir.Constant(vt, ir.Undefined)
    lane0 = builder.insert_element(undef, val, ir.Constant(ir.IntType(32), 0))
    return builder.shuffle_vector(lane0, undef, ir.Constant(ir.VectorType(ir.IntType(32), vt.count), [0] * vt.count))


def _fma(builder: IRBuilder, vt: ir.VectorType) -> Function:
    """llvm.fma for vectors of type vt (correctly rounded on every ISA)."""
    name = f"llvm.fma.v{vt.count}{'f32' if isinstance(vt.element, ir.FloatType) else 'f64'}"
    return cgutils.get_or_insert_function(builder.module, ir.FunctionType(vt, [vt, vt, vt]), name)


# ----------------------------------------------------------------------------- seeding tiles
def _make_seed_tile(nq: int, rows: int) -> Callable[..., object]:
    """Build an intrinsic scoring nq live candidates (lq[t0:t0+nq]) on `rows` rows (r0...) of block `blk` over columns [j0, j1).

    dv[q, r0+r] += (src[blk, j, r0+r] - cand[q, j])² for each column in order, as llvm.fma(t, t, acc):
    correctly rounded, so every value is the same on any FMA ISA and for any tile shape (each lane runs the same chain).
    Accumulators stay in 16-lane vector registers and each x vector feeds nq candidates (numba keeps array accumulators in L1).
    fma(t, t, acc) ≥ acc, so partial sums still only grow and the early exit stays exact.
    """
    nv = rows // SEED_LANES

    @intrinsic
    def tile(  # noqa: PLR0917  # the typing signature is the call signature
        typingctx: TypingContext,
        src: types.Array,
        blk: types.Integer,
        cand: types.Array,
        lq: types.Array,
        t0: types.Integer,
        dv: types.Array,
        r0: types.Integer,
        j0: types.Integer,
        j1: types.Integer,
    ) -> tuple[Signature, Callable[..., Value]]:
        del typingctx, blk, t0, r0, j0, j1
        sig = types.void(src, types.intp, cand, lq, types.intp, dv, types.intp, types.intp, types.intp)

        def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
            return _seed_tile_codegen(context, builder, signature, args, nq=nq, nv=nv)

        return sig, codegen

    return tile


def _seed_tile_codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value], *, nq: int, nv: int) -> Value:
    """Codegen of a `_make_seed_tile` intrinsic: nq candidates × nv vectors of SEED_LANES rows."""
    lanes = SEED_LANES
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

    def const(v: int) -> Constant:
        return ir.Constant(intp, v)

    def vptr(base: Value, off: Value) -> Instruction:
        return _vptr(builder, base, off, vt)

    qs = [builder.load(builder.gep(la.data, [builder.add(t0_, const(i))])) for i in range(nq)]
    accs = []
    for i in range(nq):
        for h in range(nv):
            accs.append(cgutils.alloca_once_value(builder, undef))
            off = builder.add(builder.mul(qs[i], dr), builder.add(r0_, const(h * lanes)))
            builder.store(builder.load(vptr(da.data, off), align=ELEM_ALIGN), accs[-1])
    base = builder.add(builder.mul(blk_, builder.mul(d, nr)), r0_)
    with cgutils.for_range_slice(builder, j0_, j1_, const(1), intp) as (j, _):
        row = builder.add(base, builder.mul(j, nr))
        xv = [builder.load(vptr(sa.data, builder.add(row, const(h * lanes))), align=ELEM_ALIGN) for h in range(nv)]
        for i in range(nq):
            cval = builder.load(builder.gep(ca.data, [builder.add(builder.mul(qs[i], cd), j)]))
            cv = _splat(builder, cval, vt)
            for h in range(nv):
                t = builder.fsub(xv[h], cv)
                a = accs[i * nv + h]
                builder.store(builder.call(fma, [t, t, builder.load(a)]), a)
    for i in range(nq):
        for h in range(nv):
            off = builder.add(builder.mul(qs[i], dr), builder.add(r0_, const(h * lanes)))
            builder.store(builder.load(accs[i * nv + h]), vptr(da.data, off), align=ELEM_ALIGN)
    return context.get_dummy_value()


_seed_tile4 = _make_seed_tile(4, BLOCK_ROWS // 2)  # the fastest shapes: 4 candidates on half a block, 2 or 1 on a whole block
_seed_tile2 = _make_seed_tile(2, BLOCK_ROWS)
_seed_tile1 = _make_seed_tile(1, BLOCK_ROWS)


@numba.njit(inline="always", nogil=True, cache=True)
def _score(block: tuple[FArr, int], cand: FArr, live: tuple[I64, int], dv: FArr, cols: tuple[int, int]) -> None:
    """Add columns cols = [j0, j1) to the D² of the live candidates (lq[:nl]) for one BLOCK_ROWS block (src, blk)."""
    src, blk = block
    lq, nl = live
    j0, j1 = cols
    t = 0
    while t + 4 <= nl:
        _seed_tile4(src, blk, cand, lq, t, dv, 0, j0, j1)
        _seed_tile4(src, blk, cand, lq, t, dv, BLOCK_ROWS // 2, j0, j1)
        t += 4
    if t + 2 <= nl:
        _seed_tile2(src, blk, cand, lq, t, dv, 0, j0, j1)
        t += 2
    if t < nl:
        _seed_tile1(src, blk, cand, lq, t, dv, 0, j0, j1)


# ----------------------------------------------------------------------------- dot tiles
def _dot_tile_body(
    context: BaseContext,
    builder: IRBuilder,
    *,
    xbty: types.Array,
    xb: Value,
    dvty: types.Array,
    dv: Value,
    r0v: Value,
    c0v: Value,
    lanes: int,
    center: Callable[[Value, int], Value],
) -> Value:
    """Shared codegen of `_dot_tile` and `_dot_tile_idx`; center(j, i) loads center c0+i’s column j."""
    xa = context.make_array(xbty)(context, builder, xb)
    da = context.make_array(dvty)(context, builder, dv)
    intp = context.get_value_type(types.intp)
    elt = context.get_value_type(xbty.dtype)
    vt = ir.VectorType(elt, lanes)
    fma = _fma(builder, vt)
    accs = [cgutils.alloca_once_value(builder, ir.Constant(vt, [ir.Constant(elt, 0.0)] * lanes)) for _ in range(TILE_CENTERS * TILE_VECTORS)]
    d = builder.extract_value(xa.shape, 0)
    nr, dvr = builder.extract_value(xa.shape, 1), builder.extract_value(da.shape, 1)
    with cgutils.for_range(builder, d) as loop:
        j = loop.index
        xrow = builder.add(builder.mul(j, nr), r0v)
        xv = (
            builder.load(_vptr(builder, xa.data, xrow, vt), align=ELEM_ALIGN),
            builder.load(_vptr(builder, xa.data, builder.add(xrow, ir.Constant(intp, lanes)), vt), align=ELEM_ALIGN),
        )
        for i in range(TILE_CENTERS):
            cv = _splat(builder, center(j, i), vt)
            for h in range(TILE_VECTORS):
                builder.store(builder.call(fma, [xv[h], cv, builder.load(accs[TILE_VECTORS * i + h])]), accs[TILE_VECTORS * i + h])
    for i in range(TILE_CENTERS):
        for h in range(TILE_VECTORS):
            off = builder.add(builder.mul(builder.add(c0v, ir.Constant(intp, i)), dvr), builder.add(r0v, ir.Constant(intp, lanes * h)))
            builder.store(builder.load(accs[TILE_VECTORS * i + h]), _vptr(builder, da.data, off, vt), align=ELEM_ALIGN)
    return context.get_dummy_value()


@intrinsic
def _dot_tile(  # noqa: PLR0917  # the typing signature is the call signature
    typingctx: TypingContext, xb: types.Array, ct: types.Array, dv: types.Array, r0: types.Integer, c0: types.Integer
) -> tuple[Signature, Callable[..., Value]]:
    """Compute dv[c0+i, r0+r] = Σ_j xb[j, r0+r]·ct[j, c0+i] for i < 4, r < 2·L with llvm.fma in vector registers.

    L = 16 lanes for float32 (512-bit vectors: 32 rows per call), 8 for float64 (16 rows).

    numba keeps array accumulators in memory (L1 load and store per term); this tile keeps 8 vectors in
    registers: LLVM’s mem2reg/SROA promotes only values that are never in addressable memory, so the
    accumulators are LLVM vector values built with llvmlite’s IRBuilder. FMA is correctly rounded, so the values are
    the same on every FMA ISA. Rows r0..r0+2L-1 and centers c0..c0+3 must exist; arrays C-contiguous.
    """
    del typingctx, r0, c0
    sig = types.void(xb, ct, dv, types.intp, types.intp)

    def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
        xbty, ctty, dvty = signature.args[:3]
        ca = context.make_array(ctty)(context, builder, args[1])
        intp = context.get_value_type(types.intp)
        kc = builder.extract_value(ca.shape, 1)
        lanes = DOT_LANES_F32 if isinstance(context.get_value_type(xbty.dtype), ir.FloatType) else DOT_LANES_F64

        def center(j: Value, i: int) -> Value:
            return builder.load(builder.gep(ca.data, [builder.add(builder.mul(j, kc), builder.add(args[4], ir.Constant(intp, i)))]))

        return _dot_tile_body(context, builder, xbty=xbty, xb=args[0], dvty=dvty, dv=args[2], r0v=args[3], c0v=args[4], lanes=lanes, center=center)

    return sig, codegen


@intrinsic
def _dot_tile_idx(  # noqa: PLR0917  # the typing signature is the call signature
    typingctx: TypingContext, xb: types.Array, crow: types.Array, idx: types.Array, dv: types.Array, r0: types.Integer, c0: types.Integer
) -> tuple[Signature, Callable[..., Value]]:
    """Compute `_dot_tile` with the centers read by index from row-major `crow`.

    dv[c0+i, r0+r] = Σ_j xb[j, r0+r]·crow[idx[c0+i], j], 8 lanes (16 rows per call).

    The ball scans tile a different subset of centers for every label; reading rows by index saves gathering
    them into a transposed copy. Same FMA chain per value as `_dot_tile`, so the same values.
    """
    del typingctx, r0, c0
    sig = types.void(xb, crow, idx, dv, types.intp, types.intp)

    def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
        xbty, crty, idty, dvty = signature.args[:4]
        xa = context.make_array(xbty)(context, builder, args[0])
        ca = context.make_array(crty)(context, builder, args[1])
        ia = context.make_array(idty)(context, builder, args[2])
        intp = context.get_value_type(types.intp)
        d = builder.extract_value(xa.shape, 0)
        rows = [builder.mul(builder.load(builder.gep(ia.data, [builder.add(args[5], ir.Constant(intp, i))])), d) for i in range(TILE_CENTERS)]

        def center(j: Value, i: int) -> Value:
            return builder.load(builder.gep(ca.data, [builder.add(rows[i], j)]))

        return _dot_tile_body(context, builder, xbty=xbty, xb=args[0], dvty=dvty, dv=args[3], r0v=args[4], c0v=args[5], lanes=IDX_LANES, center=center)

    return sig, codegen


@intrinsic
def _transpose8(  # noqa: PLR0917  # the typing signature is the call signature
    typingctx: TypingContext, x: types.Array, queue: types.Array, s: types.Integer, xb: types.Array, r0: types.Integer, j0: types.Integer
) -> tuple[Signature, Callable[..., Value]]:
    """Transpose eight gathered rows into column vectors: xb[j0 + j, r0 + r] = x[queue[s + r], j0 + j] for r, j < 8.

    Eight row vectors go through a 3-stage shuffle network into eight column vectors, instead of storing one scalar
    per element.

    Copies values only, so results don’t change. Rows queue[s..s+7] and columns j0..j0+7 must exist; arrays C-contiguous.
    """
    del typingctx, s, r0, j0
    sig = types.void(x, queue, types.intp, xb, types.intp, types.intp)

    def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
        xty, qty, _, xbty = signature.args[:4]
        xa = context.make_array(xty)(context, builder, args[0])
        qa = context.make_array(qty)(context, builder, args[1])
        ba = context.make_array(xbty)(context, builder, args[3])
        sv, r0v, j0v = args[2], args[4], args[5]
        intp = context.get_value_type(types.intp)
        elt = context.get_value_type(xty.dtype)
        vt = ir.VectorType(elt, TRANSPOSE_W)
        d = builder.extract_value(xa.shape, 1)
        nr = builder.extract_value(ba.shape, 1)

        def vptr(base: Value, off: Value) -> Instruction:
            return _vptr(builder, base, off, vt)

        v = []
        for r in range(TRANSPOSE_W):
            i = builder.load(builder.gep(qa.data, [builder.add(sv, ir.Constant(intp, r))]))
            v.append(builder.load(vptr(xa.data, builder.add(builder.mul(i, d), j0v)), align=ELEM_ALIGN))
        i32 = ir.IntType(32)
        for lo, hi, st in (
            ([0, 8, 2, 10, 4, 12, 6, 14], [1, 9, 3, 11, 5, 13, 7, 15], 1),
            ([0, 1, 8, 9, 4, 5, 12, 13], [2, 3, 10, 11, 6, 7, 14, 15], 2),
            ([0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15], 4),
        ):
            nv = list(v)
            for q in range(TRANSPOSE_W):
                if (q // st) % 2 == 0:
                    a, b = v[q], v[q + st]
                    nv[q] = builder.shuffle_vector(a, b, ir.Constant(ir.VectorType(i32, TRANSPOSE_W), lo))
                    nv[q + st] = builder.shuffle_vector(a, b, ir.Constant(ir.VectorType(i32, TRANSPOSE_W), hi))
            v = nv
        for j in range(TRANSPOSE_W):  # v[j] is column j0 + j of the eight rows
            off = builder.add(builder.mul(builder.add(j0v, ir.Constant(intp, j)), nr), r0v)
            builder.store(v[j], vptr(ba.data, off), align=ELEM_ALIGN)
        return context.get_dummy_value()

    return sig, codegen


# ----------------------------------------------------------------------------- full scans
@numba.njit(inline="always", nogil=True, cache=True)
def _scan_queue_dot(  # noqa: C901, PLR0912, PLR0915, PLR0917
    x: FArr, xnorm: F64, ct: FArr, cnorm: FArr, err_a: F64, err_b: F64, centers: F64, queue: I64, nq: int, labels: I64, upper: F64, lower: F64, pfd: int
) -> int:
    """Scan the queued rows fully, BLOCK_ROWS at a time; returns how many labels changed.

    Per block: gather the rows (8×8 register transposes), dot products x·c by FMA register tiles (4 centers × 16 rows, `_dot_tile`),
    then per row, vectorized over rows, the nearest by v = ‖c‖² - 2x·c and the two smallest of w_c = v_c - δ_c,
    where δ_c = err_a[c] + err_b[c]·‖x‖ bounds v_c’s rounding error (`_dot_allowance`).
    The label is certain if every other center’s w exceeds v1 + δ1; then the bounds come from the dot form with the allowances.
    Otherwise the centers that could be nearest (w_c ≤ v1 + δ1) get exact float64 distances.
    """
    d, k = ct.shape
    xb = np.zeros((d, BLOCK_ROWS), dtype=x.dtype)
    dv = np.empty((k, BLOCK_ROWS), dtype=x.dtype)
    d1, l1, l2, xs = np.empty(BLOCK_ROWS), np.empty(BLOCK_ROWS), np.empty(BLOCK_ROWS), np.zeros(BLOCK_ROWS)
    b1, lb1 = np.empty(BLOCK_ROWS, dtype=np.int64), np.empty(BLOCK_ROWS, dtype=np.int64)
    k4 = k - k % TILE_CENTERS
    moved = 0
    for s0 in range(0, nq, BLOCK_ROWS):
        # gather the rows
        m = min(BLOCK_ROWS, nq - s0)
        d8, m8 = d - d % TRANSPOSE_W, m - m % TRANSPOSE_W
        for r in range(m):
            if pfd and s0 + r + pfd < nq:  # queued rows are scattered in x (a few % of rows at large n)
                _prefetch_row(x, queue[s0 + r + pfd])
            i = queue[s0 + r]
            xs[r] = math.sqrt(xnorm[i])
            for j in range(d8 if r < m8 else 0, d):  # the rest of groups of 8 rows go through 8×8 register transposes
                xb[j, r] = x[i, j]
        for r0 in range(0, m8, TRANSPOSE_W):
            for j0 in range(0, d8, TRANSPOSE_W):
                _transpose8(x, queue, s0 + r0, xb, r0, j0)
        # dot products with every center
        for r0 in range(0, BLOCK_ROWS, TILE_ROWS_F32 if xb.itemsize == F32_BYTES else TILE_ROWS_F64):
            for c0 in range(0, k4, TILE_CENTERS):
                _dot_tile(xb, ct, dv, r0, c0)
        for c in range(k4, k):
            dv[c, :] = 0
            for j in range(d):
                cj = ct[j, c]
                for r in range(BLOCK_ROWS):
                    dv[c, r] += xb[j, r] * cj
        # per row: the nearest center by the dot form, and the two smallest allowance-lowered values
        for r in range(BLOCK_ROWS):
            d1[r], l1[r], l2[r] = np.inf, np.inf, np.inf
            b1[r], lb1[r] = 0, -1
        for c in range(k):  # vectorized over rows (ties: lowest index)
            cn, ea, eb = np.float64(cnorm[c]), err_a[c], err_b[c]
            for r in range(BLOCK_ROWS):
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
        # labels and bounds
        for r in range(m):
            i = queue[s0 + r]
            best, v1 = b1[r], d1[r]
            reach = v1 + err_a[best] + err_b[best] * xs[r]  # ≥ the nearest center’s true v
            wo = l1[r] if lb1[r] != best else l2[r]  # smallest w over the other centers (inf for k = 1)
            if wo > reach:  # certain; bounds from the dot form with the allowances
                e1 = xnorm[i] + reach
                e2 = max(xnorm[i] + wo, 0.0)
            else:  # exact float64 distances to the centers that could be nearest; the others’ w bound the second
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
