# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numba
from llvmlite import ir
from numba.core import cgutils, types
from numba.extending import intrinsic


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from llvmlite.ir import IRBuilder, Value
    from numba.core.base import BaseContext
    from numba.core.extending import TypingContext
    from numba.core.typing.templates import Signature

    from ._typing import I64


LINE_SLOTS = 8  # int64 slots per 64-byte cache line: per-worker counters one line apart, so their atomics don’t contend

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


PREFETCH_ROWS = 16  # rows of x prefetched ahead in gathers by row index...


PREFETCH_MIN_BYTES = 16 << 20  # ...when x is larger than this (smaller x stays in cache; prefetching it only costs)


def _prefetch_distance(x: np.ndarray) -> int:
    return PREFETCH_ROWS if x.nbytes > PREFETCH_MIN_BYTES else 0


@intrinsic
def _prefetch_row(typingctx: TypingContext, arr: types.Array, i: types.Integer) -> tuple[Signature, Callable[..., Value]]:
    """Prefetch the first cache lines of row i of a C-contiguous 2-D array (llvm.prefetch; a hint, no effect on results)."""
    del typingctx, i
    sig = types.void(arr, types.intp)

    def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
        aryty = signature.args[0]
        ary = context.make_array(aryty)(context, builder, args[0])
        ptr = cgutils.get_item_pointer(context, builder, aryty, ary, [args[1], context.get_constant(types.intp, 0)])
        i8p = ir.IntType(8).as_pointer()
        fn = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(ir.VoidType(), [i8p, ir.IntType(32), ir.IntType(32), ir.IntType(32)]), "llvm.prefetch.p0"
        )
        rowbytes = builder.mul(
            builder.extract_value(ary.shape, 1), context.get_constant(types.intp, context.get_abi_sizeof(context.get_data_type(aryty.dtype)))
        )
        base = builder.bitcast(ptr, i8p)
        with cgutils.for_range_slice(
            builder, context.get_constant(types.intp, 0), rowbytes, context.get_constant(types.intp, 64), context.get_value_type(types.intp)
        ) as (off, _):
            builder.call(fn, [builder.gep(base, [off]), ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 3), ir.Constant(ir.IntType(32), 1)])
        return context.get_dummy_value()

    return sig, codegen


@intrinsic
def _fetch_add(typingctx: TypingContext, arr: types.Array, i: types.Integer) -> tuple[Signature, Callable[..., Value]]:
    """Atomically arr[i] += 1, returning the old value (sequentially consistent)."""
    del typingctx, i
    sig = types.int64(arr, types.intp)

    def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
        aryty = signature.args[0]
        ptr = cgutils.get_item_pointer(context, builder, aryty, context.make_array(aryty)(context, builder, args[0]), [args[1]])
        return builder.atomic_rmw("add", ptr, context.get_constant(types.int64, 1), "seq_cst")

    return sig, codegen


@intrinsic
def _load_acquire(typingctx: TypingContext, arr: types.Array, i: types.Integer) -> tuple[Signature, Callable[..., Value]]:
    """arr[i], loaded with acquire ordering: writes published before the matching release store are visible after it."""
    del typingctx, i
    sig = types.int64(arr, types.intp)

    def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
        aryty = signature.args[0]
        ptr = cgutils.get_item_pointer(context, builder, aryty, context.make_array(aryty)(context, builder, args[0]), [args[1]])
        return builder.load_atomic(ptr, "acquire", 8)

    return sig, codegen


@intrinsic
def _store_release(typingctx: TypingContext, arr: types.Array, i: types.Integer, v: types.Integer) -> tuple[Signature, Callable[..., Value]]:
    """arr[i] = v with release ordering: publishes every earlier write to threads that acquire-load arr[i]."""
    del typingctx, i, v
    sig = types.void(arr, types.intp, types.int64)

    def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
        aryty = signature.args[0]
        ptr = cgutils.get_item_pointer(context, builder, aryty, context.make_array(aryty)(context, builder, args[0]), [args[1]])
        builder.store_atomic(args[2], ptr, "release", 8)
        return context.get_dummy_value()

    return sig, codegen


_YIELD_SYMBOL = "SwitchToThread" if sys.platform == "win32" else "sched_yield"


SPIN_BEFORE_YIELD = 64  # spin-wait iterations before a waiting worker yields its CPU


@intrinsic
def _yield_cpu(typingctx: TypingContext) -> tuple[Signature, Callable[..., Value]]:
    """Give the CPU to another runnable thread (sched_yield; SwitchToThread on Windows), by symbol so kernels stay cacheable.

    Spinning workers must yield: with as many threads as CPUs plus a waiting main thread (numba's workqueue layer),
    a pure spin starves the thread doing the work for whole scheduler slices.
    """
    del typingctx

    def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
        del signature, args
        fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.IntType(32), []), _YIELD_SYMBOL)
        builder.call(fn, [])
        return context.get_dummy_value()

    return types.void(), codegen


# monotonic clock by symbol (cacheable): clock_gettime(CLOCK_MONOTONIC) on Linux (id 1) and macOS (id 6), else QPC ticks
_CLOCK_SYMBOL, _CLOCK_ID = ("QueryPerformanceCounter", 0) if sys.platform == "win32" else ("clock_gettime", 6 if sys.platform == "darwin" else 1)


@intrinsic
def _clock_ticks(typingctx: TypingContext) -> tuple[Signature, Callable[..., Value]]:
    """Read a monotonic clock (ns; QueryPerformanceCounter ticks on Windows); only differences of readings are used.

    The ball scans are chosen per pass by measured cost; results don't depend on the choice.
    """
    del typingctx

    def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
        del context, signature, args
        i64 = ir.IntType(64)
        if _CLOCK_SYMBOL == "QueryPerformanceCounter":  # pragma: no cover (Windows)
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
def _cas(typingctx: TypingContext, arr: types.Array, i: types.Integer, expected: types.Integer, new: types.Integer) -> tuple[Signature, Callable[..., Value]]:
    """Atomically arr[i] = new if arr[i] == expected; returns the old value (sequentially consistent)."""
    del typingctx, i, expected, new
    sig = types.int64(arr, types.intp, types.int64, types.int64)

    def codegen(context: BaseContext, builder: IRBuilder, signature: Signature, args: Sequence[Value]) -> Value:
        aryty = signature.args[0]
        ptr = cgutils.get_item_pointer(context, builder, aryty, context.make_array(aryty)(context, builder, args[0]), [args[1]])
        return builder.extract_value(builder.cmpxchg(ptr, args[2], args[3], "seq_cst", "seq_cst"), 0)

    return sig, codegen


@numba.njit(inline="always", nogil=True, cache=True)
def _claim_range(rcnt: I64, v: int, st: int, nb: int, nr: int) -> int:
    """Claim the next chunk of step st from range v of nr (chunks v·nb/nr ...), or -1 if all are claimed.

    rcnt[LINE_SLOTS·v] counts the range's claims over all steps; a claim for step st succeeds only while it's below (st + 1)·len,
    so a worker still looking at step st can never take a chunk of step st + 1.
    """
    lo, hi = (v * nb) // nr, ((v + 1) * nb) // nr
    n = hi - lo
    if n == 0:
        return -1
    c: int = _load_acquire(rcnt, LINE_SLOTS * v)
    lim = (st + 1) * n
    while c < lim:
        old: int = _cas(rcnt, LINE_SLOTS * v, c, c + 1)
        if old == c:
            return lo + c - st * n
        c = old
    return -1
