# SPDX-License-Identifier: MPL-2.0
r"""Numba utilities, mainly used to deal with :ref:`numba-threading-layer` of :doc:`numba <numba:index>`.

``numba.config.THREADING_LAYER`` : env variable :envvar:`NUMBA_THREADING_LAYER`
    This can be set to a :class:`ThreadingLayer` or :class:`TheadingCategory`.
``numba.config.THREADING_LAYER_PRIORITY`` : env variable :envvar:`NUMBA_THREADING_LAYER_PRIORITY`
    This can be set to a list of :class:`ThreadingLayer`\ s.

``fast-array-utils`` provides the following utilities:
"""

from __future__ import annotations

import sys
import warnings
from functools import update_wrapper, wraps
from types import FunctionType
from typing import TYPE_CHECKING, Literal, cast, overload


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["TheadingCategory", "ThreadingLayer", "njit"]


type TheadingCategory = Literal["default", "safe", "threadsafe", "forksafe"]
"""Identifier for a threading layer category."""
type ThreadingLayer = Literal["tbb", "omp", "workqueue"]
"""Identifier for a concrete threading layer."""


LAYERS: dict[TheadingCategory, set[ThreadingLayer]] = {
    "default": {"tbb", "omp", "workqueue"},
    "safe": {"tbb"},
    "threadsafe": {"tbb", "omp"},
    "forksafe": {"tbb", "workqueue", *(() if sys.platform == "linux" else {"omp"})},
}


def _is_on_unsafe_thread() -> bool:
    import threading

    from ._parallel_runtime import _parallel_numba_runtime_layer

    # We deem it unsafe if the caller is not the main thread and the layer numba launches isn’t threadsafe, and therefore fall back to serial.
    return threading.current_thread() is not threading.main_thread() and _parallel_numba_runtime_layer() not in LAYERS["threadsafe"]


@overload
def njit[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def njit[**P, R]() -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def njit[**P, R](fn: Callable[P, R] | None = None, /) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Jit-compile a function using numba.

    On call, this function dispatches to a parallel or serial numba function,
    depending on the current threading environment.
    """
    # See https://github.com/numbagg/numbagg/pull/201/files#r1409374809

    def decorator(f: Callable[P, R], /) -> Callable[P, R]:
        import numba

        from ._parallel_runtime import _needs_parallel_runtime_probe, _parallel_numba_runtime_is_safe

        assert isinstance(f, FunctionType)

        # use distinct names so numba doesn’t reuse the wrong version’s cache
        fns: dict[bool, Callable[P, R]] = {
            parallel: numba.njit(_copy_function(f, __qualname__=f"{f.__qualname__}-{'parallel' if parallel else 'serial'}"), cache=True, parallel=parallel)
            for parallel in (True, False)
        }

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            msg = None
            if _is_on_unsafe_thread():  # pragma: no cover
                msg = f"Detected unsupported threading environment. Trying to run {f.__name__} in serial mode. In case of problems, install `tbb`."
            elif _needs_parallel_runtime_probe() and not _parallel_numba_runtime_is_safe():
                msg = (
                    f"Detected an unsupported numba parallel runtime. Running {f.__name__} in serial mode as a workaround. "
                    "Set `NUMBA_THREADING_LAYER=workqueue` or install `tbb` to avoid this fallback."
                )
            if not (run_parallel := msg is None):
                warnings.warn(msg, UserWarning, stacklevel=2)
            return fns[run_parallel](*args, **kwargs)

        return wrapper

    return decorator if fn is None else decorator(fn)


def _copy_function[F: FunctionType](f: F, **overrides: object) -> F:
    new = FunctionType(code=f.__code__, globals=f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    new.__kwdefaults__ = f.__kwdefaults__
    new = cast("F", update_wrapper(new, f))
    for key, value in overrides.items():
        setattr(new, key, value)
    return new
