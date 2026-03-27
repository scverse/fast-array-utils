# SPDX-License-Identifier: MPL-2.0
"""Numba utilities."""

from __future__ import annotations

import sys
import warnings
from functools import cache, update_wrapper, wraps
from types import FunctionType
from typing import TYPE_CHECKING, Literal, cast, overload


if TYPE_CHECKING:
    from collections.abc import Callable

type LayerType = Literal["default", "safe", "threadsafe", "forksafe"]
type Layer = Literal["tbb", "omp", "workqueue"]


LAYERS: dict[LayerType, set[Layer]] = {
    "default": {"tbb", "omp", "workqueue"},
    "safe": {"tbb"},
    "threadsafe": {"tbb", "omp"},
    "forksafe": {"tbb", "workqueue", *(() if sys.platform == "linux" else {"omp"})},
}


@cache
def _numba_threading_layer(layer_name: Layer | LayerType | None = None) -> Layer:
    """Get numba’s threading layer.

    This function implements the algorithm as described in
    <https://numba.readthedocs.io/en/stable/user/threading-layer.html>
    """
    import importlib

    import numba

    if layer_name is None:
        layer_name = numba.config.THREADING_LAYER

    if (available := LAYERS.get(layer_name)) is None:  # type: ignore[arg-type]  # pragma: no cover
        return cast("Layer", layer_name)  # given by direct name

    # given by layer type (safe, …)
    for layer in numba.config.THREADING_LAYER_PRIORITY:
        if layer not in available:
            continue
        if layer != "workqueue":
            try:  # `importlib.util.find_spec` doesn’t work here
                importlib.import_module(f"numba.np.ufunc.{layer}pool")
            except ImportError:
                continue
        # the layer has been found
        return layer
    msg = f"No loadable threading layer: {numba.config.THREADING_LAYER=}  ({available=}, {numba.config.THREADING_LAYER_PRIORITY=})"
    raise ValueError(msg)


def _is_in_unsafe_thread_pool() -> bool:
    import threading

    current_thread = threading.current_thread()
    # ThreadPoolExecutor threads typically have names like 'ThreadPoolExecutor-0_1'
    return current_thread.name.startswith("ThreadPoolExecutor") and _numba_threading_layer() not in LAYERS["threadsafe"]


@overload
def njit[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def njit[**P, R]() -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def njit[**P, R](fn: Callable[P, R] | None = None, /) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Jit-compile a function using numba.

    On call, this function dispatches to a parallel or sequential numba function,
    depending on if it has been called from a thread pool.

    See <https://github.com/numbagg/numbagg/pull/201/files#r1409374809>
    """

    def decorator(f: Callable[P, R], /) -> Callable[P, R]:
        import numba

        assert isinstance(f, FunctionType)

        fns: dict[bool, Callable[P, R]] = {
            parallel: numba.njit(copy_function(f, __qualname__=f"{f.__qualname__}-{parallel}"), cache=True, parallel=parallel) for parallel in (True, False)
        }

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parallel = not _is_in_unsafe_thread_pool()
            if not parallel:  # pragma: no cover
                msg = f"Detected unsupported threading environment. Trying to run {f.__name__} in serial mode. In case of problems, install `tbb`."
                warnings.warn(msg, UserWarning, stacklevel=2)
            return fns[parallel](*args, **kwargs)

        return wrapper

    return decorator if fn is None else decorator(fn)


def copy_function[F: FunctionType](f: F, **overrides: object) -> F:
    new = FunctionType(code=f.__code__, globals=f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    new.__kwdefaults__ = f.__kwdefaults__
    new = cast("F", update_wrapper(new, f))
    for key, value in overrides.items():
        setattr(new, key, value)
    return new
