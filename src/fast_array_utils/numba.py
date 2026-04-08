# SPDX-License-Identifier: MPL-2.0
r"""Numba utilities, mainly used to deal with :ref:`numba-threading-layer` of :doc:`numba <numba:index>`.

``numba.config.THREADING_LAYER`` : env variable :envvar:`NUMBA_THREADING_LAYER`
    This can be set to a :class:`ThreadingLayer` or :class:`TheadingCategory`.
``numba.config.THREADING_LAYER_PRIORITY`` : env variable :envvar:`NUMBA_THREADING_LAYER_PRIORITY`
    This can be set to a list of :class:`ThreadingLayer`\ s.

``fast-array-utils`` provides the following utilities:
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import warnings
from functools import cache, update_wrapper, wraps
from types import FunctionType
from typing import TYPE_CHECKING, Literal, cast, overload


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


__all__ = ["TheadingCategory", "ThreadingLayer", "njit", "threading_layer"]


type TheadingCategory = Literal["default", "safe", "threadsafe", "forksafe"]
"""Identifier for a threading layer category."""
type ThreadingLayer = Literal["tbb", "omp", "workqueue"]
"""Identifier for a concrete threading layer."""
type _ParallelRuntimeProbeKey = tuple[str, ThreadingLayer | TheadingCategory, tuple[ThreadingLayer, ...], tuple[str, ...]]


LAYERS: dict[TheadingCategory, set[ThreadingLayer]] = {
    "default": {"tbb", "omp", "workqueue"},
    "safe": {"tbb"},
    "threadsafe": {"tbb", "omp"},
    "forksafe": {"tbb", "workqueue", *(() if sys.platform == "linux" else {"omp"})},
}


_PARALLEL_RUNTIME_PROBE_SENTINEL = "FAST_ARRAY_UTILS_NUMBA_PROBE_OK"
_PARALLEL_RUNTIME_PROBE_TIMEOUT = 10
_PARALLEL_RUNTIME_PROBE_MODULE_WHITELIST = ("torch",)


def threading_layer(layer_or_category: ThreadingLayer | TheadingCategory | None = None, /, priority: Iterable[ThreadingLayer] | None = None) -> ThreadingLayer:
    """Get numba’s configured threading layer as specified in :ref:`numba-threading-layer`.

    ``layer_or_category`` defaults ``numba.config.THREADING_LAYER`` and ``priority`` to ``numba.config.THREADING_LAYER_PRIORITY``.
    """
    import numba

    if layer_or_category is None:
        layer_or_category = numba.config.THREADING_LAYER
    if priority is None:
        priority = numba.config.THREADING_LAYER_PRIORITY

    return _threading_layer(layer_or_category, tuple(priority))


@cache
def _threading_layer(layer_or_category: ThreadingLayer | TheadingCategory, /, priority: Iterable[ThreadingLayer]) -> ThreadingLayer:
    import importlib

    if (available := LAYERS.get(layer_or_category)) is None:  # type: ignore[arg-type]  # pragma: no cover
        return cast("ThreadingLayer", layer_or_category)  # given by direct name

    # given by layer type (safe, …)
    for layer in priority:
        if layer not in available:  # pragma: no cover
            continue
        if layer != "workqueue":
            try:  # `importlib.util.find_spec` doesn’t work here
                importlib.import_module(f"numba.np.ufunc.{layer}pool")
            except ImportError:
                continue
        # the layer has been found
        return layer
    msg = f"No threading layer matching {layer_or_category!r} ({available=}, {priority=})"  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


def _is_in_unsafe_thread_pool() -> bool:
    import threading

    current_thread = threading.current_thread()
    # ThreadPoolExecutor threads typically have names like 'ThreadPoolExecutor-0_1'
    return current_thread.name.startswith("ThreadPoolExecutor") and threading_layer() not in LAYERS["threadsafe"]


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _is_torch_loaded() -> bool:
    return "torch" in sys.modules


def _configured_threading_layer_or_category_without_probing() -> ThreadingLayer | TheadingCategory:
    import numba

    # Avoid `threading_layer()` here: resolving backends may import pool modules.
    return cast("ThreadingLayer | TheadingCategory", numba.config.THREADING_LAYER)


def _configured_explicit_threading_layer_without_probing() -> ThreadingLayer | None:
    layer_or_category = _configured_threading_layer_or_category_without_probing()
    return layer_or_category if layer_or_category in LAYERS["default"] else None


def _is_explicit_safe_threading_layer() -> bool:
    return _configured_explicit_threading_layer_without_probing() in {"workqueue", "tbb"}


def _could_select_omp_from_threading_config_without_probing() -> bool:
    if (layer := _configured_explicit_threading_layer_without_probing()) is not None:
        return layer == "omp"
    return "omp" in LAYERS[cast("TheadingCategory", _configured_threading_layer_or_category_without_probing())]


def _needs_parallel_runtime_probe() -> bool:
    if not _is_apple_silicon() or not _is_torch_loaded():
        return False
    if _is_explicit_safe_threading_layer():
        return False
    return _could_select_omp_from_threading_config_without_probing()


def _loaded_relevant_parallel_runtime_probe_modules() -> tuple[str, ...]:
    return tuple(module for module in _PARALLEL_RUNTIME_PROBE_MODULE_WHITELIST if module in sys.modules)


def _parallel_runtime_probe_code(modules: tuple[str, ...]) -> str:
    lines = [*(f"import {module}" for module in modules), "import numba", "import numpy as np", ""]
    lines.extend(
        [
            "@numba.njit(parallel=True, cache=False)",
            "def _probe(values):",
            "    total = 0.0",
            "    for i in numba.prange(values.shape[0]):",
            "        total += values[i]",
            "    return total",
            "",
            "values = np.arange(32, dtype=np.float64)",
            "assert _probe(values) == np.sum(values)",
            f"print({_PARALLEL_RUNTIME_PROBE_SENTINEL!r})",
            "",
        ]
    )
    return "\n".join(lines)


def _parallel_runtime_probe_key() -> _ParallelRuntimeProbeKey:
    import numba

    return (
        sys.executable,
        _configured_threading_layer_or_category_without_probing(),
        tuple(cast("Iterable[ThreadingLayer]", numba.config.THREADING_LAYER_PRIORITY)),
        _loaded_relevant_parallel_runtime_probe_modules(),
    )


def _build_parallel_runtime_probe_env(key: _ParallelRuntimeProbeKey | None = None) -> dict[str, str]:
    _, layer_or_category, priority, _ = _parallel_runtime_probe_key() if key is None else key
    env = dict(os.environ)
    env["NUMBA_THREADING_LAYER"] = layer_or_category
    env["NUMBA_THREADING_LAYER_PRIORITY"] = " ".join(priority)
    return env


@cache
def _parallel_numba_runtime_is_safe_cached(key: _ParallelRuntimeProbeKey) -> bool:
    try:
        result = subprocess.run(
            [key[0], "-c", _parallel_runtime_probe_code(key[3])],
            capture_output=True,
            check=False,
            env=_build_parallel_runtime_probe_env(key),
            text=True,
            timeout=_PARALLEL_RUNTIME_PROBE_TIMEOUT,
        )
    except Exception:
        return False
    return result.returncode == 0 and _PARALLEL_RUNTIME_PROBE_SENTINEL in result.stdout


def _parallel_numba_runtime_is_safe() -> bool:
    return _parallel_numba_runtime_is_safe_cached(_parallel_runtime_probe_key())


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

        assert isinstance(f, FunctionType)

        # use distinct names so numba doesn’t reuse the wrong version’s cache
        fns: dict[bool, Callable[P, R]] = {
            parallel: numba.njit(_copy_function(f, __qualname__=f"{f.__qualname__}-{'parallel' if parallel else 'serial'}"), cache=True, parallel=parallel)
            for parallel in (True, False)
        }

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if _is_in_unsafe_thread_pool():  # pragma: no cover
                msg = f"Detected unsupported threading environment. Trying to run {f.__name__} in serial mode. In case of problems, install `tbb`."
                warnings.warn(msg, UserWarning, stacklevel=2)
                return fns[False](*args, **kwargs)
            if _needs_parallel_runtime_probe() and not _parallel_numba_runtime_is_safe():
                msg = (
                    f"Detected an unsupported numba parallel runtime. Running {f.__name__} in serial mode as a workaround. "
                    "Set `NUMBA_THREADING_LAYER=workqueue` or install `tbb` to avoid this fallback."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
                return fns[False](*args, **kwargs)
            return fns[True](*args, **kwargs)

        return wrapper

    return decorator if fn is None else decorator(fn)


def _copy_function[F: FunctionType](f: F, **overrides: object) -> F:
    new = FunctionType(code=f.__code__, globals=f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    new.__kwdefaults__ = f.__kwdefaults__
    new = cast("F", update_wrapper(new, f))
    for key, value in overrides.items():
        setattr(new, key, value)
    return new
