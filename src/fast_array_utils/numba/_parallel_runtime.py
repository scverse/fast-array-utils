# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os
import platform
import subprocess
import sys
from functools import cache
from typing import TYPE_CHECKING

import numba

from . import LAYERS


if TYPE_CHECKING:
    from . import TheadingCategory, ThreadingLayer


__all__ = ["_needs_parallel_runtime_probe", "_parallel_numba_runtime_is_safe"]


type _ParallelRuntimeProbeKey = tuple[str, ThreadingLayer | TheadingCategory, tuple[ThreadingLayer, ...], tuple[str, ...]]


_PARALLEL_RUNTIME_PROBE_SENTINEL = "FAST_ARRAY_UTILS_NUMBA_PROBE_OK"
_PARALLEL_RUNTIME_PROBE_TIMEOUT = 20
_PARALLEL_RUNTIME_PROBE_MODULE_WHITELIST = ("torch",)
_PARALLEL_RUNTIME_PROBE_CODE = f"""
import numba
import numpy as np

@numba.njit(parallel=True, cache=False)
def _probe(values):
    total = 0.0
    for i in numba.prange(values.shape[0]):
        total += values[i]
    return total

values = np.arange(32, dtype=np.float64)
assert _probe(values) == np.sum(values)
print({_PARALLEL_RUNTIME_PROBE_SENTINEL!r})
"""


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _needs_parallel_runtime_probe() -> bool:
    if not _is_apple_silicon() or "torch" not in sys.modules:
        return False

    match numba.config.THREADING_LAYER:
        case "omp":
            return True
        case "tbb" | "workqueue":
            return False
        case "default" | "safe" | "threadsafe" | "forksafe" as category:
            return "omp" in LAYERS[category]


def _loaded_relevant_parallel_runtime_probe_modules() -> tuple[str, ...]:
    return tuple(module for module in _PARALLEL_RUNTIME_PROBE_MODULE_WHITELIST if module in sys.modules)


def _parallel_runtime_probe_code(modules: tuple[str, ...]) -> str:
    return "\n".join(f"import {module}" for module in modules) + _PARALLEL_RUNTIME_PROBE_CODE


def _parallel_runtime_probe_key() -> _ParallelRuntimeProbeKey:
    return (
        sys.executable,
        numba.config.THREADING_LAYER,
        tuple(numba.config.THREADING_LAYER_PRIORITY),
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
        # The probe command is built from `sys.executable` plus a generated script
        # that only imports modules from a fixed whitelist.
        result = subprocess.run(  # noqa: S603
            [key[0], "-c", _parallel_runtime_probe_code(key[3])],
            capture_output=True,
            check=False,
            env=_build_parallel_runtime_probe_env(key),
            text=True,
            timeout=_PARALLEL_RUNTIME_PROBE_TIMEOUT,
        )
    except Exception:  # noqa: BLE001
        return False
    return result.returncode == 0 and _PARALLEL_RUNTIME_PROBE_SENTINEL in result.stdout


def _parallel_numba_runtime_is_safe() -> bool:
    return _parallel_numba_runtime_is_safe_cached(_parallel_runtime_probe_key())
