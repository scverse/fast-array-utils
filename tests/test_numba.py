# SPDX-License-Identifier: MPL-2.0
# ruff: noqa: SLF001

from __future__ import annotations

import importlib
import subprocess
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


pytest.importorskip("numba")

import numba

from fast_array_utils import numba as fa_numba


def _return_true() -> bool:
    return True


def _sum_prange(values: NDArray[np.float64]) -> float:
    total = 0.0
    for i in numba.prange(values.shape[0]):
        total += values[i]
    return total


@pytest.fixture(autouse=True)
def clear_probe_cache() -> None:
    fa_numba._parallel_numba_runtime_is_safe_cached.cache_clear()


def _set_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    platform_name: str = "darwin",
    machine: str = "arm64",
    loaded: tuple[str, ...] = ("torch",),
    layer: fa_numba.ThreadingLayer | fa_numba.TheadingCategory = "default",
    priority: tuple[fa_numba.ThreadingLayer, ...] = ("tbb", "omp", "workqueue"),
    layers: dict[fa_numba.TheadingCategory, set[fa_numba.ThreadingLayer]] | None = None,
) -> None:
    monkeypatch.setattr(fa_numba.sys, "platform", platform_name)
    monkeypatch.setattr(fa_numba.platform, "machine", lambda: machine)
    for module in ("torch", "sklearn", "scanpy"):
        monkeypatch.delitem(fa_numba.sys.modules, module, raising=False)
    for module in loaded:
        monkeypatch.setitem(fa_numba.sys.modules, module, object())
    monkeypatch.setattr(numba.config, "THREADING_LAYER", layer)
    monkeypatch.setattr(numba.config, "THREADING_LAYER_PRIORITY", list(priority))
    if layers is not None:
        monkeypatch.setattr(fa_numba, "LAYERS", layers)


def _install_fake_njit(monkeypatch: pytest.MonkeyPatch, calls: list[bool]) -> None:
    def fake_njit(_fn: object, /, *, cache: bool, parallel: bool) -> Callable[..., bool]:
        assert cache is True

        def compiled(*_args: object, **_kwargs: object) -> bool:
            calls.append(parallel)
            return parallel

        return compiled

    monkeypatch.setattr(numba, "njit", fake_njit)


@pytest.mark.parametrize(
    ("platform_name", "machine", "loaded", "layer", "priority", "layers", "expected"),
    [
        pytest.param("darwin", "arm64", ("torch",), "default", ("tbb", "omp", "workqueue"), None, True, id="default"),
        pytest.param("darwin", "arm64", ("torch",), "omp", ("tbb", "omp", "workqueue"), None, True, id="omp"),
        pytest.param("darwin", "arm64", ("torch",), "threadsafe", ("omp", "tbb"), None, True, id="threadsafe"),
        pytest.param("darwin", "arm64", ("torch",), "workqueue", ("tbb", "omp", "workqueue"), None, False, id="workqueue"),
        pytest.param("darwin", "arm64", ("torch",), "safe", ("tbb",), None, False, id="safe"),
        pytest.param(
            "darwin",
            "arm64",
            ("torch",),
            "forksafe",
            ("tbb", "omp", "workqueue"),
            {**fa_numba.LAYERS, "forksafe": {"tbb", "omp", "workqueue"}},
            True,
            id="forksafe",
        ),
        pytest.param("darwin", "arm64", (), "default", ("tbb", "omp", "workqueue"), None, False, id="no-torch"),
        pytest.param("darwin", "x86_64", ("torch",), "default", ("tbb", "omp", "workqueue"), None, False, id="not-arm"),
        pytest.param("linux", "arm64", ("torch",), "default", ("tbb", "omp", "workqueue"), None, False, id="not-darwin"),
    ],
)
def test_probe_needed(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    machine: str,
    loaded: tuple[str, ...],
    layer: fa_numba.ThreadingLayer | fa_numba.TheadingCategory,
    priority: tuple[fa_numba.ThreadingLayer, ...],
    layers: dict[fa_numba.TheadingCategory, set[fa_numba.ThreadingLayer]] | None,
    *,
    expected: bool,
) -> None:
    _set_runtime(monkeypatch, platform_name=platform_name, machine=machine, loaded=loaded, layer=layer, priority=priority, layers=layers)
    assert fa_numba._needs_parallel_runtime_probe() is expected


def test_probe_check_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_runtime(monkeypatch)
    monkeypatch.setattr(fa_numba, "threading_layer", lambda: pytest.fail("threading_layer() should not be called"))

    original_import_module = importlib.import_module

    def import_module(name: str, package: str | None = None) -> object:
        if name.startswith("numba.np.ufunc.") and name.endswith("pool"):
            pytest.fail(f"backend pool module {name!r} should not be imported")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)

    assert fa_numba._needs_parallel_runtime_probe() is True


def test_probe_uses_torch_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_runtime(monkeypatch, loaded=(), layer="threadsafe", priority=("omp", "tbb"))

    first = fa_numba._parallel_runtime_probe_key()
    monkeypatch.setitem(fa_numba.sys.modules, "sklearn", object())
    monkeypatch.setitem(fa_numba.sys.modules, "scanpy", object())
    second = fa_numba._parallel_runtime_probe_key()
    monkeypatch.setitem(fa_numba.sys.modules, "torch", object())
    third = fa_numba._parallel_runtime_probe_key()

    assert fa_numba._loaded_relevant_parallel_runtime_probe_modules() == ("torch",)
    assert first == second
    assert first[3] == ()
    assert third[3] == ("torch",)

    env = fa_numba._build_parallel_runtime_probe_env(third)
    code = fa_numba._parallel_runtime_probe_code(third[3])

    assert env["NUMBA_THREADING_LAYER"] == "threadsafe"
    assert env["NUMBA_THREADING_LAYER_PRIORITY"] == "omp tbb"
    assert "import torch" in code
    assert "import sklearn" not in code
    assert "import scanpy" not in code


def test_probe_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_runtime(monkeypatch, loaded=("torch",))
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(cmd: list[str], /, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{fa_numba._PARALLEL_RUNTIME_PROBE_SENTINEL}\n", stderr="")

    monkeypatch.setattr(fa_numba.subprocess, "run", run)

    assert fa_numba._parallel_numba_runtime_is_safe() is True
    assert fa_numba._parallel_numba_runtime_is_safe() is True
    assert calls == [
        (
            [fa_numba.sys.executable, "-c", fa_numba._parallel_runtime_probe_code(("torch",))],
            {
                "capture_output": True,
                "check": False,
                "env": fa_numba._build_parallel_runtime_probe_env(),
                "text": True,
                "timeout": fa_numba._PARALLEL_RUNTIME_PROBE_TIMEOUT,
            },
        )
    ]


@pytest.mark.parametrize(
    ("result", "error"),
    [
        pytest.param(subprocess.CompletedProcess(["python"], 1, stdout="", stderr="boom"), None, id="nonzero"),
        pytest.param(subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""), None, id="missing-sentinel"),
        pytest.param(None, subprocess.TimeoutExpired(["python"], timeout=1), id="timeout"),
        pytest.param(None, RuntimeError("boom"), id="exception"),
    ],
)
def test_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
    result: subprocess.CompletedProcess[str] | None,
    error: BaseException | None,
) -> None:
    _set_runtime(monkeypatch)

    def run(_cmd: list[str], /, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if error is not None:
            raise error
        assert result is not None
        return result

    monkeypatch.setattr(fa_numba.subprocess, "run", run)

    assert fa_numba._parallel_numba_runtime_is_safe() is False


@pytest.mark.parametrize(
    ("unsafe_pool", "needs_probe", "probe_safe", "expected", "warning"),
    [
        pytest.param(True, None, None, False, "unsupported threading environment", id="thread-pool"),
        pytest.param(False, True, False, False, "unsupported numba parallel runtime", id="probe-fails"),
        pytest.param(False, True, True, True, None, id="probe-passes"),
        pytest.param(False, False, None, True, None, id="no-probe"),
    ],
)
def test_njit_chooses_version(
    monkeypatch: pytest.MonkeyPatch,
    *,
    unsafe_pool: bool,
    needs_probe: bool | None,
    probe_safe: bool | None,
    expected: bool,
    warning: str | None,
) -> None:
    calls: list[bool] = []
    _install_fake_njit(monkeypatch, calls)

    monkeypatch.setattr(fa_numba, "_is_in_unsafe_thread_pool", lambda: unsafe_pool)
    if needs_probe is None:
        monkeypatch.setattr(fa_numba, "_needs_parallel_runtime_probe", lambda: pytest.fail("probe should not be consulted"))
    else:
        monkeypatch.setattr(fa_numba, "_needs_parallel_runtime_probe", lambda: needs_probe)
    if probe_safe is None:
        monkeypatch.setattr(fa_numba, "_parallel_numba_runtime_is_safe", lambda: pytest.fail("probe should not run"))
    else:
        monkeypatch.setattr(fa_numba, "_parallel_numba_runtime_is_safe", lambda: probe_safe)

    wrapped = fa_numba.njit(_return_true)

    if warning is None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert wrapped() is expected
        assert not caught
    else:
        with pytest.warns(UserWarning, match=warning):
            assert wrapped() is expected
    assert calls == [expected]


def test_serial_fallback() -> None:
    values = np.arange(10, dtype=np.float64)
    wrapped = fa_numba.njit(_sum_prange)

    with pytest.MonkeyPatch().context() as monkeypatch:
        monkeypatch.setattr(fa_numba, "_is_in_unsafe_thread_pool", lambda: False)
        monkeypatch.setattr(fa_numba, "_needs_parallel_runtime_probe", lambda: True)
        monkeypatch.setattr(fa_numba, "_parallel_numba_runtime_is_safe", lambda: False)
        with pytest.warns(UserWarning, match="unsupported numba parallel runtime"):
            result = wrapped(values)

    assert result == pytest.approx(np.sum(values))
