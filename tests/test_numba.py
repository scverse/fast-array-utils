# SPDX-License-Identifier: MPL-2.0
# ruff: noqa: SLF001

from __future__ import annotations

import importlib
import subprocess
import threading
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
from fast_array_utils.numba import _parallel_runtime as probe


def _return_true() -> bool:
    return True


def _sum_prange(values: NDArray[np.float64]) -> float:
    total = 0.0
    for i in numba.prange(values.shape[0]):
        total += values[i]
    return total


@pytest.fixture(autouse=True)
def clear_probe_cache() -> None:
    probe._parallel_numba_runtime_layer_cached.cache_clear()


@pytest.mark.parametrize(
    ("on_main", "layer", "expected"),
    [
        pytest.param(False, "workqueue", True, id="worker-unsafe"),
        pytest.param(False, "omp", False, id="worker-threadsafe"),
        pytest.param(False, None, True, id="worker-probe-failed"),
        pytest.param(True, "workqueue", False, id="main-thread"),
    ],
)
def test_is_on_unsafe_thread(monkeypatch: pytest.MonkeyPatch, layer: fa_numba.ThreadingLayer | None, *, on_main: bool, expected: bool) -> None:
    caller = threading.main_thread() if on_main else threading.Thread()

    monkeypatch.setattr(threading, "current_thread", lambda: caller)
    monkeypatch.setattr(probe, "_parallel_numba_runtime_layer", lambda: layer)

    assert fa_numba._is_on_unsafe_thread() is expected


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
    monkeypatch.setattr(probe.sys, "platform", platform_name)
    monkeypatch.setattr(probe.platform, "machine", lambda: machine)
    for module in ("torch", "sklearn", "scanpy"):
        monkeypatch.delitem(probe.sys.modules, module, raising=False)
    for module in loaded:
        monkeypatch.setitem(probe.sys.modules, module, object())
    monkeypatch.setattr(numba.config, "THREADING_LAYER", layer)
    monkeypatch.setattr(numba.config, "THREADING_LAYER_PRIORITY", list(priority))
    if layers is not None:
        monkeypatch.setattr(probe, "LAYERS", layers)


def _install_fake_njit(monkeypatch: pytest.MonkeyPatch, calls: list[bool], *, expected_nogil: bool = True) -> None:
    def fake_njit(_fn: object, /, *, cache: bool, parallel: bool, nogil: bool) -> Callable[..., bool]:
        assert cache is True
        assert nogil is expected_nogil

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
    *,
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    machine: str,
    loaded: tuple[str, ...],
    layer: fa_numba.ThreadingLayer | fa_numba.TheadingCategory,
    priority: tuple[fa_numba.ThreadingLayer, ...],
    layers: dict[fa_numba.TheadingCategory, set[fa_numba.ThreadingLayer]] | None,
    expected: bool,
) -> None:
    _set_runtime(monkeypatch, platform_name=platform_name, machine=machine, loaded=loaded, layer=layer, priority=priority, layers=layers)
    assert probe._needs_parallel_runtime_probe() is expected


def test_probe_check_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_runtime(monkeypatch)

    original_import_module = importlib.import_module

    def import_module(name: str, package: str | None = None) -> object:
        if name.startswith("numba.np.ufunc.") and name.endswith("pool"):
            pytest.fail(f"backend pool module {name!r} should not be imported")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)

    assert probe._needs_parallel_runtime_probe() is True


def test_probe_uses_torch_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_runtime(monkeypatch, loaded=(), layer="threadsafe", priority=("omp", "tbb"))

    first = probe._parallel_runtime_probe_key()
    monkeypatch.setitem(probe.sys.modules, "sklearn", object())
    monkeypatch.setitem(probe.sys.modules, "scanpy", object())
    second = probe._parallel_runtime_probe_key()
    monkeypatch.setitem(probe.sys.modules, "torch", object())
    third = probe._parallel_runtime_probe_key()

    assert probe._loaded_relevant_parallel_runtime_probe_modules() == ("torch",)
    assert first == second
    assert first[3] == ()
    assert third[3] == ("torch",)

    env = probe._build_parallel_runtime_probe_env(third)
    code = probe._parallel_runtime_probe_code(third[3])

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
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{probe._PARALLEL_RUNTIME_PROBE_SENTINEL} tbb\n", stderr="")

    monkeypatch.setattr(probe.subprocess, "run", run)

    assert probe._parallel_numba_runtime_layer() == "tbb"
    assert probe._parallel_numba_runtime_layer() == "tbb"  # cached
    assert calls == [
        (
            [probe.sys.executable, "-c", probe._parallel_runtime_probe_code(("torch",))],
            {
                "capture_output": True,
                "check": False,
                "env": probe._build_parallel_runtime_probe_env(),
                "text": True,
                "timeout": probe._PARALLEL_RUNTIME_PROBE_TIMEOUT,
            },
        )
    ]


def test_probe_reports_launched_layer() -> None:
    """Runs the real probe, which reports what numba launched instead of what we’d predict."""
    assert probe._parallel_numba_runtime_layer() in fa_numba.LAYERS["default"]


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

    monkeypatch.setattr(probe.subprocess, "run", run)

    assert probe._parallel_numba_runtime_layer() is None


@pytest.mark.parametrize(
    ("unsafe_thread", "needs_probe", "probe_safe", "expected", "warning"),
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
    unsafe_thread: bool,
    needs_probe: bool | None,
    probe_safe: bool | None,
    expected: bool,
    warning: str | None,
) -> None:
    calls: list[bool] = []
    _install_fake_njit(monkeypatch, calls)

    monkeypatch.setattr(fa_numba, "_is_on_unsafe_thread", lambda: unsafe_thread)
    if needs_probe is None:
        monkeypatch.setattr(probe, "_needs_parallel_runtime_probe", lambda: pytest.fail("probe should not be consulted"))
    else:
        monkeypatch.setattr(probe, "_needs_parallel_runtime_probe", lambda: needs_probe)
    if probe_safe is None:
        monkeypatch.setattr(probe, "_parallel_numba_runtime_layer", lambda: pytest.fail("probe should not run"))
    else:
        monkeypatch.setattr(probe, "_parallel_numba_runtime_layer", lambda: "tbb" if probe_safe else None)

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


def test_serial_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    values = np.arange(10, dtype=np.float64)
    monkeypatch.setattr(fa_numba, "_is_on_unsafe_thread", lambda: False)
    monkeypatch.setattr(probe, "_needs_parallel_runtime_probe", lambda: True)
    monkeypatch.setattr(probe, "_parallel_numba_runtime_layer", lambda: None)
    wrapped = fa_numba.njit(_sum_prange)

    with pytest.warns(UserWarning, match="unsupported numba parallel runtime"):
        result = wrapped(values)

    assert result == pytest.approx(np.sum(values))


def test_njit_nogil(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_njit(monkeypatch, [], expected_nogil=False)
    fa_numba.njit(nogil=False)(_return_true)
