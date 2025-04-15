# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from fast_array_utils import stats, types
from testing.fast_array_utils import SUPPORTED_TYPES, Flags


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal, Protocol, TypeAlias

    from numpy.typing import NDArray
    from pytest_codspeed import BenchmarkFixture

    from fast_array_utils.typing import CpuArray, DiskArray, GpuArray
    from testing.fast_array_utils import ArrayType

    Array: TypeAlias = CpuArray | GpuArray | DiskArray | types.CSDataset | types.DaskArray

    DTypeIn = type[np.float32 | np.float64 | np.int32 | np.bool]
    DTypeOut = type[np.float32 | np.float64 | np.int64]

    class BenchFun(Protocol):  # noqa: D101
        def __call__(  # noqa: D102
            self,
            arr: CpuArray,
            *,
            axis: Literal[0, 1, None] = None,
            dtype: DTypeOut | None = None,
        ) -> NDArray[Any] | np.number[Any] | types.DaskArray: ...


pytestmark = [pytest.mark.skipif(not find_spec("numba"), reason="numba not installed")]


# can’t select these using a category filter
ATS_SPARSE_DS = {at for at in SUPPORTED_TYPES if at.mod == "anndata.abc"}
ATS_CUPY_SPARSE = {at for at in SUPPORTED_TYPES if "cupyx.scipy" in str(at)}


@pytest.fixture(scope="session", params=[0, 1, None])
def axis(request: pytest.FixtureRequest) -> Literal[0, 1, None]:
    return cast("Literal[0, 1, None]", request.param)


@pytest.fixture(scope="session", params=[np.float32, np.float64, np.int32, np.bool])
def dtype_in(request: pytest.FixtureRequest) -> DTypeIn:
    return cast("DTypeIn", request.param)


@pytest.fixture(scope="session", params=[np.float32, np.float64, None])
def dtype_arg(request: pytest.FixtureRequest) -> DTypeOut | None:
    return cast("DTypeOut | None", request.param)


@pytest.mark.array_type(skip=ATS_SPARSE_DS)
def test_sum(
    array_type: ArrayType[Array],
    dtype_in: DTypeIn,
    dtype_arg: DTypeOut | None,
    axis: Literal[0, 1, None],
) -> None:
    np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype_in)
    if array_type in ATS_CUPY_SPARSE and np_arr.dtype.kind != "f":
        pytest.skip("CuPy sparse matrices only support floats")
    arr = array_type(np_arr.copy())
    assert arr.dtype == dtype_in

    sum_ = stats.sum(arr, axis=axis, dtype=dtype_arg)

    match axis, arr:
        case _, types.DaskArray():
            assert isinstance(sum_, types.DaskArray), type(sum_)
            sum_ = sum_.compute()  # type: ignore[assignment]
            if isinstance(sum_, types.CupyArray):
                sum_ = sum_.get()
        case None, _:
            assert isinstance(sum_, np.floating | np.integer), type(sum_)
        case 0 | 1, types.CupyArray() | types.CupyCSRMatrix() | types.CupyCSCMatrix():
            assert isinstance(sum_, types.CupyArray), type(sum_)
            sum_ = sum_.get()
        case 0 | 1, _:
            assert isinstance(sum_, np.ndarray), type(sum_)
        case _:
            pytest.fail(f"Unhandled case axis {axis} for {type(arr)}: {type(sum_)}")

    assert sum_.shape == () if axis is None else arr.shape[axis], (sum_.shape, arr.shape)

    if dtype_arg is not None:
        assert sum_.dtype == dtype_arg, (sum_.dtype, dtype_arg)
    elif dtype_in in {np.bool, np.int32}:
        assert sum_.dtype == np.int64
    else:
        assert sum_.dtype == dtype_in

    expected = np.sum(np_arr, axis=axis, dtype=dtype_arg)
    np.testing.assert_array_equal(sum_, expected)


@pytest.mark.array_type(skip=ATS_SPARSE_DS)
@pytest.mark.parametrize(("axis", "expected"), [(None, 3.5), (0, [2.5, 3.5, 4.5]), (1, [2.0, 5.0])])
def test_mean(
    array_type: ArrayType[Array], axis: Literal[0, 1, None], expected: float | list[float]
) -> None:
    np_arr = np.array([[1, 2, 3], [4, 5, 6]])
    if array_type in ATS_CUPY_SPARSE and np_arr.dtype.kind != "f":
        pytest.skip("CuPy sparse matrices only support floats")
    np.testing.assert_array_equal(np.mean(np_arr, axis=axis), expected)

    arr = array_type(np_arr)
    result = stats.mean(arr, axis=axis)  # type: ignore[arg-type]  # https://github.com/python/mypy/issues/16777
    if isinstance(result, types.DaskArray):
        result = result.compute()
    if isinstance(result, types.CupyArray | types.CupyCSMatrix):
        result = result.get()
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_type(skip=Flags.Disk)
@pytest.mark.parametrize(
    ("axis", "mean_expected", "var_expected"),
    [(None, 3.5, 3.5), (0, [2.5, 3.5, 4.5], [4.5, 4.5, 4.5]), (1, [2.0, 5.0], [1.0, 1.0])],
)
def test_mean_var(
    array_type: ArrayType[CpuArray | GpuArray | types.DaskArray],
    axis: Literal[0, 1, None],
    mean_expected: float | list[float],
    var_expected: float | list[float],
) -> None:
    np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    np.testing.assert_array_equal(np.mean(np_arr, axis=axis), mean_expected)
    np.testing.assert_array_equal(np.var(np_arr, axis=axis, correction=1), var_expected)

    arr = array_type(np_arr)
    mean, var = stats.mean_var(arr, axis=axis, correction=1)
    if isinstance(mean, types.DaskArray) and isinstance(var, types.DaskArray):
        mean, var = mean.compute(), var.compute()  # type: ignore[assignment]
    if isinstance(mean, types.CupyArray) and isinstance(var, types.CupyArray):
        mean, var = mean.get(), var.get()
    np.testing.assert_array_equal(mean, mean_expected)
    np.testing.assert_array_almost_equal(var, var_expected)  # type: ignore[arg-type]


@pytest.mark.array_type(skip={Flags.Disk, *ATS_CUPY_SPARSE})
@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        pytest.param(None, False, id="None"),
        pytest.param(0, [True, True, False, False], id="0"),
        pytest.param(1, [False, False, True, True, False, True], id="1"),
    ],
)
def test_is_constant(
    array_type: ArrayType[CpuArray | types.DaskArray],
    axis: Literal[0, 1, None],
    expected: bool | list[bool],
) -> None:
    x_data = [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ]
    x = array_type(x_data, dtype=np.float64)
    result = stats.is_constant(x, axis=axis)
    if isinstance(result, types.DaskArray):
        result = cast("NDArray[np.bool] | bool", result.compute())
    if isinstance(result, types.CupyArray | types.CupyCSMatrix):
        result = result.get()
    if isinstance(expected, list):
        np.testing.assert_array_equal(expected, result)
    else:
        assert expected is result


@pytest.mark.array_type(Flags.Dask, skip=ATS_CUPY_SPARSE)
def test_dask_constant_blocks(
    dask_viz: Callable[[object], None], array_type: ArrayType[types.DaskArray, Any]
) -> None:
    """Tests if is_constant works if each chunk is individually constant."""
    x_np = np.repeat(np.repeat(np.arange(4, dtype=np.float64).reshape(2, 2), 2, axis=0), 2, axis=1)
    x = array_type(x_np)
    assert x.blocks.shape == (2, 2)
    assert all(stats.is_constant(block).compute() for block in x.blocks.ravel())

    result = stats.is_constant(x, axis=None)
    dask_viz(result)
    assert result.compute() is False


@pytest.mark.benchmark
@pytest.mark.array_type(skip=Flags.Matrix | Flags.Dask | Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("func", [stats.sum, stats.mean, stats.mean_var, stats.is_constant])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_stats_benchmark(
    benchmark: BenchmarkFixture,
    func: BenchFun,
    array_type: ArrayType[CpuArray, None],
    axis: Literal[0, 1, None],
    dtype: type[np.float32 | np.float64],
) -> None:
    shape = (10_000, 10_000) if "sparse" in array_type.mod else (1000, 1000)
    arr = array_type.random(shape, dtype=dtype)

    func(arr, axis=axis)  # warmup: numba compile
    benchmark(func, arr, axis=axis)
