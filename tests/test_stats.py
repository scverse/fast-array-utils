# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from numpy.exceptions import AxisError

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

    DTypeIn = np.float32 | np.float64 | np.int32 | np.bool
    DTypeOut = np.float32 | np.float64 | np.int64

    NdAndAx: TypeAlias = tuple[Literal[1], Literal[None]] | tuple[Literal[2], Literal[0, 1, None]]

    class StatFun(Protocol):  # noqa: D101
        def __call__(  # noqa: D102
            self,
            arr: Array,
            *,
            axis: Literal[0, 1, None] = None,
            dtype: type[DTypeOut] | None = None,
        ) -> NDArray[Any] | np.number[Any] | types.DaskArray: ...


pytestmark = [pytest.mark.skipif(not find_spec("numba"), reason="numba not installed")]


STAT_FUNCS = [stats.sum, stats.mean, stats.mean_var, stats.is_constant]

# can’t select these using a category filter
ATS_SPARSE_DS = {at for at in SUPPORTED_TYPES if at.mod == "anndata.abc"}
ATS_CUPY_SPARSE = {at for at in SUPPORTED_TYPES if "cupyx.scipy" in str(at)}


@pytest.fixture(
    scope="session",
    params=[
        pytest.param((1, None), id="1d-all"),
        pytest.param((2, None), id="2d-all"),
        pytest.param((2, 0), id="2d-ax0"),
        pytest.param((2, 1), id="2d-ax1"),
    ],
)
def ndim_and_axis(request: pytest.FixtureRequest) -> NdAndAx:
    return cast("NdAndAx", request.param)


@pytest.fixture
def ndim(ndim_and_axis: NdAndAx, array_type: ArrayType) -> Literal[1, 2]:
    return check_ndim(array_type, ndim_and_axis[0])


def check_ndim(array_type: ArrayType, ndim: Literal[1, 2]) -> Literal[1, 2]:
    inner_cls = array_type.inner.cls if array_type.inner else array_type.cls
    if ndim != 2 and issubclass(inner_cls, types.CSMatrix | types.CupyCSMatrix):
        pytest.skip("CSMatrix only supports 2D")
    if ndim != 2 and inner_cls is types.csc_array:
        pytest.skip("csc_array only supports 2D")
    return ndim


@pytest.fixture(scope="session")
def axis(ndim_and_axis: NdAndAx) -> Literal[0, 1, None]:
    return ndim_and_axis[1]


@pytest.fixture(params=[np.float32, np.float64, np.int32, np.bool])
def dtype_in(request: pytest.FixtureRequest, array_type: ArrayType) -> type[DTypeIn]:
    dtype = cast("type[DTypeIn]", request.param)
    inner_cls = array_type.inner.cls if array_type.inner else array_type.cls
    if np.dtype(dtype).kind not in "fdFD" and issubclass(inner_cls, types.CupyCSMatrix):
        pytest.skip("Cupy sparse matrices don’t support non-floating dtypes")
    return dtype


@pytest.fixture(scope="session", params=[np.float32, np.float64, None])
def dtype_arg(request: pytest.FixtureRequest) -> type[DTypeOut] | None:
    return cast("type[DTypeOut] | None", request.param)


@pytest.fixture
def np_arr(dtype_in: type[DTypeIn], ndim: Literal[1, 2]) -> NDArray[DTypeIn]:
    np_arr = cast("NDArray[DTypeIn]", np.array([[1, 0], [3, 0], [5, 6]], dtype=dtype_in))
    np_arr.flags.writeable = False
    if ndim == 1:
        np_arr = np_arr.flatten()
    return np_arr


@pytest.mark.array_type(skip={*ATS_SPARSE_DS, Flags.Matrix})
@pytest.mark.parametrize("func", STAT_FUNCS)
@pytest.mark.parametrize(
    ("ndim", "axis"), [(1, 0), (2, 3), (2, -1)], ids=["1d-ax0", "2d-ax3", "2d-axneg"]
)
def test_ndim_error(
    array_type: ArrayType[Array], func: StatFun, ndim: Literal[1, 2], axis: Literal[0, 1, None]
) -> None:
    check_ndim(array_type, ndim)
    # not using the fixture because we don’t need to test multiple dtypes
    np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    if ndim == 1:
        np_arr = np_arr.flatten()
    arr = array_type(np_arr)

    with pytest.raises(AxisError):
        func(arr, axis=axis)


@pytest.mark.array_type(skip=ATS_SPARSE_DS)
def test_sum(
    array_type: ArrayType[Array],
    dtype_in: type[DTypeIn],
    dtype_arg: type[DTypeOut] | None,
    axis: Literal[0, 1, None],
    np_arr: NDArray[DTypeIn],
) -> None:
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


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([[1, 0], [3, 0], [5, 6]], id="3x2"),
        pytest.param([[1, 2, 3], [4, 5, 6]], id="2x3"),
        pytest.param([[1, 0], [0, 2]], id="2x2"),
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.array_type(Flags.Dask)
def test_sum_dask_shapes(
    array_type: ArrayType[types.DaskArray], axis: Literal[0, 1], data: list[list[int]]
) -> None:
    np_arr = np.array(data, dtype=np.float32)
    arr = array_type(np_arr)
    assert 1 in arr.chunksize, "This test is supposed to test 1×n and n×1 chunk sizes"
    sum_ = cast("NDArray[Any] | types.CupyArray", stats.sum(arr, axis=axis).compute())
    if isinstance(sum_, types.CupyArray):
        sum_ = sum_.get()
    np.testing.assert_almost_equal(np_arr.sum(axis=axis), sum_)


@pytest.mark.array_type(skip=ATS_SPARSE_DS)
def test_mean(
    array_type: ArrayType[Array], axis: Literal[0, 1, None], np_arr: NDArray[DTypeIn]
) -> None:
    arr = array_type(np_arr)

    result = stats.mean(arr, axis=axis)  # type: ignore[arg-type]  # https://github.com/python/mypy/issues/16777
    if isinstance(result, types.DaskArray):
        result = result.compute()
    if isinstance(result, types.CupyArray | types.CupyCSMatrix):
        result = result.get()

    expected = np.mean(np_arr, axis=axis)  # type: ignore[arg-type]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_type(skip=Flags.Disk)
def test_mean_var(
    array_type: ArrayType[CpuArray | GpuArray | types.DaskArray],
    axis: Literal[0, 1, None],
    np_arr: NDArray[DTypeIn],
) -> None:
    arr = array_type(np_arr)

    mean, var = stats.mean_var(arr, axis=axis, correction=1)
    if isinstance(mean, types.DaskArray) and isinstance(var, types.DaskArray):
        mean, var = mean.compute(), var.compute()  # type: ignore[assignment]
    if isinstance(mean, types.CupyArray) and isinstance(var, types.CupyArray):
        mean, var = mean.get(), var.get()

    mean_expected = np.mean(np_arr, axis=axis)  # type: ignore[arg-type]
    var_expected = np.var(np_arr, axis=axis, correction=1)  # type: ignore[arg-type]
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
@pytest.mark.parametrize("func", STAT_FUNCS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_stats_benchmark(
    benchmark: BenchmarkFixture,
    func: StatFun,
    array_type: ArrayType[CpuArray, None],
    axis: Literal[0, 1, None],
    dtype: type[np.float32 | np.float64],
) -> None:
    shape = (10_000, 10_000) if "sparse" in array_type.mod else (1000, 1000)
    arr = array_type.random(shape, dtype=dtype)

    func(arr, axis=axis)  # warmup: numba compile
    benchmark(func, arr, axis=axis)
