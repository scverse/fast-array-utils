# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import scipy.sparse as sps
from numpy.exceptions import AxisError
from packaging.version import Version

from fast_array_utils import stats, types
from testing.fast_array_utils import SUPPORTED_TYPES, Flags


DATA_DIR = Path(__file__).parent / "data"


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from numpy.typing import NDArray
    from pytest_codspeed import BenchmarkFixture

    from fast_array_utils.stats._typing import Array, DTypeIn, DTypeOut, NdAndAx, StatFunNoDtype
    from fast_array_utils.typing import CpuArray, DiskArray, GpuArray
    from testing.fast_array_utils import ArrayType


pytestmark = [pytest.mark.skipif(not find_spec("numba"), reason="numba not installed")]


STAT_FUNCS = [stats.sum, stats.min, stats.max, stats.mean, stats.mean_var, stats.is_constant]

# can’t select these using a category filter
ATS_SPARSE_DS = {at for at in SUPPORTED_TYPES if at.mod == "anndata.abc"}
ATS_CUPY_SPARSE = {at for at in SUPPORTED_TYPES if "cupyx.scipy" in str(at)}


def _xfail_if_old_scipy(array_type: ArrayType[Any], ndim: Literal[1, 2]) -> pytest.MarkDecorator:
    cond = ndim == 1 and bool(array_type.flags & Flags.Sparse) and Version(version("scipy")) < Version("1.14")
    return pytest.mark.xfail(cond, reason="Sparse matrices don’t support 1d arrays")


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
def axis(ndim_and_axis: NdAndAx) -> Literal[0, 1] | None:
    return ndim_and_axis[1]


@pytest.fixture(params=[np.float32, np.float64, np.int32, np.bool])
def dtype_in(request: pytest.FixtureRequest, array_type: ArrayType) -> type[DTypeIn]:
    dtype = cast("type[DTypeIn]", request.param)
    inner_cls = array_type.inner.cls if array_type.inner else array_type.cls
    if np.dtype(dtype).kind not in "fdFD" and issubclass(inner_cls, types.CupyCSMatrix):
        pytest.skip("Cupy sparse matrices don’t support non-floating dtypes")
    return dtype


@pytest.fixture(scope="session", params=[np.float32, np.float64, np.int64, None])
def dtype_arg(request: pytest.FixtureRequest) -> type[DTypeOut] | None:
    return cast("type[DTypeOut] | None", request.param)


@pytest.fixture
def np_arr(dtype_in: type[DTypeIn], ndim: Literal[1, 2]) -> NDArray[DTypeIn]:
    np_arr = cast("NDArray[DTypeIn]", np.array([[1, 0], [3, 0], [5, 6]], dtype=dtype_in))
    if np.dtype(dtype_in).kind == "f":
        np_arr /= 4  # type: ignore[misc]
    np_arr.flags.writeable = False
    if ndim == 1:
        np_arr = np_arr.flatten()
    return np_arr


def to_np_dense_checked(
    stat: NDArray[DTypeOut] | np.number[Any] | types.DaskArray, axis: Literal[0, 1] | None, arr: CpuArray | GpuArray | DiskArray | types.DaskArray
) -> NDArray[DTypeOut] | np.number[Any]:
    match axis, arr:
        case _, types.DaskArray():
            assert isinstance(stat, types.DaskArray), type(stat)
            stat = stat.compute()  # type: ignore[assignment]
            return to_np_dense_checked(stat, axis, arr.compute())
        case None, _:
            assert isinstance(stat, np.floating | np.integer), type(stat)
        case 0 | 1, types.CupyArray() | types.CupyCSRMatrix() | types.CupyCSCMatrix() | types.CupyCOOMatrix():
            assert isinstance(stat, types.CupyArray), type(stat)
            return to_np_dense_checked(stat.get(), axis, arr.get())
        case 0 | 1, _:
            assert isinstance(stat, np.ndarray), type(stat)
        case _:
            pytest.fail(f"Unhandled case axis {axis} for {type(arr)}: {type(stat)}")
    return stat


@pytest.fixture(scope="session")
def pbmc64k_reduced_raw() -> sps.csr_array[np.float32]:
    """Scanpy’s pbmc68k_reduced raw data.

    Data was created using:
    >>> if not find_spec("scanpy"):
    ...     pytest.skip()
    >>> import scanpy as sc
    >>> import scipy.sparse as sps
    >>> arr = sps.csr_array(sc.datasets.pbmc68k_reduced().raw.X)
    >>> sps.save_npz("pbmc68k_reduced_raw_csr.npz", arr)
    """
    return cast("sps.csr_array[np.float32]", sps.load_npz(DATA_DIR / "pbmc68k_reduced_raw_csr.npz"))


@pytest.mark.array_type(skip={*ATS_SPARSE_DS, Flags.Matrix})
@pytest.mark.parametrize("func", STAT_FUNCS)
@pytest.mark.parametrize(("ndim", "axis"), [(1, 0), (2, 3), (2, -1)], ids=["1d-ax0", "2d-ax3", "2d-axneg"])
def test_ndim_error(
    request: pytest.FixtureRequest, array_type: ArrayType[Array], func: StatFunNoDtype, ndim: Literal[1, 2], axis: Literal[0, 1] | None
) -> None:
    request.applymarker(_xfail_if_old_scipy(array_type, ndim))
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
    request: pytest.FixtureRequest,
    array_type: ArrayType[CpuArray | GpuArray | DiskArray | types.DaskArray],
    dtype_in: type[DTypeIn],
    dtype_arg: type[DTypeOut] | None,
    axis: Literal[0, 1] | None,
    np_arr: NDArray[DTypeIn],
    ndim: Literal[1, 2],
) -> None:
    request.applymarker(_xfail_if_old_scipy(array_type, ndim))
    if np.dtype(dtype_arg).kind in "iu" and (array_type.flags & Flags.Gpu) and (array_type.flags & Flags.Sparse):
        pytest.skip("GPU sparse matrices don’t support int dtypes")
    arr = array_type(np_arr.copy())
    assert arr.dtype == dtype_in

    sum_ = stats.sum(arr, axis=axis, dtype=dtype_arg)
    sum_ = to_np_dense_checked(sum_, axis, arr)  # type: ignore[arg-type]

    assert sum_.shape == () if axis is None else arr.shape[axis], (sum_.shape, arr.shape)

    if dtype_arg is not None:
        assert sum_.dtype == dtype_arg, (sum_.dtype, dtype_arg)
    elif dtype_in in {np.bool, np.int32}:
        assert sum_.dtype == np.int64
    else:
        assert sum_.dtype == dtype_in

    expected = np.sum(np_arr, axis=axis, dtype=dtype_arg)
    np.testing.assert_array_equal(sum_, expected)


@pytest.mark.array_type(skip={*ATS_SPARSE_DS, Flags.Gpu})
def test_sum_to_int(array_type: ArrayType[CpuArray | DiskArray | types.DaskArray], axis: Literal[0, 1] | None) -> None:
    rng = np.random.default_rng(0)
    np_arr = rng.random((100, 100))
    arr = array_type(np_arr)

    sum_ = stats.sum(arr, axis=axis, dtype=np.int64)
    sum_ = to_np_dense_checked(sum_, axis, arr)

    expected = np.zeros(() if axis is None else arr.shape[axis], dtype=np.int64)
    np.testing.assert_array_equal(sum_, expected)


@pytest.mark.array_type(skip=ATS_SPARSE_DS)
@pytest.mark.parametrize("func", [stats.min, stats.max])
def test_min_max(array_type: ArrayType[CpuArray | GpuArray | DiskArray | types.DaskArray], axis: Literal[0, 1] | None, func: StatFunNoDtype) -> None:
    rng = np.random.default_rng(0)
    np_arr = rng.random((100, 100))
    arr = array_type(np_arr)

    result = to_np_dense_checked(func(arr, axis=axis), axis, arr)

    expected = (np.min if func is stats.min else np.max)(np_arr, axis=axis)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("func", [stats.sum, stats.min, stats.max])
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
def test_dask_shapes(array_type: ArrayType[types.DaskArray], axis: Literal[0, 1], data: list[list[int]], func: StatFunNoDtype) -> None:
    np_arr = np.array(data, dtype=np.float32)
    arr = array_type(np_arr)
    assert 1 in arr.chunksize, "This test is supposed to test 1×n and n×1 chunk sizes"
    stat = cast("NDArray[Any] | types.CupyArray", func(arr, axis=axis).compute())
    if isinstance(stat, types.CupyArray):
        stat = stat.get()
    np_func = getattr(np, func.__name__)
    np.testing.assert_almost_equal(stat, np_func(np_arr, axis=axis))


@pytest.mark.array_type(skip=ATS_SPARSE_DS)
def test_mean(request: pytest.FixtureRequest, array_type: ArrayType[Array], axis: Literal[0, 1] | None, np_arr: NDArray[DTypeIn], ndim: Literal[1, 2]) -> None:
    request.applymarker(_xfail_if_old_scipy(array_type, ndim))
    arr = array_type(np_arr)

    result = stats.mean(arr, axis=axis)  # type: ignore[arg-type]  # https://github.com/python/mypy/issues/16777
    if isinstance(result, types.DaskArray):
        result = result.compute()
    if isinstance(result, types.CupyArray | types.CupyCSMatrix):
        result = result.get()

    expected = np.mean(np_arr, axis=axis)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_type(skip=Flags.Disk)
def test_mean_var(
    request: pytest.FixtureRequest,
    array_type: ArrayType[CpuArray | GpuArray | types.DaskArray],
    axis: Literal[0, 1] | None,
    np_arr: NDArray[DTypeIn],
    ndim: Literal[1, 2],
) -> None:
    request.applymarker(_xfail_if_old_scipy(array_type, ndim))
    arr = array_type(np_arr)

    mean, var = stats.mean_var(arr, axis=axis, correction=1)
    if isinstance(mean, types.DaskArray) and isinstance(var, types.DaskArray):
        mean, var = mean.compute(), var.compute()  # type: ignore[assignment]
    if isinstance(mean, types.CupyArray) and isinstance(var, types.CupyArray):
        mean, var = mean.get(), var.get()

    mean_expected = np.mean(np_arr, axis=axis)
    var_expected = np.var(np_arr, axis=axis, ddof=1)
    np.testing.assert_array_equal(mean, mean_expected)
    np.testing.assert_array_almost_equal(var, var_expected)  # type: ignore[arg-type]


@pytest.mark.skipif(not find_spec("sklearn"), reason="sklearn not installed")
@pytest.mark.array_type(Flags.Sparse, skip=Flags.Matrix | Flags.Dask | Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("axis", [0, 1])
def test_mean_var_sparse_64(array_type: ArrayType[types.CSArray], axis: Literal[0, 1]) -> None:
    """Test that we’re equivalent for 64 bit."""
    from sklearn.utils.sparsefuncs import mean_variance_axis

    mtx = array_type.random((10000, 1000), dtype=np.float64)

    mean_fau, var_fau = stats.mean_var(mtx, axis=axis)
    mean_skl, var_skl = mean_variance_axis(mtx, axis)

    np.testing.assert_allclose(mean_fau, mean_skl, rtol=1.0e-5, atol=1.0e-8)
    np.testing.assert_allclose(var_fau, var_skl, rtol=1.0e-5, atol=1.0e-8)


@pytest.mark.skipif(not find_spec("sklearn"), reason="sklearn not installed")
@pytest.mark.array_type(Flags.Sparse, skip=Flags.Matrix | Flags.Dask | Flags.Disk | Flags.Gpu)
def test_mean_var_sparse_32(array_type: ArrayType[types.CSArray]) -> None:
    """Test whether we are more accurate for 32 bit."""
    from sklearn.utils.sparsefuncs import mean_variance_axis

    mtx64 = array_type.random((10000, 1000), dtype=np.float64)
    mtx32 = mtx64.astype(np.float32)

    fau, skl = {}, {}
    for n_bit, mtx in [(32, mtx32), (64, mtx64)]:
        fau[n_bit] = stats.mean_var(mtx, axis=0)
        skl[n_bit] = mean_variance_axis(mtx, 0)

    for stat, _ in enumerate(["mean", "var"]):
        resid_fau = np.mean(np.abs(fau[64][stat] - fau[32][stat]))
        resid_skl = np.mean(np.abs(skl[64][stat] - skl[32][stat]))
        assert resid_fau < resid_skl


@pytest.mark.array_type({at for at in SUPPORTED_TYPES if at.flags & Flags.Sparse and at.flags & Flags.Dask})
def test_mean_var_pbmc_dask(array_type: ArrayType[types.DaskArray], pbmc64k_reduced_raw: sps.csr_array[np.float32]) -> None:
    """Test float32 precision for bigger data.

    This test is flaky for sparse-in-dask for some reason.
    """
    mat = pbmc64k_reduced_raw
    arr = array_type(mat)

    mean_mat, var_mat = stats.mean_var(mat, axis=0, correction=1)
    mean_arr, var_arr = (to_np_dense_checked(a, 0, arr) for a in stats.mean_var(arr, axis=0, correction=1))

    rtol = 1.0e-5 if array_type.flags & Flags.Gpu else 1.0e-7
    np.testing.assert_allclose(mean_arr, mean_mat, rtol=rtol)
    np.testing.assert_allclose(var_arr, var_mat, rtol=rtol)


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
    *,
    array_type: ArrayType[CpuArray | types.DaskArray],
    axis: Literal[0, 1] | None,
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
def test_dask_constant_blocks(dask_viz: Callable[[object], None], array_type: ArrayType[types.DaskArray, Any]) -> None:
    """Tests if is_constant works if each chunk is individually constant."""
    x_np = np.repeat(np.repeat(np.arange(4, dtype=np.float64).reshape(2, 2), 2, axis=0), 2, axis=1)
    x = array_type(x_np)
    assert x.blocks.shape == (2, 2)
    assert all(stats.is_constant(block).compute() for block in x.blocks.ravel())

    result = stats.is_constant(x, axis=None)
    dask_viz(result)
    assert result.compute() is False  # type: ignore[comparison-overlap]


@pytest.mark.benchmark
@pytest.mark.array_type(skip=Flags.Matrix | Flags.Dask | Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("func", STAT_FUNCS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_stats_benchmark(
    benchmark: BenchmarkFixture,
    func: StatFunNoDtype,
    array_type: ArrayType[CpuArray, None],
    axis: Literal[0, 1] | None,
    dtype: type[np.float32 | np.float64],
) -> None:
    # test with 10M elements will take 20ms for the fastest functions
    n_elems, density = 10_000_000, 0.01
    n = int(np.sqrt(n_elems / density if "sparse" in array_type.mod else n_elems))
    arr = array_type.random((n, n), density=density, dtype=dtype)

    func(arr, axis=axis)  # warmup: numba compile
    benchmark(func, arr, axis=axis)
