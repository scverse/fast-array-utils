# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest

from fast_array_utils import stats, types
from testing.fast_array_utils import Flags


if TYPE_CHECKING:
    from typing import Any, Protocol, TypeAlias

    from numpy.typing import NDArray
    from pytest_codspeed import BenchmarkFixture

    from testing.fast_array_utils import ArrayType

    DTypeIn = type[np.float32 | np.float64 | np.int32 | np.bool_]
    DTypeOut = type[np.float32 | np.float64 | np.int64]
    Benchmarkable: TypeAlias = NDArray[Any] | types.CSBase

    class BenchFun(Protocol):  # noqa: D101
        def __call__(  # noqa: D102
            self,
            arr: Benchmarkable,
            *,
            axis: Literal[0, 1, None] = None,
            dtype: DTypeOut | None = None,
        ) -> NDArray[Any] | np.number[Any] | types.DaskArray: ...
else:
    DTypeIn = type
    DTypeOut = type


@pytest.fixture(scope="session", params=[0, 1, None])
def axis(request: pytest.FixtureRequest) -> Literal[0, 1, None]:
    return cast(Literal[0, 1, None], request.param)


@pytest.fixture(scope="session", params=[np.float32, np.float64, np.int32, np.bool_])
def dtype_in(request: pytest.FixtureRequest) -> DTypeIn:
    return cast(DTypeIn, request.param)


@pytest.fixture(scope="session", params=[np.float32, np.float64, None])
def dtype_arg(request: pytest.FixtureRequest) -> DTypeOut | None:
    return cast(DTypeOut | None, request.param)


def test_sum(
    array_type: ArrayType,
    dtype_in: DTypeIn,
    dtype_arg: DTypeOut | None,
    axis: Literal[0, 1, None],
) -> None:
    np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype_in)
    arr = array_type(np_arr.copy())
    assert arr.dtype == dtype_in

    sum_: NDArray[Any] | np.number[Any] | types.DaskArray = stats.sum(
        arr, axis=axis, dtype=dtype_arg
    )

    match axis, arr:
        case _, types.DaskArray():
            assert isinstance(sum_, types.DaskArray), type(sum_)
            sum_ = sum_.compute()  # type: ignore[no-untyped-call]
        case None, _:
            assert isinstance(sum_, np.floating | np.integer), type(sum_)
        case 0 | 1, _:
            assert isinstance(sum_, np.ndarray), type(sum_)
        case _:
            pytest.fail(f"Unhandled case axis {axis} for {type(arr)}: {type(sum_)}")

    assert sum_.shape == () if axis is None else arr.shape[axis], (sum_.shape, arr.shape)

    if dtype_arg is not None:
        assert sum_.dtype == dtype_arg, (sum_.dtype, dtype_arg)
    elif dtype_in in {np.bool_, np.int32}:
        assert sum_.dtype == np.int64
    else:
        assert sum_.dtype == dtype_in

    np.testing.assert_array_equal(sum_, np.sum(np_arr, axis=axis, dtype=dtype_arg))


@pytest.mark.benchmark
@pytest.mark.array_type(skip=Flags.Matrix | Flags.Dask | Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("func", [stats.sum])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])  # random only supports float
def test_stats_benchmark(
    benchmark: BenchmarkFixture,
    func: BenchFun,
    array_type: ArrayType[Benchmarkable, None],
    axis: Literal[0, 1, None],
    dtype: type[np.float32 | np.float64],
) -> None:
    shape = (1_000, 1_000) if "sparse" in array_type.mod else (100, 100)
    arr = array_type.random(shape, dtype=dtype)

    func(arr, axis=axis)  # warmup: numba compile
    benchmark(func, arr, axis=axis)
