# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import stats, types
from testing.fast_array_utils import SUPPORTED_TYPES_MEM
from testing.fast_array_utils.pytest import _skip_if_unimportable


if TYPE_CHECKING:
    from typing import Any, Literal

    from pytest_codspeed import BenchmarkFixture

    from testing.fast_array_utils import Array, ArrayType

    DTypeIn = type[np.float32 | np.float64 | np.int32 | np.bool_]
    DTypeOut = type[np.float32 | np.float64 | np.int64]


@pytest.fixture(scope="session", params=[0, 1, None])
def axis(request: pytest.FixtureRequest) -> Literal[0, 1, None]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session", params=[np.float32, np.float64, np.int32, np.bool_])
def dtype_in(request: pytest.FixtureRequest) -> DTypeIn:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session", params=[np.float32, np.float64, None])
def dtype_arg(request: pytest.FixtureRequest) -> DTypeOut | None:
    return request.param  # type: ignore[no-any-return]


def test_sum(
    array_type: ArrayType,
    dtype_in: DTypeIn,
    dtype_arg: DTypeOut | None,
    axis: Literal[0, 1, None],
) -> None:
    np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype_in)
    arr = array_type(np_arr.copy())
    assert arr.dtype == dtype_in

    sum_: Array[Any] | np.floating = stats.sum(arr, axis=axis, dtype=dtype_arg)  # type: ignore[type-arg,arg-type]

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

    np.testing.assert_array_equal(sum_, np.sum(np_arr, axis=axis, dtype=dtype_arg))  # type: ignore[arg-type]


@pytest.mark.benchmark
@pytest.mark.parametrize("dtype", [np.float32, np.float64])  # random only supports float
@pytest.mark.parametrize(
    "array_type",
    # TODO(flying-sheep): remove need for private import  # noqa: TD003
    [pytest.param(t, id=str(t), marks=_skip_if_unimportable(t)) for t in SUPPORTED_TYPES_MEM],
)
def test_sum_benchmark(
    benchmark: BenchmarkFixture,
    array_type: ArrayType,
    axis: Literal[0, 1, None],
    dtype: type[np.float32 | np.float64],
) -> None:
    shape = (1_000, 1_000) if "sparse" in array_type.mod else (100, 100)
    arr = array_type.random(shape, dtype=dtype)

    stats.sum(arr, axis=axis)  # type: ignore[arg-type]  # warmup: numba compile
    benchmark(stats.sum, arr, axis=axis)
