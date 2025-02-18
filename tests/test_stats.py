# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest

from testing.fast_array_utils import random_array


if TYPE_CHECKING or find_spec("scipy"):
    from scipy.sparse import sparray, spmatrix
else:
    spmatrix = sparray = type("spmatrix", (), {})

from fast_array_utils import stats, types


if TYPE_CHECKING:
    from typing import Any, Literal

    from pytest_codspeed import BenchmarkFixture

    from testing.fast_array_utils import Array, ToArray

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
    array_cls: type[Array],
    to_array: ToArray,
    dtype_in: DTypeIn,
    dtype_arg: DTypeOut | None,
    axis: Literal[0, 1, None],
) -> None:
    np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype_in)
    arr = to_array(np_arr.copy())
    assert arr.dtype == dtype_in

    sum_: Array[Any] | np.floating = stats.sum(arr, axis=axis, dtype=dtype_arg)  # type: ignore[type-arg,arg-type]

    match axis, arr:
        case _, types.DaskArray():
            assert isinstance(sum_, types.DaskArray), type(sum_)
            sum_ = sum_.compute()  # type: ignore[no-untyped-call]
        case None, _:
            assert isinstance(sum_, np.floating | np.integer), type(sum_)
        case 0 | 1, spmatrix() | sparray() | types.ZarrArray() | types.H5Dataset():
            assert isinstance(sum_, np.ndarray), type(sum_)
        case 0 | 1, _:
            assert isinstance(sum_, array_cls), type(sum_)
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
def test_sum_benchmark(
    benchmark: BenchmarkFixture,
    array_cls_name: str,
    axis: Literal[0, 1, None],
    dtype: type[np.float32 | np.float64],
) -> None:
    try:
        shape = (1_000, 1_000) if "sparse" in array_cls_name else (100, 100)
        arr = random_array(array_cls_name, shape, dtype=dtype)
    except NotImplementedError:
        pytest.skip("random_array not implemented for dtype")

    stats.sum(arr, axis=axis)  # type: ignore[arg-type]  # warmup: numba compile
    benchmark(stats.sum, arr, axis=axis)
