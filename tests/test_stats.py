# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import stats, types
from testing.fast_array_utils import random_array


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


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        pytest.param(None, False, id="None"),
        pytest.param(0, [True, True, False, False], id="0"),
        pytest.param(1, [False, False, True, True, False, True], id="1"),
    ],
)
def test_is_constant(
    request: pytest.FixtureRequest,
    to_array: ToArray,
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
    x = to_array(x_data)
    if isinstance(x, types.H5Dataset | types.ZarrArray):
        reason = "H5Dataset and ZarrArray not yet supported for is_constant"
        request.applymarker(pytest.mark.xfail(reason=reason))
    result = stats.is_constant(x, axis=axis)
    if isinstance(result, types.DaskArray):
        result = result.compute()  # type: ignore[no-untyped-call]
    if isinstance(expected, list):
        np.testing.assert_array_equal(expected, result)
    else:
        assert expected is result


@pytest.mark.skipif(not find_spec("dask"), reason="dask not installed")
def test_is_constant_dask() -> None:
    if TYPE_CHECKING:
        import dask.array.core as da
    else:
        import dask.array as da

    x_np = np.repeat(np.repeat(np.arange(4).reshape(2, 2), 2, axis=0), 2, axis=1)
    x: da.Array = da.from_array(x_np, (2, 2))  # type: ignore[no-untyped-call]
    result = stats.is_constant(x, axis=None).compute()  # type: ignore[attr-defined]
    assert result is False


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
