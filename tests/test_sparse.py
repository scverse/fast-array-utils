# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from fast_array_utils.conv.scipy import to_dense
from fast_array_utils.types import CSBase
from testing.fast_array_utils.array_type import Flags


if TYPE_CHECKING:
    from typing import Literal

    from pytest_codspeed import BenchmarkFixture

    from testing.fast_array_utils.array_type import ArrayType, _DTypeLikeFloat32, _DTypeLikeFloat64


pytestmark = [pytest.mark.skipif(not find_spec("scipy"), reason="scipy not installed")]


@pytest.fixture(scope="session", params=["csr", "csc"])
def sp_fmt(request: pytest.FixtureRequest) -> Literal["csr", "csc"]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session", params=["array", "matrix"])
def sp_container(request: pytest.FixtureRequest) -> Literal["array", "matrix"]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session", params=[np.float32, np.float64])
def dtype(request: pytest.FixtureRequest) -> type[np.float32 | np.float64]:
    return request.param  # type: ignore[no-any-return]


@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask)
@pytest.mark.parametrize("order", ["C", "F"])
def test_to_dense(
    array_type: ArrayType,
    order: Literal["C", "F"],
    dtype: _DTypeLikeFloat32 | _DTypeLikeFloat64,
) -> None:
    mat = cast(CSBase, array_type.random((10, 10), density=0.1, dtype=dtype))
    arr = to_dense(mat, order=order)
    assert arr.flags[order]
    assert arr.dtype == mat.dtype
    np.testing.assert_equal(arr, mat.toarray(order=order))


@pytest.mark.benchmark
@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask)
@pytest.mark.parametrize("order", ["C", "F"])
def test_to_dense_benchmark(
    benchmark: BenchmarkFixture,
    array_type: ArrayType,
    order: Literal["C", "F"],
    dtype: _DTypeLikeFloat32 | _DTypeLikeFloat64,
) -> None:
    mat = cast(CSBase, array_type.random((1_000, 1_000), dtype=dtype))
    to_dense(mat, order=order)  # warmup: numba compile
    benchmark(to_dense, mat, order=order)
