# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest

from fast_array_utils.conv.scipy import to_dense
from testing.fast_array_utils import Flags


if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    from fast_array_utils.types import CSBase
    from testing.fast_array_utils import ArrayType
    from testing.fast_array_utils._array_type import _DTypeLikeFloat32, _DTypeLikeFloat64


pytestmark = [pytest.mark.skipif(not find_spec("scipy"), reason="scipy not installed")]


@pytest.fixture(scope="session", params=["csr", "csc"])
def sp_fmt(request: pytest.FixtureRequest) -> Literal["csr", "csc"]:
    return cast(Literal["csr", "csc"], request.param)


@pytest.fixture(scope="session", params=["array", "matrix"])
def sp_container(request: pytest.FixtureRequest) -> Literal["array", "matrix"]:
    return cast(Literal["array", "matrix"], request.param)


@pytest.fixture(scope="session", params=[np.float32, np.float64])
def dtype(request: pytest.FixtureRequest) -> type[np.float32 | np.float64]:
    return cast(type[np.float32 | np.float64], request.param)


@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask)
@pytest.mark.parametrize("order", ["C", "F"])
def test_to_dense(
    array_type: ArrayType[CSBase, None],
    order: Literal["C", "F"],
    dtype: _DTypeLikeFloat32 | _DTypeLikeFloat64,
) -> None:
    mat = array_type.random((10, 10), density=0.1, dtype=dtype)
    arr = to_dense(mat, order=order)
    assert arr.flags[order]
    assert arr.dtype == mat.dtype
    np.testing.assert_equal(arr, mat.toarray(order=order))


@pytest.mark.benchmark
@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask)
@pytest.mark.parametrize("order", ["C", "F"])
def test_to_dense_benchmark(
    benchmark: BenchmarkFixture,
    array_type: ArrayType[CSBase, None],
    order: Literal["C", "F"],
    dtype: _DTypeLikeFloat32 | _DTypeLikeFloat64,
) -> None:
    mat = array_type.random((1_000, 1_000), dtype=dtype)
    to_dense(mat, order=order)  # warmup: numba compile
    benchmark(to_dense, mat, order=order)
