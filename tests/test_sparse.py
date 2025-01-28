# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils.scipy import to_dense


if TYPE_CHECKING:
    from typing import Literal, SupportsFloat, TypeVar

    from pytest_codspeed import BenchmarkFixture

    from fast_array_utils.types import CSBase

    DType = TypeVar("DType", bound=np.generic)
    DType_float = TypeVar("DType_float", np.float32, np.float64)


pytestmark = [pytest.mark.skipif(not find_spec("scipy"), reason="scipy not installed")]


@pytest.fixture(scope="session", params=["csr", "csc"])
def sp_fmt(request: pytest.FixtureRequest) -> Literal["csr", "csc"]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session", params=["array", "matrix"])
def sp_container(request: pytest.FixtureRequest) -> Literal["array", "matrix"]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session", params=[np.float32, np.float64])
def dtype(request: pytest.FixtureRequest) -> np.dtype[np.float32 | np.float64]:
    return np.dtype(request.param)


def random_mat(
    shape: tuple[int, int],
    *,
    density: SupportsFloat = 0.01,
    format: Literal["csr", "csc"] = "csr",  # noqa: A002
    dtype: np.dtype[DType_float] | None = None,
    container: Literal["array", "matrix"] = "array",
) -> CSBase[DType_float]:
    from scipy.sparse import random, random_array

    m, n = shape
    return (
        random(m, n, density=density, format=format, dtype=dtype)
        if container == "matrix"
        else random_array(shape, density=density, format=format, dtype=dtype)
    )


@pytest.mark.parametrize("order", ["C", "F"])
def test_to_dense(
    order: Literal["C", "F"],
    sp_fmt: Literal["csr", "csc"],
    dtype: np.dtype[DType_float],
    sp_container: Literal["array", "matrix"],
) -> None:
    mat = random_mat((10, 10), density=0.1, format=sp_fmt, dtype=dtype, container=sp_container)
    arr = to_dense(mat, order=order)
    assert arr.flags[order]
    assert arr.dtype == mat.dtype
    np.testing.assert_equal(arr, mat.toarray(order=order))


@pytest.mark.benchmark
@pytest.mark.parametrize("order", ["C", "F"])
def test_to_dense_benchmark(
    benchmark: BenchmarkFixture,
    order: Literal["C", "F"],
    sp_fmt: Literal["csr", "csc"],
    dtype: np.dtype[DType_float],
) -> None:
    mat = random_mat((1_000, 1_000), format=sp_fmt, dtype=dtype, container="array")
    to_dense(mat, order=order)  # warmup: numba compile
    benchmark(to_dense, mat, order=order)
