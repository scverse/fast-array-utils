# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils.scipy import to_dense


if TYPE_CHECKING:
    from typing import Literal, TypeVar

    from fast_array_utils.types import CSBase

    DType = TypeVar("DType", bound=np.generic)
    DType_float = TypeVar("DType_float", np.float32, np.float64)


@pytest.fixture(scope="session", params=["csr", "csc"])
def sp_fmt(request: pytest.FixtureRequest) -> Literal["csr", "csc"]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session", params=["matrix", "array"])
def sp_container(request: pytest.FixtureRequest) -> Literal["matrix", "array"]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session", params=[np.float32, np.float64])
def dtype(request: pytest.FixtureRequest) -> np.dtype[np.float32 | np.float64]:
    return np.dtype(request.param)


@pytest.fixture
def sp_mat(
    sp_fmt: Literal["csr", "csc"],
    sp_container: Literal["matrix", "array"],
    dtype: np.dtype[DType_float],
) -> CSBase[DType_float]:
    pytest.importorskip("scipy")
    from scipy.sparse import random, random_array

    return (
        random(10, 10, format=sp_fmt, dtype=dtype)
        if sp_container == "matrix"
        else random_array((10, 10), format=sp_fmt, dtype=dtype)
    )


@pytest.mark.parametrize("order", ["C", "F"])
def test_to_dense(order: Literal["C", "F"], sp_mat: CSBase[np.float64]) -> None:
    arr = to_dense(sp_mat, order=order)
    assert arr.flags[order]
    assert arr.dtype == sp_mat.dtype
    np.testing.assert_equal(arr, sp_mat.toarray(order=order))


@pytest.mark.benchmark
def test_to_dense_benchmark(order: Literal["C", "F"], sp_mat: CSBase[np.float64]) -> None:
    to_dense(sp_mat, order=order)
