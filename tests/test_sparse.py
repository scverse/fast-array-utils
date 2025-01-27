# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils.scipy import to_dense


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal, TypeVar

    from numpy.typing import NDArray

    from fast_array_utils.types import CSBase

    DType = TypeVar("DType", bound=np.generic)
    DType_float = TypeVar("DType_float", np.float32, np.float64)


skip_if_no_scipy = pytest.mark.skipif(not find_spec("scipy"), reason="scipy not installed")


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("csr_array", marks=skip_if_no_scipy),
        pytest.param("csc_array", marks=skip_if_no_scipy),
        pytest.param("csr_matrix", marks=skip_if_no_scipy),
        pytest.param("csc_matrix", marks=skip_if_no_scipy),
    ],
)
def sp_cls(request: pytest.FixtureRequest) -> Callable[[NDArray[DType]], CSBase[DType]]:
    from scipy import sparse

    return getattr(sparse, request.param)  # type: ignore[no-any-return]


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_to_dense(
    order: Literal["C", "F"],
    dtype: np.dtype[DType_float],
    sp_cls: Callable[[NDArray[DType_float]], CSBase[DType_float]],
) -> None:
    rng = np.random.default_rng()
    mat = sp_cls(rng.random((10, 10), dtype=dtype))
    arr = to_dense(mat, order=order)
    assert arr.flags[order]
    assert arr.dtype == mat.dtype
    np.testing.assert_equal(arr, mat.toarray(order=order))
