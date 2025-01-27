# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from fast_array_utils._asarray import asarray


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from numpy.typing import ArrayLike, NDArray


def skip_if_no(dist: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not find_spec(dist), reason=f"{dist} not installed")


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("numpy.ndarray"),
        pytest.param("scipy.sparse.csr_array", marks=skip_if_no("scipy")),
        pytest.param("scipy.sparse.csc_array", marks=skip_if_no("scipy")),
        pytest.param("scipy.sparse.csr_matrix", marks=skip_if_no("scipy")),
        pytest.param("scipy.sparse.csc_matrix", marks=skip_if_no("scipy")),
        pytest.param("dask.array.Array", marks=skip_if_no("dask")),
        pytest.param("h5py.Dataset", marks=skip_if_no("h5py")),
        pytest.param("cupy.ndarray", marks=skip_if_no("cupy")),
        pytest.param("cupyx.scipy.sparse.csr_matrix", marks=skip_if_no("cupy")),
        pytest.param("cupyx.scipy.sparse.csc_matrix", marks=skip_if_no("cupy")),
    ],
)
def array_cls(request: pytest.FixtureRequest) -> Callable[[ArrayLike], NDArray[Any]]:
    qualname = cast(str, request.param)
    match qualname.split("."):
        case "numpy", "ndarray":
            return np.asarray
        case "scipy", "sparse", ("csr_array" | "csc_array" | "csr_matrix" | "csc_matrix") as n:
            from scipy import sparse

            return getattr(sparse, n)  # type: ignore[no-any-return]
        case "dask", "array", "Array":
            import dask.array as da

            return da.asarray
        case "h5py", "Dataset":
            msg = "TODO: test h5py.Dataset"
            raise NotImplementedError(msg)
        case "cupy", "ndarray":
            import cupy

            return cupy.asarray
        case "cupyx", "scipy", "sparse", ("csr_matrix" | "csc_matrix") as n:
            from cupyx.scipy import sparse

            return getattr(sparse, n)  # type: ignore[no-any-return]
        case _:
            msg = f"Unknown array type: {qualname}"
            raise AssertionError(msg)


def test_asarray(array_cls: Callable[[ArrayLike], Any]) -> None:
    x = array_cls([[1, 2, 3], [4, 5, 6]])
    arr = asarray(x)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
