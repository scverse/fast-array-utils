# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from fast_array_utils._asarray import asarray


if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import Any, TypeAlias

    from numpy.typing import ArrayLike, NDArray

    from fast_array_utils import types

    SupportedArray: TypeAlias = (
        NDArray[Any]
        | types.DaskArray
        | types.H5Dataset
        | types.ZarrArray
        | types.CupyArray
        | types.CupySparseMatrix
    )


def skip_if_no(dist: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not find_spec(dist), reason=f"{dist} not installed")


@pytest.fixture(scope="session")
# worker_id for xdist since we don't want to override open files
def to_h5py_dataset(
    tmp_path_factory: pytest.TempPathFactory, worker_id: str = "serial"
) -> Generator[Callable[[ArrayLike], types.H5Dataset], None, None]:
    import h5py

    tmp_path = tmp_path_factory.mktemp("backed_adata")
    tmp_path = tmp_path / f"test_{worker_id}.h5ad"

    with h5py.File(tmp_path, "x") as f:

        def to_h5py_dataset(x: ArrayLike) -> types.H5Dataset:
            arr = np.asarray(x)
            return f.create_dataset("data", arr.shape, arr.dtype)

        yield to_h5py_dataset


def to_zarr_array(x: ArrayLike) -> types.ZarrArray:
    import zarr

    arr = np.asarray(x)
    za = zarr.create_array({}, shape=arr.shape, dtype=arr.dtype)
    za[...] = arr
    return za


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
        pytest.param("zarr.Array", marks=skip_if_no("zarr")),
        pytest.param("cupy.ndarray", marks=skip_if_no("cupy")),
        pytest.param("cupyx.scipy.sparse.csr_matrix", marks=skip_if_no("cupy")),
        pytest.param("cupyx.scipy.sparse.csc_matrix", marks=skip_if_no("cupy")),
    ],
)
def array_cls(  # noqa: PLR0911
    request: pytest.FixtureRequest,
) -> Callable[[ArrayLike], SupportedArray]:
    qualname = cast(str, request.param)
    match qualname.split("."):
        case "numpy", "ndarray":
            return np.asarray
        case "scipy", "sparse", ("csr_array" | "csc_array" | "csr_matrix" | "csc_matrix") as n:
            import scipy.sparse

            return getattr(scipy.sparse, n)  # type: ignore[no-any-return]
        case "dask", "array", "Array":
            import dask.array as da

            return da.asarray  # type: ignore[no-any-return]
        case "h5py", "Dataset":
            return request.getfixturevalue("to_h5py_dataset")  # type: ignore[no-any-return]
        case "zarr", "Array":
            return to_zarr_array
        case "cupy", "ndarray":
            import cupy

            return cupy.asarray  # type: ignore[no-any-return]
        case "cupyx", "scipy", "sparse", ("csr_matrix" | "csc_matrix") as n:
            import cupyx.scipy.sparse

            return getattr(cupyx.scipy.sparse, n)  # type: ignore[no-any-return]
        case _:
            msg = f"Unknown array type: {qualname}"
            raise AssertionError(msg)


def test_asarray(array_cls: Callable[[ArrayLike], SupportedArray]) -> None:
    x = array_cls([[1, 2, 3], [4, 5, 6]])
    arr: NDArray[Any] = asarray(x)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
