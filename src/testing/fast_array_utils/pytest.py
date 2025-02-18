# SPDX-License-Identifier: MPL-2.0
"""Testing utilities."""

from __future__ import annotations

import os
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from fast_array_utils import types

from . import get_array_cls


if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, TypeVar

    from numpy.typing import ArrayLike, DTypeLike

    from testing.fast_array_utils import ToArray

    from . import Array

    _SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.generic)


def _skip_if_no(dist: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not find_spec(dist), reason=f"{dist} not installed")


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("numpy.ndarray"),
        pytest.param("scipy.sparse.csr_array", marks=_skip_if_no("scipy")),
        pytest.param("scipy.sparse.csc_array", marks=_skip_if_no("scipy")),
        pytest.param("scipy.sparse.csr_matrix", marks=_skip_if_no("scipy")),
        pytest.param("scipy.sparse.csc_matrix", marks=_skip_if_no("scipy")),
        pytest.param("dask.array.Array[numpy.ndarray]", marks=_skip_if_no("dask")),
        pytest.param("dask.array.Array[scipy.sparse.csr_array]", marks=_skip_if_no("dask")),
        pytest.param("dask.array.Array[scipy.sparse.csc_array]", marks=_skip_if_no("dask")),
        pytest.param("dask.array.Array[scipy.sparse.csr_matrix]", marks=_skip_if_no("dask")),
        pytest.param("dask.array.Array[scipy.sparse.csc_matrix]", marks=_skip_if_no("dask")),
        pytest.param("h5py.Dataset", marks=_skip_if_no("h5py")),
        pytest.param("zarr.Array", marks=_skip_if_no("zarr")),
        pytest.param("cupy.ndarray", marks=_skip_if_no("cupy")),
        pytest.param("cupyx.scipy.sparse.csr_matrix", marks=_skip_if_no("cupy")),
        pytest.param("cupyx.scipy.sparse.csc_matrix", marks=_skip_if_no("cupy")),
    ],
)
def array_cls_name(request: pytest.FixtureRequest) -> str:
    """Fixture for a supported array class."""
    return cast(str, request.param)


@pytest.fixture(scope="session")
def array_cls(array_cls_name: str) -> type[Array[Any]]:
    """Fixture for a supported array class."""
    return get_array_cls(array_cls_name)


@pytest.fixture(scope="session")
def to_array(
    request: pytest.FixtureRequest, array_cls: type[Array[_SCT_co]], array_cls_name: str
) -> ToArray[_SCT_co]:
    """Fixture for conversion into a supported array."""
    return get_to_array(array_cls, array_cls_name, request)


def get_to_array(
    array_cls: type[Array[_SCT_co]],
    array_cls_name: str | None = None,
    request: pytest.FixtureRequest | None = None,
) -> ToArray[_SCT_co]:
    """Create a function to convert to a supported array."""
    if array_cls is np.ndarray:
        return np.asarray  # type: ignore[return-value]
    if array_cls is types.DaskArray:
        assert array_cls_name is not None
        return to_dask_array(array_cls_name)
    if array_cls is types.H5Dataset:
        assert request is not None
        return request.getfixturevalue("to_h5py_dataset")  # type: ignore[no-any-return]
    if array_cls is types.ZarrArray:
        return to_zarr_array
    if array_cls is types.CupyArray:
        import cupy as cu

        return cu.asarray  # type: ignore[no-any-return]

    return array_cls  # type: ignore[return-value]


def _half_chunk_size(a: tuple[int, ...]) -> tuple[int, ...]:
    def half_rounded_up(x: int) -> int:
        div, mod = divmod(x, 2)
        return div + (mod > 0)

    return tuple(half_rounded_up(x) for x in a)


def to_dask_array(array_cls_name: str) -> ToArray[Any]:
    """Convert to a dask array."""
    if TYPE_CHECKING:
        import dask.array.core as da
    else:
        import dask.array as da

    inner_cls_name = array_cls_name.removeprefix("dask.array.Array[").removesuffix("]")
    inner_cls = get_array_cls(inner_cls_name)
    to_array_fn: ToArray[Any] = get_to_array(array_cls=inner_cls)

    def to_dask_array(x: ArrayLike, *, dtype: DTypeLike | None = None) -> types.DaskArray:
        x = np.asarray(x, dtype=dtype)
        return da.from_array(to_array_fn(x), _half_chunk_size(x.shape))  # type: ignore[no-untyped-call,no-any-return]

    return to_dask_array


@pytest.fixture(scope="session")
# worker_id for xdist since we don't want to override open files
def to_h5py_dataset(
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str = "serial",
) -> Generator[ToArray[Any], None, None]:
    """Convert to a h5py dataset."""
    import h5py

    tmp_path = tmp_path_factory.mktemp("backed_adata")
    tmp_path = tmp_path / f"test_{worker_id}.h5ad"

    with h5py.File(tmp_path, "x") as f:

        def to_h5py_dataset(x: ArrayLike, *, dtype: DTypeLike | None = None) -> types.H5Dataset:
            arr = np.asarray(x, dtype=dtype)
            test_name = os.environ["PYTEST_CURRENT_TEST"].rsplit(":", 1)[-1].split(" ", 1)[0]
            return f.create_dataset(test_name, arr.shape, arr.dtype, data=arr)

        yield to_h5py_dataset


def to_zarr_array(x: ArrayLike, *, dtype: DTypeLike | None = None) -> types.ZarrArray:
    """Convert to a zarr array."""
    import zarr

    arr = np.asarray(x, dtype=dtype)
    za = zarr.create_array({}, shape=arr.shape, dtype=arr.dtype)
    za[...] = arr
    return za
