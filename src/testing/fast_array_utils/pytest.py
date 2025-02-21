# SPDX-License-Identifier: MPL-2.0
"""Testing utilities."""

from __future__ import annotations

import os
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from fast_array_utils import types

from . import SUPPORTED_TYPES, ArrayType


if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import ArrayLike, DTypeLike

    from testing.fast_array_utils import ToArray


def _skip_if_no(dist: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not find_spec(dist), reason=f"{dist} not installed")


@pytest.fixture(
    scope="session",
    params=[pytest.param(t, marks=_skip_if_no(t.mod.split(".")[0])) for t in SUPPORTED_TYPES],
)
def array_type(request: pytest.FixtureRequest) -> ArrayType:
    """Fixture for a supported array class."""
    return cast(ArrayType, request.param)


@pytest.fixture(scope="session")
def to_array(request: pytest.FixtureRequest, array_type: ArrayType) -> ToArray:
    """Fixture for conversion into a supported array."""
    return get_to_array(array_type, request)


def get_to_array(array_type: ArrayType, request: pytest.FixtureRequest | None = None) -> ToArray:
    """Create a function to convert to a supported array."""
    if array_type.cls is np.ndarray:
        return np.asarray  # type: ignore[return-value]
    if array_type.cls is types.DaskArray:
        assert array_type.inner is not None
        return to_dask_array(array_type.inner)
    if array_type.cls is types.H5Dataset:
        assert request is not None
        return request.getfixturevalue("to_h5py_dataset")  # type: ignore[no-any-return]
    if array_type.cls is types.ZarrArray:
        return to_zarr_array
    if array_type.cls is types.CupyArray:
        import cupy as cu

        return cu.asarray  # type: ignore[no-any-return]

    return array_type.cls  # type: ignore[return-value]


def _half_chunk_size(a: tuple[int, ...]) -> tuple[int, ...]:
    def half_rounded_up(x: int) -> int:
        div, mod = divmod(x, 2)
        return div + (mod > 0)

    return tuple(half_rounded_up(x) for x in a)


def to_dask_array(array_type: ArrayType) -> ToArray:
    """Convert to a dask array."""
    if TYPE_CHECKING:
        import dask.array.core as da
    else:
        import dask.array as da

    inner_cls = array_type.inner
    assert inner_cls is not None
    to_array_fn: ToArray = get_to_array(inner_cls)

    def to_dask_array(x: ArrayLike, *, dtype: DTypeLike | None = None) -> types.DaskArray:
        x = np.asarray(x, dtype=dtype)
        return da.from_array(to_array_fn(x), _half_chunk_size(x.shape))  # type: ignore[no-untyped-call,no-any-return]

    return to_dask_array


@pytest.fixture(scope="session")
# worker_id for xdist since we don't want to override open files
def to_h5py_dataset(
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str = "serial",
) -> Generator[ToArray, None, None]:
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
