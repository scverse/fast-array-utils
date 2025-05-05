# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import catch_warnings, filterwarnings

import numpy as np
import pytest

from fast_array_utils import types
from testing.fast_array_utils import Flags
from testing.fast_array_utils.pytest import array_type


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import DTypeLike, NDArray

    from testing.fast_array_utils import Array, ArrayType


other_array_type = array_type


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_conv(array_type: ArrayType, dtype: DTypeLike) -> None:
    arr = array_type(np.arange(12).reshape(3, 4), dtype=dtype)
    assert isinstance(arr, array_type.cls)
    if isinstance(arr, types.DaskArray):
        arr = arr.compute()
    elif isinstance(arr, types.CupyArray):
        arr = arr.get()
    assert arr.shape == (3, 4)
    assert arr.dtype == dtype


def test_conv_other(array_type: ArrayType, other_array_type: ArrayType) -> None:
    src_arr = array_type(np.arange(12).reshape(3, 4), dtype=np.float32)
    with catch_warnings():
        filterwarnings("ignore", r"numba is not installed; falling back to slow conversion", RuntimeWarning)
        arr = other_array_type(src_arr)
    assert isinstance(arr, other_array_type.cls)
    if isinstance(arr, types.DaskArray):
        arr = arr.compute()
    elif isinstance(arr, types.CupyArray):
        arr = arr.get()
    assert arr.shape == (3, 4)
    assert arr.dtype == np.float32


@pytest.mark.array_type(skip=Flags.Dask | Flags.Disk | Flags.Gpu)
def test_conv_extra(
    array_type: ArrayType[NDArray[np.number[Any]] | types.CSBase],
    coo_matrix_type: ArrayType[types.COOBase | types.CupyCOOMatrix],
) -> None:
    src_arr = array_type(np.arange(12).reshape(3, 4), dtype=np.float32)
    arr = coo_matrix_type(src_arr)
    assert type(arr).__module__.startswith(coo_matrix_type.mod)
    assert type(arr).__name__ == coo_matrix_type.name
    assert arr.shape == (3, 4)
    assert arr.dtype == np.float32


def test_array_types(array_type: ArrayType) -> None:
    assert array_type.flags & Flags.Any
    assert array_type.flags & ~Flags(0)
    assert not (array_type.flags & Flags(0))
    is_sparse = "sparse" in str(array_type) or array_type.name in {"CSCDataset", "CSRDataset"}
    assert is_sparse == bool(array_type.flags & Flags.Sparse)
    assert ("cupy" in str(array_type)) == bool(array_type.flags & Flags.Gpu)
    assert ("dask" in str(array_type)) == bool(array_type.flags & Flags.Dask)
    is_disk = any(getattr(t, "mod", None) in {"zarr", "h5py"} for t in (array_type, array_type.inner))
    assert is_disk == bool(array_type.flags & Flags.Disk)


@pytest.fixture(scope="session")
def session_scoped_array(array_type: ArrayType) -> Array:
    return array_type(np.arange(12).reshape(3, 4), dtype=np.float32)


def test_session_scoped_array(session_scoped_array: Array) -> None:
    """Tests that creating a session-scoped array works."""
    assert session_scoped_array.shape == (3, 4)
    assert session_scoped_array.dtype == np.float32
