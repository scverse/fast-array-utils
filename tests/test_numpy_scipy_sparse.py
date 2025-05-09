# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import types
from testing.fast_array_utils._array_type import Flags


pytest.importorskip("numba")

import numba


if TYPE_CHECKING:
    from testing.fast_array_utils._array_type import ArrayType


@numba.njit(cache=True)
def mat_ndim(mat: types.CSBase) -> int:
    return mat.ndim


@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask | Flags.Disk | Flags.Gpu)
def test_ndim(array_type: ArrayType[types.CSBase, None]) -> None:
    mat = array_type.random((10, 10), density=0.1)
    assert mat_ndim(mat) == mat.ndim


@numba.njit(cache=True)
def mat_shape(mat: types.CSBase) -> tuple[int, ...]:
    return np.shape(mat)


@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask | Flags.Disk | Flags.Gpu)
def test_shape(array_type: ArrayType[types.CSBase, None]) -> None:
    mat = array_type.random((10, 10), density=0.1)
    assert mat_shape(mat) == mat.shape


@numba.njit(cache=True)
def copy_mat(mat: types.CSBase) -> types.CSBase:
    return mat.copy()


@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask | Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("dtype_ind", [np.int32, np.int64], ids=["i=32", "i=64"])
@pytest.mark.parametrize("dtype_data", [np.int64, np.float64], ids=["d=i64", "d=f64"])
def test_copy(
    array_type: ArrayType[types.CSBase, None],
    dtype_data: type[np.int64 | np.float64],
    dtype_ind: type[np.int32 | np.int64],
) -> None:
    mat = array_type.random((10, 10), density=0.1, dtype=dtype_data)
    mat.indptr = mat.indptr.astype(dtype_ind)
    mat.indices = mat.indices.astype(dtype_ind)

    copied = copy_mat(mat)

    # check that the copied arrays point to different memory locations
    assert mat.data.ctypes.data != copied.data.ctypes.data
    assert mat.indices.ctypes.data != copied.indices.ctypes.data
    assert mat.indptr.ctypes.data != copied.indptr.ctypes.data
    # check that the array contents and dtypes are the same
    assert mat.shape == copied.shape
    np.testing.assert_array_equal(copied.toarray(), mat.toarray(), strict=True)
    np.testing.assert_array_equal(copied.data, mat.data, strict=True)
    np.testing.assert_array_equal(copied.indices, mat.indices, strict=not downcasts_idx(mat))
    np.testing.assert_array_equal(copied.indptr, mat.indptr, strict=not downcasts_idx(mat))


def downcasts_idx(mat: types.CSBase) -> bool:
    """Check if `mat`’s class downcast’s indices to 32-bit.

    See https://github.com/scipy/scipy/pull/18509
    """
    return isinstance(mat, types.CSMatrix)
