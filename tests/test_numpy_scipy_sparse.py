# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from testing.fast_array_utils._array_type import Flags


pytest.importorskip("numba")

import numba


if TYPE_CHECKING:
    from fast_array_utils.types import CSBase
    from testing.fast_array_utils._array_type import ArrayType


@numba.njit(cache=True)
def mat_ndim(mat: CSBase) -> int:
    return mat.ndim


@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask | Flags.Disk | Flags.Gpu)
def test_ndim(array_type: ArrayType[CSBase, None]) -> None:
    mat = array_type.random((10, 10), density=0.1)
    assert mat_ndim(mat) == mat.ndim


@numba.njit(cache=True)
def mat_shape(mat: CSBase) -> tuple[int, ...]:
    return np.shape(mat)  # type: ignore[arg-type]


@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask | Flags.Disk | Flags.Gpu)
def test_shape(array_type: ArrayType[CSBase, None]) -> None:
    mat = array_type.random((10, 10), density=0.1)
    assert mat_shape(mat) == mat.shape


@numba.njit(cache=True)
def copy_mat(mat: CSBase) -> CSBase:
    return mat.copy()


@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask | Flags.Disk | Flags.Gpu)
@pytest.mark.parametrize("dtype_indptr", [np.int32, np.int64], ids=["p=32", "p=64"])
@pytest.mark.parametrize("dtype_index", [np.int32, np.int64], ids=["i=32", "i=64"])
@pytest.mark.parametrize("dtype_data", [np.int64, np.float64], ids=["d=i64", "d=f64"])
def test_copy(
    array_type: ArrayType[CSBase, None],
    dtype_data: type[np.int64 | np.float64],
    dtype_index: type[np.int32 | np.int64],
    dtype_indptr: type[np.int32 | np.int64],
) -> None:
    mat = array_type.random((10, 10), density=0.1, dtype=dtype_data)
    mat.indices = mat.indices.astype(dtype_index)
    mat.indptr = mat.indptr.astype(dtype_indptr)
    copied = copy_mat(mat)
    assert mat.data is not copied.data
    assert mat.indices is not copied.indices
    assert mat.indptr is not copied.indptr
    assert mat.shape == copied.shape
    np.testing.assert_equal(mat.toarray(), copied.toarray(), strict=True)
    np.testing.assert_equal(mat.data, copied.data, strict=True)
    np.testing.assert_equal(mat.indices, copied.indices, strict=True)
    np.testing.assert_equal(mat.indptr, copied.indptr, strict=True)
