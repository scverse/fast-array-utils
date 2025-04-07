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
def copy_mat(mat: CSBase) -> CSBase:
    return mat.copy()


@pytest.mark.array_type(select=Flags.Sparse, skip=Flags.Dask | Flags.Disk | Flags.Gpu)
def test_copy(array_type: ArrayType[CSBase, None]) -> None:
    mat = array_type.random((10, 10), density=0.1)
    copied = copy_mat(mat)
    assert mat.data is not copied.data
    assert mat.indices is not copied.indices
    assert mat.indptr is not copied.indptr
    assert mat.shape == copied.shape
    np.testing.assert_equal(mat.toarray(), copied.toarray())
    np.testing.assert_equal(mat.data, copied.data)
    np.testing.assert_equal(mat.indices, copied.indices)
    np.testing.assert_equal(mat.indptr, copied.indptr)
