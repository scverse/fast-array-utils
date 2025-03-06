# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import types
from testing.fast_array_utils import Flags


if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from testing.fast_array_utils import ArrayType


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


def test_array_types(array_type: ArrayType) -> None:
    assert array_type.flags & Flags.Any
    assert array_type.flags & ~Flags(0)
    assert not (array_type.flags & Flags(0))
    assert ("sparse" in str(array_type) or array_type.name in {"CSCDataset", "CSRDataset"}) == bool(
        array_type.flags & Flags.Sparse
    )
    assert ("cupy" in str(array_type)) == bool(array_type.flags & Flags.Gpu)
    assert ("dask" in str(array_type)) == bool(array_type.flags & Flags.Dask)
    assert any(
        getattr(t, "mod", None) in {"zarr", "h5py"} for t in (array_type, array_type.inner)
    ) == bool(array_type.flags & Flags.Disk)
