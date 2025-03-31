# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import types
from testing.fast_array_utils import ArrayType, Flags
from testing.fast_array_utils.pytest import array_type


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import DTypeLike, NDArray


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
    arr = other_array_type(src_arr)
    assert isinstance(arr, other_array_type.cls)
    if isinstance(arr, types.DaskArray):
        arr = arr.compute()
    elif isinstance(arr, types.CupyArray):
        arr = arr.get()
    assert arr.shape == (3, 4)
    assert arr.dtype == np.float32


@pytest.mark.parametrize(
    ("mod", "name"),
    [
        ("scipy.sparse", "coo_matrix"),
        ("scipy.sparse", "coo_array"),
        ("cupyx.scipy.sparse", "coo_matrix"),
    ],
)
@pytest.mark.array_type(skip=Flags.Dask | Flags.Disk | Flags.Gpu)
def test_conv_extra(
    array_type: ArrayType[NDArray[np.number[Any]] | types.CSBase], mod: str, name: str
) -> None:
    src_arr = array_type(np.arange(12).reshape(3, 4), dtype=np.float32)
    arr = ArrayType(mod, name)(src_arr)
    assert type(arr).__module__.startswith(mod)
    assert type(arr).__name__ == name
    assert arr.shape == (3, 4)
    assert arr.dtype == np.float32


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
