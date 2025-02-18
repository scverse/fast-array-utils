# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import types


if TYPE_CHECKING:
    from typing import TypeVar

    from testing.fast_array_utils import Array, ToArray

    DType_float = TypeVar("DType_float", np.float32, np.float64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_conv(
    array_cls: type[Array[DType_float]], to_array: ToArray[DType_float], dtype: DType_float
) -> None:
    arr = to_array(np.arange(12).reshape(3, 4), dtype=dtype)
    assert isinstance(arr, array_cls)
    if isinstance(arr, types.DaskArray):
        arr = arr.compute()  # type: ignore[no-untyped-call]
    elif isinstance(arr, types.CupyArray):
        arr = arr.get()
    assert arr.shape == (3, 4)
    assert arr.dtype == dtype
