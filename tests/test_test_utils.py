# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import types


if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from testing.fast_array_utils import ArrayType


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_conv(array_type: ArrayType, dtype: DTypeLike) -> None:
    arr = array_type(np.arange(12).reshape(3, 4), dtype=dtype)
    assert isinstance(arr, array_type.cls)
    if isinstance(arr, types.DaskArray):
        arr = arr.compute()  # type: ignore[no-untyped-call]
    elif isinstance(arr, types.CupyArray):
        arr = arr.get()
    assert arr.shape == (3, 4)
    assert arr.dtype == dtype
