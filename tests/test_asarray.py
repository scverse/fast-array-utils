# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fast_array_utils.conv import asarray


if TYPE_CHECKING:
    from testing.fast_array_utils import ArrayType


def test_asarray(array_type: ArrayType) -> None:
    x = array_type([[1, 2, 3], [4, 5, 6]])
    arr = asarray(x)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
