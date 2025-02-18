# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fast_array_utils.conv import asarray


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from testing.fast_array_utils import ToArray


def test_asarray(to_array: ToArray[Any]) -> None:
    x = to_array([[1, 2, 3], [4, 5, 6]])
    arr: NDArray[Any] = asarray(x)  # type: ignore[arg-type]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
