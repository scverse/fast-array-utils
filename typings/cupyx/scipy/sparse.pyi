# SPDX-License-Identifier: MPL-2.0
from typing import Any, Literal, Self

import cupy
import numpy as np
from numpy.typing import NDArray

class spmatrix:
    dtype: np.dtype[Any]
    shape: tuple[int, int]
    def toarray(self, order: Literal["C", "F", None] = None, out: None = None) -> cupy.ndarray: ...
    def __power__(self, other: int) -> Self: ...
    def __array__(self) -> NDArray[Any]: ...
