# SPDX-License-Identifier: MPL-2.0
from typing import Literal, Self

import cupy
import numpy as np

class spmatrix:
    dtype: np.dtype[np.generic]
    shape: tuple[int, int]
    def toarray(self, order: Literal["C", "F", None] = None, out: None = None) -> cupy.ndarray: ...
    def __power__(self, other: int) -> Self: ...
