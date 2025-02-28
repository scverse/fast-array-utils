# SPDX-License-Identifier: MPL-2.0
from typing import Any, Literal

import cupy
import numpy as np

class spmatrix:
    dtype: np.dtype[Any]
    shape: tuple[int, int]
    def toarray(self, order: Literal["C", "F", None] = None, out: None = None) -> cupy.ndarray: ...
