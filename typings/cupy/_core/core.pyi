# SPDX-License-Identifier: MPL-2.0
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

class ndarray:
    dtype: np.dtype[Any]
    shape: tuple[int, ...]

    def get(self) -> NDArray[Any]: ...
    def __power__(self, other: int) -> Self: ...
    def __array__(self) -> NDArray[Any]: ...
