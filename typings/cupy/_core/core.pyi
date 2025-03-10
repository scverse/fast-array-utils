# SPDX-License-Identifier: MPL-2.0
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

class ndarray:
    dtype: np.dtype[Any]
    shape: tuple[int, ...]

    # cupy-specific
    def get(self) -> NDArray[Any]: ...

    # operators
    def __power__(self, other: int) -> Self: ...
    def __array__(self) -> NDArray[Any]: ...

    # methods
    def squeeze(self, axis: int | None = None) -> ndarray: ...
    def ravel(self, order: Literal["C", "F", "A", "K"] = "C") -> ndarray: ...
    def flatten(self, order: Literal["C", "F", "A", "K"] = "C") -> ndarray: ...
