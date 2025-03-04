# SPDX-License-Identifier: MPL-2.0
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

class ndarray:
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    def get(self) -> NDArray[Any]: ...
    def __power__(self, other: int) -> Self: ...
    def __array__(self) -> NDArray[Any]: ...

def asarray(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    order: Literal["C", "F", "A", "K", None] = None,
    *,
    blocking: bool = False,
) -> ndarray: ...
