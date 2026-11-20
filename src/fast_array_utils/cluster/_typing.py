# SPDX-License-Identifier: MPL-2.0
"""Array aliases of the k-means kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


type F64 = NDArray[np.float64]
type I64 = NDArray[np.int64]
type U8 = NDArray[np.uint8]
type FArr = NDArray[np.floating[Any]]
"""An array in x’s dtype: float32 or float64."""
