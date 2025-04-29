# SPDX-License-Identifier: MPL-2.0
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

def mean_variance_axis(
    X: csc_array | csc_matrix | csr_array | csr_matrix,  # noqa: N803
    axis: Literal[0, 1],
    weights: NDArray[np.floating] | None = None,
    return_sum_weights: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
