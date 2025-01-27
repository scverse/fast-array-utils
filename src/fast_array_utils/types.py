# SPDX-License-Identifier: MPL-2.0
"""Shared types."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any, TypeAlias, TypeVar

    import numpy as np
    from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

    _SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.generic)

    CSMatrix: TypeAlias = csr_matrix[Any, np.dtype[_SCT_co]] | csc_matrix[Any, np.dtype[_SCT_co]]
    CSBase: TypeAlias = csr_array[Any, np.dtype[_SCT_co]] | csc_array[Any, np.dtype[_SCT_co]]
