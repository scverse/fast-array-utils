# SPDX-License-Identifier: MPL-2.0
from typing import Literal, Self

from numpy.typing import DTypeLike

from ._base import spmatrix

class _compressed_sparse_matrix(spmatrix):
    format: Literal["csr", "csc"]

    def power(self, n: int, dtype: DTypeLike | None = None) -> Self: ...
