# SPDX-License-Identifier: MPL-2.0
from typing import Literal

from ._compressed import _compressed_sparse_matrix

class csc_matrix(_compressed_sparse_matrix):
    format: Literal["csc"] = "csc"
