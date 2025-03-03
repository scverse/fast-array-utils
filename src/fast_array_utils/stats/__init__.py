# SPDX-License-Identifier: MPL-2.0
"""Statistics utilities."""

from __future__ import annotations

from ._is_constant import is_constant
from ._mean import mean
from ._mean_var import mean_var
from ._sum import sum


__all__ = ["is_constant", "mean", "mean_var", "sum"]
