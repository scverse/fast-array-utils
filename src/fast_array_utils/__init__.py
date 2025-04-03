# SPDX-License-Identifier: MPL-2.0
"""Fast array utilities with minimal dependencies.

:mod:`fast_array_utils.conv`
    This submodule is always available and contains conversion utilities.

:mod:`fast_array_utils.stats`
    This submodule requires :doc:`numba <numba:index>` to be installed
    and contains statistics utilities.

:mod:`fast_array_utils.typing` and :mod:`fast_array_utils.types`
    These submodules contain types for annotations and checks, respectively.
    Stubs for these types are available even if the respective packages are not installed.
"""

from __future__ import annotations

from . import conv, stats, types
from ._plugins import dask, numba_sparse


__all__ = ["conv", "stats", "types"]

dask.patch()
numba_sparse.register()
