# SPDX-License-Identifier: MPL-2.0
"""Fast array utilities with minimal dependencies.

:mod:`fast_array_utils.conv`
    This submodule is always available and contains conversion utilities.

:mod:`fast_array_utils.stats`
    This submodule requires :doc:`numba <numba:index>` to be installed
    and contains statistics utilities.

:mod:`fast_array_utils.cluster`
    This submodule requires :doc:`numba <numba:index>` to be installed
    and contains an exact, deterministic k-means.

:mod:`fast_array_utils.numba`
    This submodule contains numba utilities.

:mod:`fast_array_utils.typing` and :mod:`fast_array_utils.types`
    These submodules contain types for annotations and checks, respectively.
    Stubs for these types are available even if the respective packages are not installed.
"""

from __future__ import annotations

from . import _plugins, cluster, conv, numba, stats, types


__all__ = ["cluster", "conv", "numba", "stats", "types"]

_plugins.patch_dask()
_plugins.register_numba_sparse()
