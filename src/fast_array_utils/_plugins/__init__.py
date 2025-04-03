# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations


__all__ = ["patch_dask", "register_numba_sparse"]


def patch_dask() -> None:
    try:
        from .dask import patch
    except ImportError:
        pass
    else:
        patch()


def register_numba_sparse() -> None:
    try:
        from .numba_sparse import register
    except ImportError:
        pass
    else:
        register()
