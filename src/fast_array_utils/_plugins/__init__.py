# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations


__all__ = ["patch_dask", "register_numba_sparse"]


def patch_dask() -> None:
    r"""Patch Dask Arrays so it supports `scipy.sparse.sparray`\ s."""
    try:
        from .dask import patch
    except ImportError:
        pass
    else:
        patch()


def register_numba_sparse() -> None:
    r"""Register `scipy.sparse.sp{matrix,array}`\ s with Numba.

    This makes it cleaner to write numba functions operating on these types.
    """
    try:
        from .numba_sparse import register
    except ImportError:
        pass
    else:
        register()
