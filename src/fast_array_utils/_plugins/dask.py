# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from dask.array.dispatch import concatenate_lookup, tensordot_lookup
from scipy.sparse import sparray, spmatrix


try:
    from dask.array.dispatch import take_lookup
except ImportError:
    take_lookup = None


# TODO(flying-sheep): upstream
# https://github.com/dask/dask/issues/11749
def patch() -> None:  # pragma: no cover
    """Patch dask to support sparse arrays.

    See <https://github.com/dask/dask/blob/d9b5c5b0256208f1befe94b26bfa8eaabcd0536d/dask/array/backends.py#L239-L241>
    """
    # Avoid patch if already patched or upstream support has been added
    if concatenate_lookup.dispatch(sparray) is concatenate_lookup.dispatch(spmatrix):
        return

    concatenate_lookup.register(sparray, concatenate_lookup.dispatch(spmatrix))
    tensordot_lookup.register(sparray, tensordot_lookup.dispatch(spmatrix))
    if take_lookup is not None:
        take_lookup.register(sparray, take_lookup.dispatch(spmatrix))
