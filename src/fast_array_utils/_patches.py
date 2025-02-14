# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import numpy as np


# TODO(flying-sheep): upstream
# https://github.com/dask/dask/issues/11749
def patch_dask() -> None:
    """Patch dask to support sparse arrays.

    See <https://github.com/dask/dask/blob/4d71629d1f22ced0dd780919f22e70a642ec6753/dask/array/backends.py#L212-L232>
    """
    try:
        # Other lookup candidates: tensordot_lookup and take_lookup
        from dask.array.dispatch import concatenate_lookup
        from scipy.sparse import sparray, spmatrix
    except ImportError:
        return  # No need to patch if dask or scipy is not installed

    # Avoid patch if already patched or upstream support has been added
    if concatenate_lookup.dispatch(sparray) is not np.concatenate:  # type: ignore[no-untyped-call]
        return

    concatenate = concatenate_lookup.dispatch(spmatrix)  # type: ignore[no-untyped-call]
    concatenate_lookup.register(sparray, concatenate)  # type: ignore[no-untyped-call]
