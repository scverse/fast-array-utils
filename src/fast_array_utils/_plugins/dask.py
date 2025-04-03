# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import numpy as np

# Other lookup candidates: tensordot_lookup and take_lookup
from dask.array.dispatch import concatenate_lookup
from scipy.sparse import sparray, spmatrix


# TODO(flying-sheep): upstream
# https://github.com/dask/dask/issues/11749
def patch() -> None:  # pragma: no cover
    """Patch dask to support sparse arrays.

    See <https://github.com/dask/dask/blob/4d71629d1f22ced0dd780919f22e70a642ec6753/dask/array/backends.py#L212-L232>
    """
    # Avoid patch if already patched or upstream support has been added
    if concatenate_lookup.dispatch(sparray) is not np.concatenate:
        return

    concatenate = concatenate_lookup.dispatch(spmatrix)
    concatenate_lookup.register(sparray, concatenate)
