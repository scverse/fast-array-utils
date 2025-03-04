# SPDX-License-Identifier: MPL-2.0
# pyright: reportPrivateUsage=false
from collections.abc import Callable, Sequence

import numpy as np
from optype.numpy import ToDType

from .core import Array as Array
from .core import _Chunks
from .overlap import map_overlap as map_overlap
from .reductions import reduction as reduction

def map_blocks(
    func: Callable[[Array], Array],
    *args: Array,
    name: str | None = None,
    token: str | None = None,
    dtype: ToDType[np.generic] | None = None,
    chunks: _Chunks | None = None,
    drop_axis: Sequence[int] | int | None = None,
    new_axis: Sequence[int] | int | None = None,
    enforce_ndim: bool = False,
    meta: object | None = None,
    **kwargs: object,
) -> Array: ...
