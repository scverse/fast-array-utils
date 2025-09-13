# SPDX-License-Identifier: MPL-2.0
from typing import Literal

import h5py
import zarr
from numpy.typing import ArrayLike, DTypeLike

from .._core import ndarray

def asarray(
    a: ArrayLike | h5py.Dataset | zarr.Array,
    dtype: DTypeLike | None = None,
    order: Literal["C", "F", "A", "K"] | None = None,
    *,
    blocking: bool = False,
) -> ndarray: ...
