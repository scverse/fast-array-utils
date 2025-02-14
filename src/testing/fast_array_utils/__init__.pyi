# SPDX-License-Identifier: MPL-2.0
from typing import Generic, Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fast_array_utils import types

_SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.generic)
_SCT_contra = TypeVar("_SCT_contra", contravariant=True, bound=np.generic)

_Array: TypeAlias = (
    NDArray[_SCT_co]
    | types.CSBase[_SCT_co]
    | types.CupyArray[_SCT_co]
    | types.DaskArray
    | types.H5Dataset
    | types.ZarrArray
)

class _ToArray(Protocol, Generic[_SCT_contra]):
    def __call__(
        self, data: ArrayLike, /, *, dtype: _SCT_contra | None = None
    ) -> _Array[_SCT_contra]: ...

__all__ = ["_Array", "_ToArray"]
