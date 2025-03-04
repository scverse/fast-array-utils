# SPDX-License-Identifier: MPL-2.0

from collections.abc import Mapping, Sequence
from typing import Protocol, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from optype.numpy import ToDType

from .core import Array

class _Chunk(Protocol):
    @overload
    def __call__(
        self,
        x_chunk: object,
        /,
        *,
        weights_chunk: NDArray[np.generic] | None = None,
        axis: tuple[int, ...],
        keepdims: bool,
        **kwargs: object,
    ) -> object: ...
    @overload
    def __call__(
        self, x_chunk: object, /, *, axis: tuple[int, ...], keepdims: bool, **kwargs: object
    ) -> object: ...

class _CB(Protocol):
    def __call__(
        self, x_chunk: object, /, *, axis: tuple[int, ...], keepdims: bool, **kwargs: object
    ) -> object: ...

def reduction(
    x: Array,
    chunk: _Chunk,
    aggregate: _CB,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: ToDType[np.generic] | None = None,
    split_every: int | Mapping[int, int] | None = None,
    combine: _CB | None = None,
    name: str | None = None,
    out: Array | None = None,
    concatenate: bool = True,
    output_size: int = 1,
    meta: object = None,
    weights: ArrayLike | None = None,
) -> Array: ...
