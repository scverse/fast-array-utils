# SPDX-License-Identifier: MPL-2.0

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, overload

from numpy.typing import ArrayLike, DTypeLike, NDArray

from .core import Array, _Chunk

class _Chunk(Protocol):
    @overload
    def __call__(self, x_chunk: _Chunk, /, *, weights_chunk: NDArray[Any] | None = None, axis: tuple[int, ...], keepdims: bool, **kwargs: object) -> _Chunk: ...
    @overload
    def __call__(self, x_chunk: _Chunk, /, *, axis: tuple[int, ...], keepdims: bool, **kwargs: object) -> _Chunk: ...

class _CB(Protocol):
    def __call__(self, x_chunk: _Chunk, /, *, axis: tuple[int, ...], keepdims: bool, **kwargs: object) -> _Chunk: ...

def reduction(
    x: Array,
    chunk: _Chunk,
    aggregate: _CB,
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike | None = None,
    split_every: int | Mapping[int, int] | None = None,
    combine: _CB | None = None,
    name: str | None = None,
    out: Array | None = None,
    concatenate: bool = True,
    output_size: int = 1,
    meta: _Chunk | None = None,
    weights: ArrayLike | None = None,
) -> Array: ...
