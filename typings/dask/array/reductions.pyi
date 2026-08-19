# SPDX-License-Identifier: MPL-2.0

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, overload

from numpy.typing import ArrayLike, DTypeLike, NDArray

from .core import Array, _Array

# dask only passes `computing_meta` to callbacks that accept it, so it’s optional here.
class _Chunk[T](Protocol):
    @overload
    def __call__(
        self,
        x_chunk: _Array,
        /,
        *,
        weights_chunk: NDArray[Any] | None = None,
        axis: tuple[int, ...],
        keepdims: bool,
        computing_meta: bool = ...,
        **kwargs: object,
    ) -> _Array | T: ...
    @overload
    def __call__(self, x_chunk: _Array, /, *, axis: tuple[int, ...], keepdims: bool, computing_meta: bool = ..., **kwargs: object) -> _Array | T: ...

class _CB[T](Protocol):
    # When `concatenate=False`, dask passes a (possibly nested) list of the previous
    # step’s raw outputs instead of concatenating them into a single `_Array`.
    def __call__(
        self, x_chunk: _Array | T | Sequence[Any], /, *, axis: tuple[int, ...], keepdims: bool, computing_meta: bool = ..., **kwargs: object
    ) -> _Array | T: ...

def reduction[T](
    x: Array,
    chunk: _Chunk[T],
    aggregate: _CB[T],
    *,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
    dtype: DTypeLike | None = None,
    split_every: int | Mapping[int, int] | None = None,
    combine: _CB[T] | None = None,
    name: str | None = None,
    out: Array | None = None,
    concatenate: bool = True,
    output_size: int = 1,
    meta: _Array | None = None,
    weights: ArrayLike | None = None,
) -> Array: ...
