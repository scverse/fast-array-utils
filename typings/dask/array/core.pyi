# SPDX-License-Identifier: MPL-2.0
# pyright: reportIncompatibleMethodOverride=false
from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Literal

import cupy
import cupyx.scipy.sparse
import numpy as np
import scipy.sparse
from numpy.typing import DTypeLike, NDArray

from ..utils import SerializableLock

type _Chunks = tuple[int, ...] | tuple[tuple[int, ...], ...]
type _Array = (
    NDArray[Any]
    | scipy.sparse.csr_array
    | scipy.sparse.csc_array
    | scipy.sparse.csr_matrix
    | scipy.sparse.csc_matrix
    | cupy.ndarray
    | cupyx.scipy.sparse.csr_matrix
    | cupyx.scipy.sparse.csc_matrix
)

class BlockView:
    size: int
    shape: tuple[int, ...]

    def __getitem__(self, index: object) -> Array: ...
    def ravel(self) -> list[Array]: ...

class Array[C: _Array = _Array]:
    # array methods and attrs
    ndim: int
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    def __eq__(self, value: object, /) -> Array: ...  # ty:ignore[invalid-method-override]
    def __getitem__(self, index: object) -> Array: ...
    def all(self) -> Array: ...

    # dask methods and attrs
    _meta: C
    blocks: BlockView
    chunks: tuple[tuple[int, ...], ...]
    chunksize: tuple[int, ...]

    def compute(self) -> C: ...
    def visualize(
        self,
        filename: str = "mydask",
        format: Literal["png", "pdf", "dot", "svg", "jpeg", "jpg"] | None = None,
        optimize_graph: bool = False,
        *,
        traverse: bool = True,
        maxval: float | None = None,
        color: Literal["order", "ages", "freed", "memoryincreases", "memorydecreases", "memorypressure"] | None = None,
        collapse_outputs: bool = False,
        verbose: bool = False,
        engine: str = "ipycytoscape",
    ) -> object: ...
    def map_blocks[C2: _Array, **P](
        self,
        func: Callable[Concatenate[C, P], C2],
        *args: P.args,
        name: str | None = None,
        token: str | None = None,
        dtype: DTypeLike | None = None,
        chunks: _Chunks | None = None,
        drop_axis: Sequence[int] | int | None = None,
        new_axis: Sequence[int] | int | None = None,
        enforce_ndim: bool = False,
        meta: C2 | None = None,
        **kwargs: P.kwargs,
    ) -> Array: ...

def from_array[C: _Array = _Array](
    x: C,
    chunks: _Chunks | str | Literal["auto"] = "auto",  # noqa: PYI051
    name: str | None = None,
    lock: bool | SerializableLock = False,
    asarray: bool | None = None,
    fancy: bool = True,
    getitem: object = None,  # undocumented
    meta: C | None = None,
    inline_array: bool = False,
) -> Array: ...
def map_blocks[C: _Array = _Array](
    func: Callable[[object], C],
    *args: Array,
    name: str | None = None,
    token: str | None = None,
    dtype: DTypeLike | None = None,
    chunks: _Chunks | None = None,
    drop_axis: Sequence[int] | int | None = None,
    new_axis: Sequence[int] | int | None = None,
    enforce_ndim: bool = False,
    meta: object | None = None,
    **kwargs: object,
) -> Array[C]: ...
