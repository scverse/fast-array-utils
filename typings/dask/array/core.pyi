# SPDX-License-Identifier: MPL-2.0
# pyright: reportIncompatibleMethodOverride=false
from collections.abc import Callable, Sequence
from typing import Any, Literal, Never, TypeAlias

import cupy
import cupyx.scipy.sparse
import numpy as np
import scipy.sparse
from numpy.typing import DTypeLike, NDArray
from typing_extensions import override

from ..utils import SerializableLock

_Chunks: TypeAlias = tuple[int, ...] | tuple[tuple[int, ...], ...]
_Array: TypeAlias = (
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

class Array:
    # array methods and attrs
    ndim: int
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    @override
    def __eq__(self, value: object, /) -> Array: ...  # type: ignore[override]
    def __getitem__(self, index: object) -> Array: ...
    def all(self) -> Array: ...

    # dask methods and attrs
    _meta: _Array
    blocks: BlockView
    chunks: tuple[tuple[int, ...], ...]
    chunksize: tuple[int, ...]

    def compute(self) -> _Array: ...
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
    def map_blocks(
        self,
        # TODO(flying-sheep): make this generic, _Array the default  # noqa: TD003
        func: Callable[[object], object],
        *args: Never,
        name: str | None = None,
        token: str | None = None,
        dtype: DTypeLike | None = None,
        chunks: _Chunks | None = None,
        drop_axis: Sequence[int] | int | None = None,
        new_axis: Sequence[int] | int | None = None,
        enforce_ndim: bool = False,
        meta: _Array | None = None,
        **kwargs: object,
    ) -> Array: ...

def from_array(
    x: _Array,
    chunks: _Chunks | str | Literal["auto"] = "auto",  # noqa: PYI051
    name: str | None = None,
    lock: bool | SerializableLock = False,
    asarray: bool | None = None,
    fancy: bool = True,
    getitem: object = None,  # undocumented
    meta: _Array | None = None,
    inline_array: bool = False,
) -> Array: ...
def map_blocks(
    # TODO(flying-sheep): make this generic, _Array the default  # noqa: TD003
    func: Callable[[object], object],
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
) -> Array: ...
