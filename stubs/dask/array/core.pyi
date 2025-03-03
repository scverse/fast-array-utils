# pyright: reportIncompatibleMethodOverride=false
from collections.abc import Callable, Sequence
from typing import Never, TypeAlias, override

import cupy
import cupyx.scipy.sparse
import numpy as np
import scipy.sparse
from numpy.typing import NDArray
from optype.numpy import ToDType

_Chunks: TypeAlias = tuple[int, ...] | tuple[tuple[int, ...], ...]
_Array: TypeAlias = (
    NDArray[np.generic]
    | scipy.sparse.csr_array
    | scipy.sparse.csc_array
    | scipy.sparse.csr_matrix
    | scipy.sparse.csc_matrix
    | cupy.ndarray
    | cupyx.scipy.sparse.spmatrix
)

class BlockView:
    size: int
    shape: tuple[int, ...]

    def __getitem__(self, index: object) -> Array: ...
    def ravel(self) -> list[Array]: ...

class Array:
    shape: tuple[int, ...]
    _meta: object

    blocks: BlockView

    @override
    def __eq__(self, value: object, /) -> Array: ...
    def compute(self) -> _Array: ...
    def map_blocks(
        self,
        func: Callable[[], Array],
        *args: Never,
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
