# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import partial, singledispatch
from typing import TYPE_CHECKING, overload

import numpy as np

from .. import types
from .._validation import validate_axis


if TYPE_CHECKING:
    from typing import Literal, TypeVar

    from numpy._typing._array_like import _ArrayLikeFloat_co as ArrayLike
    from numpy.typing import NDArray
    from optype.numpy import ToDType

    Scalar = np.float32 | np.float64 | np.int64
    # all supported types except Dask and OutOfCoreDataset (TODO)
    StdArray = (
        NDArray[Scalar]
        | types.CSBase
        | types.H5Dataset
        | types.ZarrArray
        | types.CupyArray
        | types.CupySparseMatrix
    )
    Sc = TypeVar("Sc", bound=Scalar)


@overload
def sum(x: ArrayLike | StdArray, /, *, axis: None = None, dtype: None = None) -> Scalar: ...
@overload
def sum(x: ArrayLike | StdArray, /, *, axis: None = None, dtype: ToDType[Sc]) -> Sc: ...
@overload
def sum(
    x: ArrayLike | StdArray, /, *, axis: Literal[0, 1], dtype: None = None
) -> NDArray[Scalar]: ...
@overload
def sum(x: ArrayLike | StdArray, /, *, axis: Literal[0, 1], dtype: ToDType[Sc]) -> NDArray[Sc]: ...
@overload
def sum(
    x: types.DaskArray, /, *, axis: Literal[0, 1, None] = None, dtype: ToDType[Sc] | None = None
) -> types.DaskArray: ...


def sum(
    x: ArrayLike | StdArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: ToDType[Sc] | None = None,
) -> NDArray[Sc] | Sc | types.DaskArray:
    """Sum over both or one axis.

    Returns
    -------
    If ``axis`` is :data:`None`, then the sum over all elements is returned as a scalar.
    Otherwise, the sum over the given axis is returned as a 1D array.

    See Also
    --------
    :func:`numpy.sum`

    """
    validate_axis(axis)
    return _sum(x, axis=axis, dtype=dtype)


@singledispatch
def _sum(
    x: ArrayLike | StdArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: ToDType[Sc] | None = None,
) -> NDArray[Sc] | Sc | types.DaskArray:
    if TYPE_CHECKING:
        # these are never passed to this fallback function, but `singledispatch` wants them
        assert not isinstance(x, types.CSBase | types.DaskArray)
        # np.sum supports these, but doesn’t know it. (TODO: test cupy)
        assert not isinstance(
            x, types.ZarrArray | types.H5Dataset | types.CupyArray | types.CupySparseMatrix
        )
    return np.sum(x, axis=axis, dtype=dtype)


@_sum.register(types.CSBase)
def _sum_cs(  # pyright: ignore[reportUnusedFunction]
    x: types.CSBase, /, *, axis: Literal[0, 1, None] = None, dtype: ToDType[Sc] | None = None
) -> NDArray[Sc] | Sc:
    import scipy.sparse as sp

    if isinstance(x, types.CSMatrix):
        x = sp.csr_array(x) if x.format == "csr" else sp.csc_array(x)
    if TYPE_CHECKING:
        assert isinstance(x, ArrayLike)
    return np.sum(x, axis=axis, dtype=dtype)


@_sum.register(types.DaskArray)
def _sum_dask(  # pyright: ignore[reportUnusedFunction]
    x: types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: ToDType[np.generic] | None = None,
) -> types.DaskArray:
    import dask.array as da

    if isinstance(x._meta, np.matrix):  # pragma: no cover  # noqa: SLF001
        msg = "sum does not support numpy matrices"
        raise TypeError(msg)

    def sum_drop_keepdims(
        a: NDArray[np.generic] | types.CSBase,
        /,
        *,
        axis: (
            tuple[Literal[0], Literal[1]]
            | tuple[Literal[1], Literal[0]]
            | tuple[Literal[0, 1]]
            | Literal[0, 1, None]
        ) = None,
        dtype: ToDType[np.float64 | np.float32],
        keepdims: bool,
    ) -> NDArray[np.generic]:
        del keepdims
        match axis:
            case (0 | 1 as n,):
                axis = n
            case (0, 1) | (1, 0):
                axis = None
            case tuple():  # pragma: no cover  # pyright: ignore[reportUnnecessaryComparison]
                msg = f"`sum` can only sum over `axis=0|1|(0,1)` but got {axis} instead"
                raise ValueError(msg)
            case _:
                pass
        rv_maybe_sc = sum(a, axis=axis, dtype=dtype)
        rv = np.array(rv_maybe_sc, ndmin=1)  # make sure rv is at least 1D
        return rv.reshape((1, len(rv)))

    if dtype is None:
        # Explicitly use numpy result dtype (e.g. `NDArray[bool].sum().dtype == int64`)
        dtype = np.zeros(1, dtype=x.dtype).sum().dtype

    return da.reduction(
        x,
        sum_drop_keepdims,
        partial(np.sum, dtype=dtype),
        axis=axis,
        dtype=dtype,
        meta=np.array([], dtype=dtype),
    )
