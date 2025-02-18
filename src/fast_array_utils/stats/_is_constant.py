# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import partial, singledispatch
from numbers import Integral
from typing import TYPE_CHECKING, overload

import numba
import numpy as np

from ..types import CSBase, DaskArray, H5Dataset, ZarrArray


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal, TypeVar

    from numpy.typing import NDArray

    C = TypeVar("C", bound=Callable[..., Any])


@overload
def is_constant(
    a: NDArray[Any] | CSBase | H5Dataset | ZarrArray | DaskArray, axis: None = None
) -> bool: ...
@overload
def is_constant(
    a: NDArray[Any] | CSBase | H5Dataset | ZarrArray | DaskArray, axis: Literal[0, 1]
) -> NDArray[np.bool_]: ...


def is_constant(
    a: NDArray[Any] | CSBase | H5Dataset | ZarrArray | DaskArray, axis: Literal[0, 1] | None = None
) -> bool | NDArray[np.bool_]:
    """Check whether values in array are constant.

    Params
    ------
    a
        Array to check
    axis
        Axis to reduce over.

    Returns
    -------
    Boolean array, True values were constant.

    Example
    -------
    >>> a = np.array([[0, 1], [0, 0]])
    >>> a
    array([[0, 1],
           [0, 0]])
    >>> is_constant(a)
    False
    >>> is_constant(a, axis=0)
    array([ True, False])
    >>> is_constant(a, axis=1)
    array([False,  True])

    """
    if axis is not None:
        if not isinstance(axis, Integral):
            msg = "axis must be integer or None."
            raise TypeError(msg)
        if axis not in (0, 1):
            msg = "We only support axis 0 and 1 at the moment"
            raise NotImplementedError(msg)

    return _is_constant(a, axis)


@singledispatch
def _is_constant(
    a: NDArray[Any] | CSBase | H5Dataset | ZarrArray | DaskArray, axis: Literal[0, 1] | None = None
) -> bool | NDArray[np.bool_]:
    raise NotImplementedError


@_is_constant.register(np.ndarray)
@_is_constant.register(H5Dataset)
@_is_constant.register(ZarrArray)
def _(a: NDArray[Any], axis: Literal[0, 1] | None = None) -> bool | NDArray[np.bool_]:
    # Should eventually support nd, not now.
    match axis:
        case None:
            return bool((a == a[0, 0]).all())
        case 0:
            return _is_constant_rows(a.T)
        case 1:
            return _is_constant_rows(a)


def _is_constant_rows(a: NDArray[Any]) -> NDArray[np.bool_]:
    b = np.broadcast_to(a[:, 0][:, np.newaxis], a.shape)
    return (a == b).all(axis=1)  # type: ignore[no-any-return]


@_is_constant.register(CSBase)  # type: ignore[call-overload,misc]
def _(a: CSBase, axis: Literal[0, 1] | None = None) -> bool | NDArray[np.bool_]:
    n_row, n_col = a.shape
    if axis is None:
        if len(a.data) == n_row * n_col:
            return is_constant(a.data)
        return (a.data == 0).all()  # type: ignore[no-any-return]
    shape = (n_row, n_col) if axis == 1 else (n_col, n_row)
    match axis, a.format:
        case 0, "csr":
            a = a.T.tocsr()
        case 1, "csc":
            a = a.T.tocsc()
    return _is_constant_csr_rows(a.data, a.indptr, shape)


@numba.njit(cache=True)
def _is_constant_csr_rows(
    data: NDArray[np.number[Any]],
    indptr: NDArray[np.integer[Any]],
    shape: tuple[int, int],
) -> NDArray[np.bool_]:
    n = len(indptr) - 1
    result = np.ones(n, dtype=np.bool_)
    for i in numba.prange(n):
        start = indptr[i]
        stop = indptr[i + 1]
        val = data[start] if stop - start == shape[1] else 0
        for j in range(start, stop):
            if data[j] != val:
                result[i] = False
                break
    return result


@_is_constant.register(DaskArray)
def _(a: DaskArray, axis: Literal[0, 1] | None = None) -> bool | NDArray[np.bool_]:
    if TYPE_CHECKING:
        from dask.array.core import map_blocks
    else:
        from dask.array import map_blocks

    if isinstance(a._meta, np.ndarray) and axis is None:  # noqa: SLF001
        v = a[0, 0].compute()
        return (a == v).all()  # type: ignore[no-any-return]
    # TODO(flying-sheep): use overlapping blocks and reduction instead of `drop_axis`  # noqa: TD003
    return map_blocks(  # type: ignore[no-any-return,no-untyped-call]
        partial(is_constant, axis=axis),
        a,
        drop_axis=(0, 1) if axis is None else axis,
        meta=np.array([], dtype=a.dtype),
    )
