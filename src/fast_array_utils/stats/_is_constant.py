# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import partial, singledispatch
from typing import TYPE_CHECKING, Any, cast, overload

import numba
import numpy as np
from numpy.typing import NDArray

from .. import types
from .._validation import validate_axis


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal, TypeVar

    C = TypeVar("C", bound=Callable[..., Any])


@overload
def is_constant(a: types.DaskArray, /, *, axis: Literal[0, 1, None] = None) -> types.DaskArray: ...
@overload
def is_constant(a: NDArray[Any] | types.CSBase, /, *, axis: None = None) -> bool: ...
@overload
def is_constant(a: NDArray[Any] | types.CSBase, /, *, axis: Literal[0, 1]) -> NDArray[np.bool]: ...


def is_constant(
    a: NDArray[Any] | types.CSBase | types.DaskArray, /, *, axis: Literal[0, 1, None] = None
) -> bool | NDArray[np.bool] | types.DaskArray:
    """Check whether values in array are constant.

    Params
    ------
    a
        Array to check
    axis
        Axis to reduce over.

    Returns
    -------
    If ``axis`` is :data:`None`, return if all values were constant.
    Else returns a boolean array with :data:`True` representing constant columns/rows.

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
    validate_axis(axis)
    return _is_constant(a, axis=axis)


@singledispatch
def _is_constant(
    a: NDArray[Any] | types.CSBase | types.DaskArray, /, *, axis: Literal[0, 1, None] = None
) -> bool | NDArray[np.bool] | types.DaskArray:  # pragma: no cover
    raise NotImplementedError


@_is_constant.register(np.ndarray)
def _(a: NDArray[Any], /, *, axis: Literal[0, 1, None] = None) -> bool | NDArray[np.bool]:
    # Should eventually support nd, not now.
    match axis:
        case None:
            return bool((a == a.flat[0]).all())
        case 0:
            return _is_constant_rows(a.T)
        case 1:
            return _is_constant_rows(a)


def _is_constant_rows(a: NDArray[Any]) -> NDArray[np.bool]:
    b = np.broadcast_to(a[:, 0][:, np.newaxis], a.shape)
    return cast(NDArray[np.bool], (a == b).all(axis=1))


@_is_constant.register(types.CSBase)  # type: ignore[call-overload,misc]
def _(a: types.CSBase, /, *, axis: Literal[0, 1, None] = None) -> bool | NDArray[np.bool]:
    if len(a.shape) == 1:  # pragma: no cover
        msg = "array must have 2 dimensions"
        raise ValueError(msg)
    n_row, n_col = a.shape
    if axis is None:
        if len(a.data) == n_row * n_col:
            return is_constant(cast(NDArray[Any], a.data))
        return bool((a.data == 0).all())
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
) -> NDArray[np.bool]:
    n = len(indptr) - 1
    result = np.ones(n, dtype=np.bool)
    for i in numba.prange(n):
        start = indptr[i]
        stop = indptr[i + 1]
        val = data[start] if stop - start == shape[1] else 0
        for j in range(start, stop):
            if data[j] != val:
                result[i] = False
                break
    return result


@_is_constant.register(types.DaskArray)
def _(a: types.DaskArray, /, *, axis: Literal[0, 1, None] = None) -> types.DaskArray:
    import dask.array as da

    if axis is not None:
        return cast(
            types.DaskArray,
            da.map_blocks(  # type: ignore[no-untyped-call]
                partial(is_constant, axis=axis), a, drop_axis=axis, meta=np.array([], dtype=np.bool)
            ),
        )

    rv = cast(
        types.DaskArray,
        (a == a[0, 0].compute()).all()
        if isinstance(a._meta, np.ndarray)  # noqa: SLF001
        else da.map_overlap(  # type: ignore[no-untyped-call]
            lambda a: np.array([[is_constant(a)]]),
            a,
            # use asymmetric overlaps to avoid unnecessary computation
            depth={d: (0, 1) for d in range(a.ndim)},
            trim=False,
            meta=np.array([], dtype=bool),
        ).all(),
    )
    return cast(
        types.DaskArray,
        da.map_blocks(bool, rv, meta=np.array([], dtype=bool)),  # type: ignore[no-untyped-call]
    )
