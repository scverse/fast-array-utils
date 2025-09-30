# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from functools import partial, singledispatch
from typing import TYPE_CHECKING, cast

import numba
import numpy as np

from .. import types


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal, TypeVar

    from numpy.typing import NDArray

    C = TypeVar("C", bound=Callable[..., Any])


@singledispatch
def is_constant_(
    a: NDArray[Any] | types.CSBase | types.CupyArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1] | None = None,
) -> bool | NDArray[np.bool] | types.CupyArray | types.DaskArray:  # pragma: no cover
    raise NotImplementedError


@is_constant_.register(np.ndarray | types.CupyArray)
def _is_constant_ndarray(a: NDArray[Any] | types.CupyArray, /, *, axis: Literal[0, 1] | None = None) -> bool | NDArray[np.bool] | types.CupyArray:
    # Should eventually support nd, not now.
    match axis:
        case None:
            return bool((a == a.flat[0]).all())
        case 0:
            return _is_constant_rows(a.T)
        case 1:
            return _is_constant_rows(a)


def _is_constant_rows(a: NDArray[Any] | types.CupyArray) -> NDArray[np.bool] | types.CupyArray:
    b = np.broadcast_to(a[:, 0][:, np.newaxis], a.shape)
    return cast("NDArray[np.bool]", (a == b).all(axis=1))


@is_constant_.register(types.CSBase)
def _is_constant_cs(a: types.CSBase, /, *, axis: Literal[0, 1] | None = None) -> bool | NDArray[np.bool]:
    from . import is_constant

    if len(a.shape) == 1:  # pragma: no cover
        msg = "array must have 2 dimensions"
        raise ValueError(msg)
    n_row, n_col = a.shape
    if axis is None:
        if len(a.data) == n_row * n_col:
            return is_constant(a.data)
        return bool((a.data == 0).all())
    shape = (n_row, n_col) if axis == 1 else (n_col, n_row)
    match axis, a.format:
        case 0, "csr":
            a = a.T.tocsr()
        case 1, "csc":
            a = a.T.tocsc()
    return _is_constant_cs_major(a, shape)


@numba.njit(cache=True)
def _is_constant_cs_major(a: types.CSBase, shape: tuple[int, int]) -> NDArray[np.bool]:
    n = len(a.indptr) - 1
    result = np.ones(n, dtype=np.bool)
    for i in numba.prange(n):
        start = a.indptr[i]
        stop = a.indptr[i + 1]
        val = a.data[start] if stop - start == shape[1] else 0
        for j in range(start, stop):
            if a.data[j] != val:
                result[i] = False
                break
    return result


@is_constant_.register(types.DaskArray)
def _is_constant_dask(a: types.DaskArray, /, *, axis: Literal[0, 1] | None = None) -> types.DaskArray:
    import dask.array as da

    from . import is_constant

    if axis is not None:
        return da.map_blocks(partial(is_constant, axis=axis), a, drop_axis=axis, meta=np.array([], dtype=np.bool))

    rv = (
        (a == a[0, 0].compute()).all()
        if isinstance(a._meta, np.ndarray)  # noqa: SLF001
        else da.map_overlap(
            lambda a: np.array([[is_constant(a)]]),  # type: ignore[arg-type]
            a,
            # use asymmetric overlaps to avoid unnecessary computation
            depth=dict.fromkeys(range(a.ndim), (0, 1)),
            trim=False,
            meta=np.array([], dtype=bool),
        ).all()
    )
    return da.map_blocks(bool, rv, meta=np.array([], dtype=bool))
