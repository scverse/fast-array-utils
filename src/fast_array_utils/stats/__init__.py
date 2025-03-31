# SPDX-License-Identifier: MPL-2.0
"""Statistics utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from .._validation import validate_axis
from ..typing import CpuArray, DiskArray, GpuArray  # noqa: TC001


if TYPE_CHECKING:
    from typing import Any, Literal

    import numpy as np
    from numpy.typing import DTypeLike, NDArray
    from optype.numpy import ToDType

    from .. import types


__all__ = ["is_constant", "mean", "mean_var", "sum"]


@overload
def is_constant(
    a: NDArray[Any] | types.CSBase | types.CupyArray, /, *, axis: None = None
) -> bool: ...
@overload
def is_constant(a: NDArray[Any] | types.CSBase, /, *, axis: Literal[0, 1]) -> NDArray[np.bool]: ...
@overload
def is_constant(a: types.CupyArray, /, *, axis: Literal[0, 1]) -> types.CupyArray: ...
@overload
def is_constant(a: types.DaskArray, /, *, axis: Literal[0, 1, None] = None) -> types.DaskArray: ...


def is_constant(
    a: NDArray[Any] | types.CSBase | types.CupyArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
) -> bool | NDArray[np.bool] | types.CupyArray | types.DaskArray:
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
    >>> import numpy as np
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
    from ._is_constant import is_constant_

    validate_axis(axis)
    return is_constant_(a, axis=axis)


# TODO(flying-sheep): support CSDataset (TODO)
# https://github.com/scverse/fast-array-utils/issues/52
@overload
def mean(
    x: CpuArray | GpuArray | DiskArray,
    /,
    *,
    axis: Literal[None] = None,
    dtype: DTypeLike | None = None,
) -> np.number[Any]: ...
@overload
def mean(
    x: CpuArray | DiskArray, /, *, axis: Literal[0, 1], dtype: DTypeLike | None = None
) -> NDArray[np.number[Any]]: ...
@overload
def mean(
    x: GpuArray, /, *, axis: Literal[0, 1], dtype: DTypeLike | None = None
) -> types.CupyArray: ...
@overload
def mean(
    x: types.DaskArray, /, *, axis: Literal[0, 1], dtype: ToDType[Any] | None = None
) -> types.DaskArray: ...


def mean(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
) -> NDArray[np.number[Any]] | types.CupyArray | np.number[Any] | types.DaskArray:
    """Mean over both or one axis.

    Returns
    -------
    If ``axis`` is :data:`None`, then the sum over all elements is returned as a scalar.
    Otherwise, the sum over the given axis is returned as a 1D array.

    See Also
    --------
    :func:`numpy.mean`
    """
    from ._mean import mean_

    validate_axis(axis)
    return mean_(x, axis=axis, dtype=dtype)  # type: ignore[no-any-return]  # literally the same type, wtf mypy


@overload
def mean_var(
    x: CpuArray | GpuArray, /, *, axis: Literal[None] = None, correction: int = 0
) -> tuple[np.float64, np.float64]: ...
@overload
def mean_var(
    x: CpuArray, /, *, axis: Literal[0, 1], correction: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
@overload
def mean_var(
    x: GpuArray, /, *, axis: Literal[0, 1], correction: int = 0
) -> tuple[types.CupyArray, types.CupyArray]: ...
@overload
def mean_var(
    x: types.DaskArray, /, *, axis: Literal[0, 1, None] = None, correction: int = 0
) -> tuple[types.DaskArray, types.DaskArray]: ...


def mean_var(
    x: CpuArray | GpuArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    correction: int = 0,
) -> (
    tuple[np.float64, np.float64]
    | tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[types.CupyArray, types.CupyArray]
    | tuple[types.DaskArray, types.DaskArray]
):
    """Mean and variance over both or one axis.

    Parameters
    ----------
    x
        Array to compute mean and variance for.
    axis
        Axis to reduce over.
    correction
        Degrees of freedom correction.

    Returns
    -------
    mean
        See below:
    var
        If ``axis`` is :data:`None`,
        the mean and variance over all elements are returned as scalars.
        Otherwise, the means and variances over the given axis are returned as 1D arrays.

    See Also
    --------
    :func:`numpy.mean`
    :func:`numpy.var`
    """
    from ._mean_var import mean_var_

    return mean_var_(x, axis=axis, correction=correction)  # type: ignore[no-any-return]


# TODO(flying-sheep): support CSDataset (TODO)
# https://github.com/scverse/fast-array-utils/issues/52
@overload
def sum(
    x: CpuArray | GpuArray | DiskArray, /, *, axis: None = None, dtype: DTypeLike | None = None
) -> np.number[Any]: ...
@overload
def sum(
    x: CpuArray | DiskArray, /, *, axis: Literal[0, 1], dtype: DTypeLike | None = None
) -> NDArray[Any]: ...
@overload
def sum(
    x: GpuArray, /, *, axis: Literal[0, 1], dtype: DTypeLike | None = None
) -> types.CupyArray: ...
@overload
def sum(
    x: types.DaskArray, /, *, axis: Literal[0, 1, None] = None, dtype: DTypeLike | None = None
) -> types.DaskArray: ...


def sum(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1, None] = None,
    dtype: DTypeLike | None = None,
) -> NDArray[Any] | types.CupyArray | np.number[Any] | types.DaskArray:
    """Sum over both or one axis.

    Returns
    -------
    If ``axis`` is :data:`None`, then the sum over all elements is returned as a scalar.
    Otherwise, the sum over the given axis is returned as a 1D array.

    See Also
    --------
    :func:`numpy.sum`

    """
    from ._sum import sum_

    validate_axis(axis)
    return sum_(x, axis=axis, dtype=dtype)
