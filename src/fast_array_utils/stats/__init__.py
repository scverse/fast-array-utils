# SPDX-License-Identifier: MPL-2.0
"""Statistics utilities for 2D arrays.

All of these allow you to specify an ``axis``,
which allows you to choose whether to compute the statistic across rows, columns, or all elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, get_args, overload

from .._validation import validate_axis
from ..typing import CpuArray, DiskArray, GpuArray  # noqa: TC001
from ._generic_ops import DtypeOps


if TYPE_CHECKING:
    from typing import Any, Literal

    import numpy as np
    from numpy.typing import DTypeLike, NDArray
    from optype.numpy import ToDType

    from .. import types
    from ._generic_ops import Ops
    from ._typing import NoDtypeOps, StatFunDtype, StatFunNoDtype


__all__ = ["is_constant", "max", "mean", "mean_var", "min", "sum"]


@overload
def is_constant(x: NDArray[Any] | types.CSBase | types.CupyArray, /, *, axis: None = None) -> bool: ...
@overload
def is_constant(x: NDArray[Any] | types.CSBase, /, *, axis: Literal[0, 1]) -> NDArray[np.bool]: ...
@overload
def is_constant(x: types.CupyArray, /, *, axis: Literal[0, 1]) -> types.CupyArray: ...
@overload
def is_constant(x: types.DaskArray, /, *, axis: Literal[0, 1] | None = None) -> types.DaskArray: ...


def is_constant(
    x: NDArray[Any] | types.CSBase | types.CupyArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1] | None = None,
) -> bool | NDArray[np.bool] | types.CupyArray | types.DaskArray:
    """Check whether values in array are constant.

    Parameters
    ----------
    x
        Array to check.
    axis
        Axis to reduce over.

    Returns
    -------
    If ``axis`` is :data:`None`, return if all values were constant.
    Else returns a boolean array with :data:`True` representing constant columns/rows.

    Example
    -------
    >>> import numpy as np
    >>> x = np.array([
    ...     [0, 1, 2],
    ...     [0, 0, 0],
    ... ])
    >>> is_constant(x)
    False
    >>> is_constant(x, axis=0)
    array([ True, False, False])
    >>> is_constant(x, axis=1)
    array([False,  True])

    """
    from ._is_constant import is_constant_

    validate_axis(x.ndim, axis)
    return is_constant_(x, axis=axis)


# TODO(flying-sheep): support CSDataset (TODO)
# https://github.com/scverse/fast-array-utils/issues/52
@overload
def mean(x: CpuArray | GpuArray | DiskArray, /, *, axis: None = None, dtype: DTypeLike | None = None) -> np.number[Any]: ...
@overload
def mean(x: CpuArray | DiskArray, /, *, axis: Literal[0, 1], dtype: DTypeLike | None = None) -> NDArray[np.number[Any]]: ...
@overload
def mean(x: GpuArray, /, *, axis: Literal[0, 1], dtype: DTypeLike | None = None) -> types.CupyArray: ...
@overload
def mean(x: types.DaskArray, /, *, axis: Literal[0, 1], dtype: ToDType[Any] | None = None) -> types.DaskArray: ...


def mean(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1] | None = None,
    dtype: DTypeLike | None = None,
) -> NDArray[np.number[Any]] | types.CupyArray | np.number[Any] | types.DaskArray:
    """Mean over both or one axis.

    Parameters
    ----------
    x
        Array to calculate mean(s) for.
    axis
        Axis to reduce over.

    Returns
    -------
    If ``axis`` is :data:`None`, then the sum over all elements is returned as a scalar.
    Otherwise, the sum over the given axis is returned as a 1D array.

    Example
    -------
    >>> import numpy as np
    >>> x = np.array([
    ...     [0, 1, 2],
    ...     [0, 0, 0],
    ... ])
    >>> mean(x)
    np.float64(0.5)
    >>> mean(x, axis=0)
    array([0. , 0.5, 1. ])
    >>> mean(x, axis=1)
    array([1., 0.])

    See Also
    --------
    :func:`numpy.mean`
    """
    from ._mean import mean_

    validate_axis(x.ndim, axis)
    return mean_(x, axis=axis, dtype=dtype)


@overload
def mean_var(x: CpuArray | GpuArray, /, *, axis: None = None, correction: int = 0) -> tuple[np.float64, np.float64]: ...
@overload
def mean_var(x: CpuArray, /, *, axis: Literal[0, 1], correction: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
@overload
def mean_var(x: GpuArray, /, *, axis: Literal[0, 1], correction: int = 0) -> tuple[types.CupyArray, types.CupyArray]: ...
@overload
def mean_var(x: types.DaskArray, /, *, axis: Literal[0, 1] | None = None, correction: int = 0) -> tuple[types.DaskArray, types.DaskArray]: ...


def mean_var(
    x: CpuArray | GpuArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1] | None = None,
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

    Example
    -------
    >>> import numpy as np
    >>> x = np.array([
    ...     [0, 1, 2],
    ...     [0, 0, 0],
    ... ])
    >>> mean_var(x)  # doctest: +FLOAT_CMP
    (np.float64(0.5), np.float64(0.5833333333333334))
    >>> mean_var(x, axis=0)
    (array([0. , 0.5, 1. ]), array([0.  , 0.25, 1.  ]))
    >>> mean_var(x, axis=1)
    (array([1., 0.]), array([0.66666667, 0.        ]))

    See Also
    --------
    :func:`numpy.mean`
    :func:`numpy.var`
    """
    from ._mean_var import mean_var_

    validate_axis(x.ndim, axis)
    return mean_var_(x, axis=axis, correction=correction)  # type: ignore[no-any-return]


@overload
def _mk_generic_op(op: NoDtypeOps) -> StatFunNoDtype: ...
@overload
def _mk_generic_op(op: DtypeOps) -> StatFunDtype: ...


# TODO(flying-sheep): support CSDataset (TODO)
# https://github.com/scverse/fast-array-utils/issues/52
def _mk_generic_op(op: Ops) -> StatFunNoDtype | StatFunDtype:
    def _generic_op(
        x: CpuArray | GpuArray | DiskArray | types.DaskArray,
        /,
        *,
        axis: Literal[0, 1] | None = None,
        dtype: DTypeLike | None = None,
        keep_cupy_as_array: bool = False,
    ) -> NDArray[Any] | np.number[Any] | types.CupyArray | types.DaskArray:
        from ._generic_ops import generic_op

        assert dtype is None or op in get_args(DtypeOps), f"`dtype` is not supported for operation {op!r}"

        validate_axis(x.ndim, axis)
        return generic_op(x, op, axis=axis, keep_cupy_as_array=keep_cupy_as_array, dtype=dtype)

    _generic_op.__name__ = op
    return cast("StatFunNoDtype | StatFunDtype", _generic_op)


_min = _mk_generic_op("min")
_max = _mk_generic_op("max")
_sum = _mk_generic_op("sum")


@overload
def min(x: CpuArray | DiskArray, /, *, axis: None = None, keep_cupy_as_array: bool = False) -> np.number[Any]: ...
@overload
def min(x: CpuArray | DiskArray, /, *, axis: Literal[0, 1], keep_cupy_as_array: bool = False) -> NDArray[Any]: ...
@overload
def min(x: GpuArray, /, *, axis: None = None, keep_cupy_as_array: Literal[False] = False) -> np.number[Any]: ...
@overload
def min(x: GpuArray, /, *, axis: None, keep_cupy_as_array: Literal[True]) -> types.CupyArray: ...
@overload
def min(x: GpuArray, /, *, axis: Literal[0, 1], keep_cupy_as_array: bool = False) -> types.CupyArray: ...
@overload
def min(x: types.DaskArray, /, *, axis: Literal[0, 1] | None = None, keep_cupy_as_array: bool = False) -> types.DaskArray: ...
def min(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1] | None = None,
    keep_cupy_as_array: bool = False,
) -> object:
    """Find the minimum along both or one axis.

    Parameters
    ----------
    x
        Array to find the minimum(s) in.
    axis
        Axis to reduce over.

    Returns
    -------
    If ``axis`` is :data:`None`, then the minimum element is returned as a scalar.
    Otherwise, the minimum along the given axis is returned as a 1D array.

    Example
    -------
    >>> import numpy as np
    >>> x = np.array([
    ...     [0, 1, 2],
    ...     [1, 1, 1],
    ... ])
    >>> min(x)
    np.int64(0)
    >>> min(x, axis=0)
    array([0, 1, 1])
    >>> min(x, axis=1)
    array([0, 1])

    See Also
    --------
    :func:`numpy.min`

    """
    return _min(x, axis=axis, keep_cupy_as_array=keep_cupy_as_array)


@overload
def max(x: CpuArray | DiskArray, /, *, axis: None = None, keep_cupy_as_array: bool = False) -> np.number[Any]: ...
@overload
def max(x: CpuArray | DiskArray, /, *, axis: Literal[0, 1], keep_cupy_as_array: bool = False) -> NDArray[Any]: ...
@overload
def max(x: GpuArray, /, *, axis: None = None, keep_cupy_as_array: Literal[False] = False) -> np.number[Any]: ...
@overload
def max(x: GpuArray, /, *, axis: None, keep_cupy_as_array: Literal[True]) -> types.CupyArray: ...
@overload
def max(x: GpuArray, /, *, axis: Literal[0, 1], keep_cupy_as_array: bool = False) -> types.CupyArray: ...
@overload
def max(x: types.DaskArray, /, *, axis: Literal[0, 1] | None = None, keep_cupy_as_array: bool = False) -> types.DaskArray: ...
def max(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1] | None = None,
    keep_cupy_as_array: bool = False,
) -> object:
    """Find the maximum along both or one axis.

    Parameters
    ----------
    x
        Array to find the maximum(s) in.
    axis
        Axis to reduce over.

    Returns
    -------
    If ``axis`` is :data:`None`, then the maximum element is returned as a scalar.
    Otherwise, the maximum along the given axis is returned as a 1D array.

    Example
    -------
    >>> import numpy as np
    >>> x = np.array([
    ...     [0, 1, 2],
    ...     [0, 0, 0],
    ... ])
    >>> max(x)
    np.int64(2)
    >>> max(x, axis=0)
    array([0, 1, 2])
    >>> max(x, axis=1)
    array([2, 0])

    See Also
    --------
    :func:`numpy.max`

    """
    return _max(x, axis=axis, keep_cupy_as_array=keep_cupy_as_array)


@overload
def sum(x: CpuArray | DiskArray, /, *, axis: None = None, dtype: DTypeLike | None = None, keep_cupy_as_array: bool = False) -> np.number[Any]: ...
@overload
def sum(x: CpuArray | DiskArray, /, *, axis: Literal[0, 1], dtype: DTypeLike | None = None, keep_cupy_as_array: bool = False) -> NDArray[Any]: ...
@overload
def sum(x: GpuArray, /, *, axis: None = None, dtype: DTypeLike | None = None, keep_cupy_as_array: Literal[False] = False) -> np.number[Any]: ...
@overload
def sum(x: GpuArray, /, *, axis: None = None, dtype: DTypeLike | None = None, keep_cupy_as_array: Literal[True]) -> types.CupyArray: ...
@overload
def sum(x: GpuArray, /, *, axis: Literal[0, 1], dtype: DTypeLike | None = None, keep_cupy_as_array: bool = False) -> types.CupyArray: ...
@overload
def sum(x: types.DaskArray, /, *, axis: Literal[0, 1] | None = None, dtype: DTypeLike | None = None, keep_cupy_as_array: bool = False) -> types.DaskArray: ...
def sum(
    x: CpuArray | GpuArray | DiskArray | types.DaskArray,
    /,
    *,
    axis: Literal[0, 1] | None = None,
    dtype: DTypeLike | None = None,
    keep_cupy_as_array: bool = False,
) -> NDArray[Any] | types.CupyArray | np.number[Any] | types.DaskArray:
    """Sum over both or one axis.

    Parameters
    ----------
    x
        Array to sum up.
    axis
        Axis to reduce over.

    Returns
    -------
    If ``axis`` is :data:`None`, then the sum over all elements is returned as a scalar.
    Otherwise, the sum over the given axis is returned as a 1D array.

    Example
    -------
    >>> import numpy as np
    >>> x = np.array([
    ...     [0, 1, 2],
    ...     [0, 0, 0],
    ... ])
    >>> sum(x)
    np.int64(3)
    >>> sum(x, axis=0)
    array([0, 1, 2])
    >>> sum(x, axis=1)
    array([3, 0])

    See Also
    --------
    :func:`numpy.sum`

    """
    return _sum(x, axis=axis, dtype=dtype, keep_cupy_as_array=keep_cupy_as_array)
