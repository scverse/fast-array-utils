# SPDX-License-Identifier: MPL-2.0
"""Statistics utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from .._validation import validate_axis
from ._is_constant import is_constant_
from ._mean import mean
from ._mean_var import mean_var
from ._sum import sum


if TYPE_CHECKING:
    from typing import Any, Literal

    import numpy as np
    from numpy.typing import NDArray

    from .. import types


__all__ = ["is_constant", "mean", "mean_var", "sum"]


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
    return is_constant_(a, axis=axis)
