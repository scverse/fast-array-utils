# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest


if TYPE_CHECKING or find_spec("scipy"):
    from scipy.sparse import sparray, spmatrix
else:
    spmatrix = sparray = type("spmatrix", (), {})

from fast_array_utils import stats, types


if TYPE_CHECKING:
    from typing import Any, Literal

    from testing.fast_array_utils import _Array, _ToArray


@pytest.mark.parametrize("dtype_in", [np.float32, np.float64, np.int32, np.bool_])
@pytest.mark.parametrize("dtype_arg", [np.float32, np.float64, None])
@pytest.mark.parametrize("axis", [0, 1, None])
def test_sum(
    array_cls: type[_Array[Any]],
    to_array: _ToArray[Any],
    dtype_in: type[np.generic],
    dtype_arg: type[np.generic] | None,
    axis: Literal[0, 1, None],
) -> None:
    np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype_in)
    arr = to_array(np_arr.copy())
    assert arr.dtype == dtype_in

    sum_: _Array[Any] | np.floating = stats.sum(arr, axis=axis, dtype=dtype_arg)  # type: ignore[type-arg,arg-type]

    match axis, arr:
        case _, types.DaskArray():
            assert isinstance(sum_, types.DaskArray), type(sum_)
            sum_ = sum_.compute()  # type: ignore[no-untyped-call]
        case None, _:
            assert isinstance(sum_, np.floating | np.integer), type(sum_)
        case 0 | 1, spmatrix() | sparray() | types.ZarrArray() | types.H5Dataset():
            assert isinstance(sum_, np.ndarray), type(sum_)
        case 0 | 1, _:
            assert isinstance(sum_, array_cls), type(sum_)
        case _:
            pytest.fail(f"Unhandled case axis {axis} for {type(arr)}: {type(sum_)}")

    assert sum_.shape == () if axis is None else arr.shape[axis], (sum_.shape, arr.shape)

    if dtype_arg is not None:
        assert sum_.dtype == dtype_arg, (sum_.dtype, dtype_arg)
    elif dtype_in in {np.bool_, np.int32}:
        assert sum_.dtype == np.int64
    else:
        assert sum_.dtype == dtype_in

    np.testing.assert_array_equal(sum_, np.sum(np_arr, axis=axis, dtype=dtype_arg))  # type: ignore[arg-type]
