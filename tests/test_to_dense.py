# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from contextlib import nullcontext
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import types
from fast_array_utils.conv import to_dense


if TYPE_CHECKING:
    from typing import Literal, TypeAlias

    from fast_array_utils.typing import CpuArray, DiskArray, GpuArray
    from testing.fast_array_utils import ArrayType

    Array: TypeAlias = CpuArray | GpuArray | DiskArray | types.CSDataset | types.DaskArray


WARNS_NUMBA = pytest.warns(RuntimeWarning, match="numba is not installed; falling back to slow conversion")


@pytest.mark.parametrize("to_cpu_memory", [True, False], ids=["to_cpu_memory", "not_to_cpu_memory"])
@pytest.mark.parametrize("order", argvalues=["K", "C", "F"])  # “A” behaves like “K”
def test_to_dense(array_type: ArrayType[Array], *, order: Literal["K", "C", "F"], to_cpu_memory: bool) -> None:
    x = array_type([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    if not to_cpu_memory and array_type.cls in {types.CSCDataset, types.CSRDataset}:
        with pytest.raises(ValueError, match="to_cpu_memory must be True if x is an CS{R,C}Dataset"):
            to_dense(x, order=order, to_cpu_memory=to_cpu_memory)
        return

    with WARNS_NUMBA if issubclass(array_type.cls, types.CSBase) and not find_spec("numba") else nullcontext():
        arr = to_dense(x, order=order, to_cpu_memory=to_cpu_memory)

    assert_expected_cls(x, arr, to_cpu_memory=to_cpu_memory)
    assert arr.shape == (2, 3)
    assert_expected_order(x, arr, order=order)


@pytest.mark.parametrize("to_cpu_memory", [True, False], ids=["to_cpu_memory", "not_to_cpu_memory"])
@pytest.mark.parametrize("order", argvalues=["K", "C", "F"])  # “A” behaves like “K”
def test_to_dense_extra(coo_matrix_type: ArrayType[Array], *, order: Literal["K", "C", "F"], to_cpu_memory: bool) -> None:
    src_mtx = coo_matrix_type([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    with WARNS_NUMBA if not find_spec("numba") else nullcontext():
        arr = to_dense(src_mtx, order=order, to_cpu_memory=to_cpu_memory)

    assert_expected_cls(src_mtx, arr, to_cpu_memory=to_cpu_memory)
    assert arr.shape == (2, 3)
    assert_expected_order(src_mtx, arr, order=order)


def assert_expected_cls(orig: Array, converted: Array, *, to_cpu_memory: bool) -> None:
    match (to_cpu_memory, orig):
        case False, types.DaskArray():
            assert isinstance(converted, types.DaskArray)
            assert_expected_cls(orig.compute(), converted.compute(), to_cpu_memory=to_cpu_memory)
        case False, types.CupyArray() | types.CupySpMatrix():
            assert isinstance(converted, types.CupyArray)
        case _:
            assert isinstance(converted, np.ndarray)


def assert_expected_order(orig: Array, converted: Array, *, order: Literal["K", "C", "F"]) -> None:
    match converted:
        case types.CupyArray() | np.ndarray():
            order_expected = get_order(orig) if order == "K" else order
            assert converted.flags.c_contiguous == (order_expected == "C")
            assert converted.flags.f_contiguous == (order_expected == "F")
        case types.DaskArray():
            assert_expected_order(orig, converted.compute(), order=order)
        case _:
            pytest.fail(f"Unsupported array type: {type(converted)}")


def get_order(orig: Array) -> Literal["C", "F"]:
    match orig:
        case np.ndarray() | types.CupyArray():
            return "C" if orig.flags.c_contiguous else "F"
        case _ if isinstance(orig, types.CSBase | types.CupyCSMatrix | types.CSDataset):
            return "C" if orig.format == "csr" else "F"
        case types.DaskArray():
            return get_order(orig.compute())
        case types.ZarrArray() | types.H5Dataset():
            return "C"
    pytest.fail(f"Unsupported array type: {type(orig)}")
