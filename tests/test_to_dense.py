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
    from typing import TypeAlias

    from fast_array_utils.typing import CpuArray, DiskArray, GpuArray
    from testing.fast_array_utils import ArrayType

    Array: TypeAlias = CpuArray | GpuArray | DiskArray | types.CSDataset | types.DaskArray


WARNS_NUMBA = pytest.warns(
    RuntimeWarning, match="numba is not installed; falling back to slow conversion"
)


@pytest.mark.parametrize("to_cpu_memory", [True, False], ids=["to_cpu_memory", "not_to_cpu_memory"])
def test_to_dense(array_type: ArrayType[Array], *, to_cpu_memory: bool) -> None:
    x = array_type([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    if not to_cpu_memory and array_type.cls in {types.CSCDataset, types.CSRDataset}:
        with pytest.raises(
            ValueError, match="to_cpu_memory must be True if x is an CS{R,C}Dataset"
        ):
            to_dense(x, to_cpu_memory=to_cpu_memory)
        return

    with (
        WARNS_NUMBA
        if issubclass(array_type.cls, types.CSBase) and not find_spec("numba")
        else nullcontext()
    ):
        arr = to_dense(x, to_cpu_memory=to_cpu_memory)
    assert_expected_cls(x, arr, to_cpu_memory=to_cpu_memory)
    assert arr.shape == (2, 3)


@pytest.mark.parametrize("to_cpu_memory", [True, False], ids=["to_cpu_memory", "not_to_cpu_memory"])
def test_to_dense_extra(coo_matrix_type: ArrayType[Array], *, to_cpu_memory: bool) -> None:
    src_mtx = coo_matrix_type([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    with WARNS_NUMBA if not find_spec("numba") else nullcontext():
        arr = to_dense(src_mtx, to_cpu_memory=to_cpu_memory)
    assert_expected_cls(src_mtx, arr, to_cpu_memory=to_cpu_memory)
    assert arr.shape == (2, 3)


def assert_expected_cls(orig: Array, converted: Array, *, to_cpu_memory: bool) -> None:
    match (to_cpu_memory, orig):
        case False, types.DaskArray():
            assert isinstance(converted, types.DaskArray)
            assert_expected_cls(orig._meta, converted._meta, to_cpu_memory=to_cpu_memory)  # noqa: SLF001
        case False, types.CupyArray() | types.CupySpMatrix():
            assert isinstance(converted, types.CupyArray)
        case _:
            assert isinstance(converted, np.ndarray)
