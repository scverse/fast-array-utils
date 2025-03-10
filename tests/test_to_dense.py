# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import types
from fast_array_utils.conv import to_dense


if TYPE_CHECKING:
    from fast_array_utils.typing import CpuArray, DiskArray, GpuArray
    from testing.fast_array_utils import ArrayType


@pytest.mark.parametrize("to_memory", [True, False], ids=["to_memory", "not_to_memory"])
def test_to_dense(
    array_type: ArrayType[CpuArray | GpuArray | DiskArray | types.CSDataset | types.DaskArray],
    *,
    to_memory: bool,
) -> None:
    x = array_type([[1, 2, 3], [4, 5, 6]])
    if not to_memory and array_type.cls in {types.CSCDataset, types.CSRDataset}:
        with pytest.raises(ValueError, match="to_memory must be True if x is an CS{R,C}Dataset"):
            to_dense(x, to_memory=to_memory)
        return

    arr = to_dense(x, to_memory=to_memory)
    match (to_memory, x):
        case False, types.DaskArray():
            assert isinstance(arr, types.DaskArray)
            assert isinstance(arr._meta, np.ndarray)  # noqa: SLF001
        case False, types.CupyArray() | types.CupySparseMatrix():
            assert isinstance(arr, types.CupyArray)
        case _:
            assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
