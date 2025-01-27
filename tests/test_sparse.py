# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import sparse as sp

from fast_array_utils.scipy.to_dense import to_dense


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from numpy.typing import NDArray

    from fast_array_utils.types import CSBase


pytestmark = [pytest.mark.skipif(not find_spec("scipy"), reason="scipy not installed")]


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("format_", [sp.csr_array, sp.csc_array, sp.csr_matrix, sp.csc_matrix])
def test_to_dense(
    order: Literal["C", "F"],
    format_: Callable[[NDArray[Any]], CSBase[Any]],
) -> None:
    rng = np.random.default_rng()
    mat = format_(rng.random((10, 10)))
    arr = to_dense(mat, order=order)
    assert arr.flags[order]
    np.testing.assert_equal(arr, mat.toarray(order=order))
