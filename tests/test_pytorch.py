# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import stats
from fast_array_utils.conv import to_dense


if TYPE_CHECKING:
    from typing import Any, Literal

pytestmark = pytest.mark.skipif(not find_spec("torch"), reason="torch not installed")


@pytest.fixture
def torch_arr() -> Any:  # noqa: ANN401
    import torch

    return torch.tensor([[1, 0], [2, 0], [3, 0]], dtype=torch.float32)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_sum(torch_arr: Any, axis: Literal[0, 1] | None) -> None:  # noqa: ANN401
    import torch

    result = stats.sum(torch_arr, axis=axis)
    expected = torch.sum(torch_arr, dim=axis) if axis is not None else torch.sum(torch_arr)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_min(torch_arr: Any, axis: Literal[0, 1] | None) -> None:  # noqa: ANN401
    import torch

    result = stats.min(torch_arr, axis=axis)
    expected = torch.min(torch_arr, dim=axis).values if axis is not None else torch.min(torch_arr)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_max(torch_arr: Any, axis: Literal[0, 1] | None) -> None:  # noqa: ANN401
    import torch

    result = stats.max(torch_arr, axis=axis)
    expected = torch.max(torch_arr, dim=axis).values if axis is not None else torch.max(torch_arr)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_mean(torch_arr: Any, axis: Literal[0, 1] | None) -> None:  # noqa: ANN401
    import torch

    result = stats.mean(torch_arr, axis=axis)
    expected = torch.mean(torch_arr, dim=axis) if axis is not None else torch.mean(torch_arr)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_is_constant(axis: Literal[0, 1] | None) -> None:
    import torch

    x = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    result = stats.is_constant(x, axis=axis)

    if axis is None:
        assert bool(result) is False
    elif axis == 0:
        expected = torch.tensor([True, True, False, False])
        assert torch.equal(result, expected)
    else:
        expected = torch.tensor([False, False, True, True, False, True])
        assert torch.equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_mean_var(torch_arr: Any, axis: Literal[0, 1] | None) -> None:  # noqa: ANN401
    import torch

    mean, var = stats.mean_var(torch_arr, axis=axis, correction=1)

    # converting to float64 to match the result
    arr64 = torch_arr.to(torch.float64)
    mean_expected = torch.mean(arr64, dim=axis) if axis is not None else torch.mean(arr64)
    n = torch_arr.numel() if axis is None else torch_arr.shape[axis]
    var_expected = torch.var(arr64, dim=axis, correction=0) * n / (n - 1) if axis is not None else torch.var(arr64, correction=0) * n / (n - 1)

    assert torch.allclose(mean, mean_expected)
    assert torch.allclose(var, var_expected)


def test_to_dense(torch_arr: Any) -> None:  # noqa: ANN401
    import torch

    result = to_dense(torch_arr)
    assert torch.equal(result, torch_arr)


def test_to_dense_to_cpu(torch_arr: Any) -> None:  # noqa: ANN401
    result = to_dense(torch_arr, to_cpu_memory=True)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, torch_arr.numpy())
