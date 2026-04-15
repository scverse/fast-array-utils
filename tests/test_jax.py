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

pytestmark = pytest.mark.skipif(not find_spec("jax"), reason="jax not installed")

if find_spec("jax"):
    # enabling 64-bit precision in JAX as it defaults to 32-bit only
    # problem as mean_var passes dtype= np.float64 internally, which crashes without this fix
    import jax

    jax.config.update("jax_enable_x64", True)


@pytest.fixture
def jax_arr() -> Any:  # noqa: ANN401
    import jax.numpy as jnp

    return jnp.array([[1, 0], [2, 0], [3, 0]], dtype=jnp.float32)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_sum(jax_arr: Any, axis: Literal[0, 1] | None) -> None:  # noqa: ANN401
    import jax.numpy as jnp

    result = stats.sum(jax_arr, axis=axis)
    expected = jnp.sum(jax_arr, axis=axis)
    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_min(jax_arr: Any, axis: Literal[0, 1] | None) -> None:
    import jax.numpy as jnp

    result = stats.min(jax_arr, axis=axis)
    expected = jnp.min(jax_arr, axis=axis)
    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_max(jax_arr: Any, axis: Literal[0, 1] | None) -> None:  # noqa: ANN401
    import jax.numpy as jnp

    result = stats.max(jax_arr, axis=axis)
    expected = jnp.max(jax_arr, axis=axis)
    assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_mean(jax_arr: Any, axis: Literal[0, 1] | None) -> None:  # noqa: ANN401
    import jax.numpy as jnp

    result = stats.mean(jax_arr, axis=axis)
    expected = jnp.mean(jax_arr, axis=axis)
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_is_constant(axis: Literal[0, 1] | None) -> None:
    import jax.numpy as jnp

    x = jnp.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=jnp.float32,
    )
    result = stats.is_constant(x, axis=axis)

    if axis is None:
        assert bool(result) is False
    elif axis == 0:
        expected = jnp.array([True, True, False, False])
        assert jnp.array_equal(result, expected)
    else:
        expected = jnp.array([False, False, True, True, False, True])
        assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_mean_var(jax_arr: Any, axis: Literal[0, 1] | None) -> None:  # noqa: ANN401
    import jax.numpy as jnp

    mean, var = stats.mean_var(jax_arr, axis=axis, correction=1)

    mean_expected = jnp.mean(jax_arr, axis=axis)
    n = jax_arr.size if axis is None else jax_arr.shape[axis]
    var_expected = jnp.var(jax_arr, axis=axis) * n / (n - 1)

    assert jnp.allclose(mean, mean_expected)
    assert jnp.allclose(var, var_expected)


def test_to_dense(jax_arr: Any) -> None:  # noqa: ANN401
    import jax.numpy as jnp

    result = to_dense(jax_arr)
    assert jnp.array_equal(result, jax_arr)


def test_to_dense_to_cpu(jax_arr: Any) -> None:  # noqa: ANN401
    result = to_dense(jax_arr, to_cpu_memory=True)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.asarray(jax_arr))
