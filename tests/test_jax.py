# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fast_array_utils import stats
from fast_array_utils.conv import to_dense


if TYPE_CHECKING:
    from typing import Literal


pytestmark = pytest.mark.skipif(not find_spec("jax"), reason="jax not installed")

if find_spec("jax"):
    # enabling 64-bit precision in JAX as it defaults to 32-bit only
    # problem as mean_var passes dtype= np.float64 internally, which crashes without this fix
    import jax

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]  # noqa: FBT003


@pytest.fixture
def jax_arr() -> jax.Array:
    import jax.numpy as jnp

    return jnp.array([[1, 0], [2, 0], [3, 0]], dtype=jnp.float32)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("func", ["sum", "min", "max", "mean"])
def test_simple_stat(jax_arr: jax.Array, func: Literal["sum", "min", "max", "mean"], axis: Literal[0, 1] | None) -> None:
    import jax.numpy as jnp

    result = getattr(stats, func)(jax_arr, axis=axis)
    expected = getattr(jnp, func)(jax_arr, axis=axis)

    assert type(result) is type(expected)
    if func == "mean":
        assert jnp.allclose(result, expected)
    else:
        assert jnp.array_equal(result, expected)


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
        assert not result
    elif axis == 0:
        expected = jnp.array([True, True, False, False])
        assert type(result) is type(expected)
        assert jnp.array_equal(result, expected)
    else:
        expected = jnp.array([False, False, True, True, False, True])
        assert type(result) is type(expected)
        assert jnp.array_equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_mean_var(subtests: pytest.Subtests, jax_arr: jax.Array, axis: Literal[0, 1] | None) -> None:
    import jax.numpy as jnp

    mean, var = stats.mean_var(jax_arr, axis=axis, correction=1)

    for name, result in dict(mean=mean, var=var).items():
        if name == "mean":
            expected = jnp.mean(jax_arr, axis=axis)
        else:
            n = jax_arr.size if axis is None else jax_arr.shape[axis]
            expected = jnp.var(jax_arr, axis=axis) * n / (n - 1)

        with subtests.test(name):
            assert type(result) is type(expected)
            assert jnp.allclose(result, expected)


@pytest.mark.parametrize("to_cpu_memory", [True, False], ids=["to_cpu_memory", "not_to_cpu_memory"])
def test_to_dense(*, jax_arr: jax.Array, to_cpu_memory: bool) -> None:
    import jax.numpy as jnp

    result = to_dense(jax_arr, to_cpu_memory=to_cpu_memory)

    if to_cpu_memory:
        assert isinstance(result, np.ndarray)
    else:
        assert isinstance(result, jax.Array)
    assert jnp.array_equal(result, jax_arr)
