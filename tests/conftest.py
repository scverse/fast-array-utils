# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import pytest

from testing.fast_array_utils import ArrayType
from testing.fast_array_utils.pytest import _skip_if_unimportable


if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_array_utils import types


@pytest.fixture
def dask_viz(request: pytest.FixtureRequest, cache: pytest.Cache) -> Callable[[object], None]:
    """Visualize dask arrays using ipycytoscape and store in pytest’s cache."""

    def viz(obj: object) -> None:
        from fast_array_utils.types import DaskArray

        if not isinstance(obj, DaskArray) or not find_spec("ipycytoscape"):
            return

        path = cache.mkdir("dask-viz") / cast("pytest.Item", request.node).name
        obj.visualize(str(path), engine="ipycytoscape")

    return viz


COO_PARAMS = [
    pytest.param(at := ArrayType(mod, name), id=f"{mod}.{name}", marks=_skip_if_unimportable(at))
    for mod, name in [
        ("scipy.sparse", "coo_matrix"),
        ("scipy.sparse", "coo_array"),
        ("cupyx.scipy.sparse", "coo_matrix"),
    ]
]


@pytest.fixture(scope="session", params=COO_PARAMS)
def coo_matrix_type(request: pytest.FixtureRequest) -> ArrayType[types.COOBase | types.CupyCOOMatrix]:
    return cast("ArrayType[types.COOBase | types.CupyCOOMatrix]", request.param)
