# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import pytest

from testing.fast_array_utils import ArrayType
from testing.fast_array_utils.pytest import _skip_if_unimportable


if TYPE_CHECKING:
    from collections.abc import Callable


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


@pytest.fixture(
    scope="session",
    params=[
        ("scipy.sparse", "coo_matrix"),
        ("scipy.sparse", "coo_array"),
        ("cupyx.scipy.sparse", "coo_matrix"),
    ],
    ids=["scipy.sparse.coo_matrix", "scipy.sparse.coo_array", "cupyx.scipy.sparse.coo_matrix"],
)
def coo_matrix_type(request: pytest.FixtureRequest) -> ArrayType:
    at = ArrayType(*request.param)
    request.applymarker(_skip_if_unimportable(at))
    return at
