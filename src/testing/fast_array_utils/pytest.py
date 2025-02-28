# SPDX-License-Identifier: MPL-2.0
"""Pytest fixtures to get supported array types.

Can be used as pytest plugin: ``pytest -p testing.fast_array_utils.pytest``.
"""

from __future__ import annotations

import dataclasses
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import pytest

from . import SUPPORTED_TYPES, ArrayType, ConversionContext, Flags


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from _pytest.nodes import Node
else:
    Node = object


__all__ = ["array_type", "conversion_context"]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "array_type: filter tests using `testing.fast_array_utils.Flags`"
    )


def _skip_if_unimportable(array_type: ArrayType) -> pytest.MarkDecorator:
    dist = None
    skip = False
    for t in (array_type, array_type.inner):
        if t and not find_spec(dist := t.mod.split(".", 1)[0]):
            skip = True
    return pytest.mark.skipif(skip, reason=f"{dist} not installed")


def _resolve_sel(
    select: Flags = ~Flags(0), skip: Flags = Flags(0), *, reason: str | None = None
) -> tuple[Flags, Flags, str | None]:
    return select, skip, reason


@pytest.fixture(
    params=[pytest.param(t, id=str(t), marks=_skip_if_unimportable(t)) for t in SUPPORTED_TYPES],
)
def array_type(request: pytest.FixtureRequest) -> ArrayType:
    """Fixture for a supported :class:`~testing.fast_array_utils.ArrayType`.

    Use :class:`testing.fast_array_utils.Flags` to select or skip array types

    #.  using ``select=``/``args[0]``:

        ..  code:: python

            @pytest.mark.array_type(Flags.Sparse, reason="`something` only supports sparse arrays")
            def test_something(array_type: ArrayType) -> None:
                ...

    #.  and/or using ``skip=``/``args[1]``:

        .. code:: python

            @pytest.mark.array_type(skip=Flags.Dask | Flags.Disk | Flags.Gpu)
            def test_something(array_type: ArrayType) -> None:
                ...
    """
    from fast_array_utils.types import H5Dataset

    at = cast(ArrayType, request.param)

    mark = cast(Node, request.node).get_closest_marker("array_type")
    if mark:
        select, skip, reason = _resolve_sel(*mark.args, **mark.kwargs)
        if not (at.flags & select) or (at.flags & skip):
            pytest.skip(reason or f"{at} not included in {select=}, {skip=}")

    if at.cls is H5Dataset:
        ctx = request.getfixturevalue("conversion_context")
        at = dataclasses.replace(at, conversion_context=ctx)

    return at


@pytest.fixture
# worker_id for xdist since we don't want to override open files
def conversion_context(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str = "serial",
) -> Generator[ConversionContext, None, None]:
    """Fixture providing a :class:`~testing.fast_array_utils.ConversionContext`.

    Makes sure h5py works even when running tests in parallel.
    """
    import h5py

    node = cast(Node, request.node)
    tmp_path = tmp_path_factory.mktemp("backed_adata")
    tmp_path = tmp_path / f"test_{node.name}_{worker_id}.h5ad"

    with h5py.File(tmp_path, "x") as f:
        yield ConversionContext(hdf5_file=f)


@pytest.fixture
def dask_viz(request: pytest.FixtureRequest, cache: pytest.Cache) -> Callable[[object], None]:
    def viz(obj: object) -> None:
        from fast_array_utils.types import DaskArray

        if not isinstance(obj, DaskArray) or not find_spec("ipycytoscape"):
            return

        path = cache.mkdir("dask-viz") / cast(Node, request.node).name
        obj.visualize(str(path), engine="ipycytoscape")  # type: ignore[no-untyped-call]

    return viz
