# SPDX-License-Identifier: MPL-2.0
"""Pytest fixtures to get supported array types.

Can be used as pytest plugin: ``pytest -p testing.fast_array_utils.pytest``.
"""

from __future__ import annotations

import dataclasses
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import pytest

from fast_array_utils import types
from testing.fast_array_utils import SUPPORTED_TYPES, ArrayType, ConversionContext, Flags


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from pathlib import Path

    import h5py


__all__ = ["SUPPORTED_TYPE_PARAMS", "array_type"]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "array_type: filter tests using `testing.fast_array_utils.Flags`"
    )


def _selected(
    array_type: ArrayType,
    /,
    select: Flags | ArrayType | Iterable[Flags | ArrayType] = ~Flags(0),
    skip: Flags | ArrayType | Iterable[Flags | ArrayType] = Flags(0),
) -> bool:
    """Check if ``array_type`` matches.

    Returns ``True`` if ``array_type`` matches (one of the conditions in) ``select``
    and does not match (one of the conditions in) ``skip``.
    """
    if isinstance(select, Flags | ArrayType):
        select = [select]
    if isinstance(skip, Flags | ArrayType):
        skip = [skip]

    def matches(selector: Flags | ArrayType) -> bool:
        if isinstance(selector, ArrayType):
            return array_type == selector
        return bool(array_type.flags & selector)

    return any(map(matches, select)) and not any(map(matches, skip))


def pytest_collection_modifyitems(
    session: pytest.Session,  # noqa: ARG001
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    """Filter tests using `pytest.mark.array_type` based on `testing.fast_array_utils.Flags`."""
    # reverse so we can .pop() items from the back without changing others’ index
    for i, item in reversed(list(enumerate(items))):
        if not (
            isinstance(item, pytest.Function) and (mark := item.get_closest_marker("array_type"))
        ):
            continue

        msg = "Test function marked with `pytest.mark.array_type` must have `array_type` parameter"
        if not (at := item.callspec.params.get("array_type")):
            raise TypeError(msg)
        if not isinstance(at, ArrayType):
            msg = f"{msg} of type {ArrayType.__name__}, got {type(at).__name__}"
            raise TypeError(msg)
        if not _selected(at, *mark.args, **mark.kwargs):
            del items[i]


def _skip_if_unimportable(array_type: ArrayType) -> pytest.MarkDecorator:
    dist = None
    skip = False
    for t in (array_type, array_type.inner):
        if t and not find_spec(dist := t.mod.split(".", 1)[0]):
            skip = True
    return pytest.mark.skipif(skip, reason=f"{dist} not installed")


SUPPORTED_TYPE_PARAMS = [
    pytest.param(t, id=str(t), marks=_skip_if_unimportable(t)) for t in SUPPORTED_TYPES
]


@pytest.fixture(params=SUPPORTED_TYPE_PARAMS)
def array_type(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[ArrayType, None, None]:
    """Fixture for a supported :class:`~testing.fast_array_utils.ArrayType`.

    Use :class:`testing.fast_array_utils.Flags` to select or skip array types:

    #.  using ``select=``/``args[0]``:

        ..  code:: python

            @pytest.mark.array_type(Flags.Sparse)
            def test_something(array_type: ArrayType) -> None:
                ...

    #.  and/or using ``skip=``/``args[1]``:

        .. code:: python

            @pytest.mark.array_type(skip=Flags.Dask | Flags.Disk | Flags.Gpu)
            def test_something(array_type: ArrayType) -> None:
                ...

    For special cases, you can also specify a :class:`set` of array types and flags.
    This is useful if you want to select or skip only specific array types.

    .. code:: python

        from testing.fast_array_utils import SUPPORTED_TYPES

        SPARSE_AND_DASK = {
            at for at in SUPPORTED_TYPES if at.flags & Flags.Sparse and at.flags & Flags.Dask
        }

        @pytest.mark.array_type(skip={*SPARSE_AND_DASK, Flags.Disk})
        def test_something(array_type: ArrayType) -> None:
            ...
    """
    at = cast("ArrayType", request.param)
    f: h5py.File | None = None
    if at.cls is types.H5Dataset or (at.inner and at.inner.cls is types.H5Dataset):
        import h5py

        f = h5py.File(tmp_path / f"{request.fixturename}.h5", "w")
        ctx = ConversionContext(hdf5_file=f)
        at = dataclasses.replace(at, conversion_context=ctx)
    yield at
    if f:
        f.close()
