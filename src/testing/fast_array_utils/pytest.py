# SPDX-License-Identifier: MPL-2.0
"""Pytest fixtures to get supported array types.

Can be used as pytest plugin: ``pytest -p testing.fast_array_utils.pytest``.
"""

from __future__ import annotations

import dataclasses
import os
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import pytest

from . import SUPPORTED_TYPES, ArrayType, ConversionContext


if TYPE_CHECKING:
    from collections.abc import Generator


__all__ = ["array_type", "conversion_context"]


def _skip_if_no(dist: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not find_spec(dist), reason=f"{dist} not installed")


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(t, id=str(t), marks=_skip_if_no(t.mod.split(".")[0])) for t in SUPPORTED_TYPES
    ],
)
def array_type(request: pytest.FixtureRequest) -> ArrayType:
    """Fixture for a supported :class:`~testing.fast_array_utils.ArrayType`."""
    from fast_array_utils.types import H5Dataset

    at = cast(ArrayType, request.param)
    if at.cls is H5Dataset:
        ctx = request.getfixturevalue("conversion_context")
        at = dataclasses.replace(at, conversion_context=ctx)
    return at


@pytest.fixture(scope="session")
# worker_id for xdist since we don't want to override open files
def conversion_context(
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str = "serial",
) -> Generator[ConversionContext, None, None]:
    """Fixture providing a :class:`~testing.fast_array_utils.ConversionContext`.

    Makes sure h5py works even when running tests in parallel.
    """
    import h5py

    tmp_path = tmp_path_factory.mktemp("backed_adata")
    tmp_path = tmp_path / f"test_{worker_id}.h5ad"

    def get_ds_name() -> str:
        return os.environ["PYTEST_CURRENT_TEST"].rsplit(":", 1)[-1].split(" ", 1)[0]

    with h5py.File(tmp_path, "x") as f:
        yield ConversionContext(hdf5_file=f, get_ds_name=get_ds_name)
