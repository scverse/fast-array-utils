# SPDX-License-Identifier: MPL-2.0
"""Shared types."""

from __future__ import annotations

from functools import cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, cast, runtime_checkable

from ._import import import_by_qualname


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import UnionType


__all__ = [
    "CSBase",
    "CupyArray",
    "CupySparseMatrix",
    "DaskArray",
    "H5Dataset",
    "OutOfCoreDataset",
    "ZarrArray",
]

T_co = TypeVar("T_co", covariant=True)


# registry for lazy exports:


_REGISTRY: dict[str, str | Callable[[], UnionType]] = {}


def _register(name: str) -> Callable[[Callable[[], UnionType]], Callable[[], UnionType]]:
    def _decorator(fn: Callable[[], UnionType]) -> Callable[[], UnionType]:
        _REGISTRY[name] = fn
        return fn

    return _decorator


@cache
def __getattr__(name: str) -> type | UnionType:
    if (source := _REGISTRY.get(name)) is None:
        # A name we don’t know about
        raise AttributeError(name) from None

    try:
        if callable(source):
            return source()

        return cast(type, import_by_qualname(source))
    except ImportError:  # A name we can’t import
        return type(name, (), {})


# lazy exports:


if TYPE_CHECKING:
    from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

    CSArray = csr_array | csc_array
    CSMatrix = csr_matrix | csc_matrix
    CSBase = CSMatrix | CSArray
else:
    # cs?_array isn’t available in older scipy versions,
    # so we import them separately

    @_register("CSMatrix")
    def _get_cs_matrix() -> UnionType:
        from scipy.sparse import csc_matrix, csr_matrix

        return csr_matrix | csc_matrix

    @_register("CSArray")
    def _get_cs_array() -> UnionType:
        from scipy.sparse import csc_array, csr_array

        return csr_array | csc_array

    @_register("CSBase")
    def _get_cs_base() -> UnionType:
        return __getattr__("CSMatrix") | __getattr__("CSArray")


if TYPE_CHECKING:
    from cupy import ndarray as CupyArray
    from cupyx.scipy.sparse import spmatrix as CupySparseMatrix
else:
    _REGISTRY["CupyArray"] = "cupy:ndarray"
    _REGISTRY["CupySparseMatrix"] = "cupyx.scipy.sparse:spmatrix"


if TYPE_CHECKING:  # https://github.com/dask/dask/issues/8853
    from dask.array.core import Array as DaskArray
else:
    _REGISTRY["DaskArray"] = "dask.array:Array"


if TYPE_CHECKING:
    from h5py import Dataset as H5Dataset
else:
    _REGISTRY["H5Dataset"] = "h5py:Dataset"


if TYPE_CHECKING or find_spec("zarr"):
    from zarr import Array as ZarrArray
else:
    _REGISTRY["ZarrArray"] = "zarr:Array"


# protocols:


@runtime_checkable
class OutOfCoreDataset(Protocol, Generic[T_co]):
    """An out-of-core dataset."""

    def to_memory(self) -> T_co:
        """Load data into memory."""
        ...
