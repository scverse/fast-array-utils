# SPDX-License-Identifier: MPL-2.0
"""Testing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from typing import Any, Literal, Protocol, SupportsFloat, TypeAlias

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from fast_array_utils import types
    from fast_array_utils.types import CSBase

    Array: TypeAlias = (
        NDArray[Any]
        | types.CSBase
        | types.CupyArray
        | types.DaskArray
        | types.H5Dataset
        | types.ZarrArray
    )

    class ToArray(Protocol):
        """Convert to a supported array."""

        def __call__(  # noqa: D102
            self, data: ArrayLike, /, *, dtype: DTypeLike | None = None
        ) -> Array: ...

    _DTypeLikeFloat32 = np.dtype[np.float32] | type[np.float32]
    _DTypeLikeFloat64 = np.dtype[np.float64] | type[np.float64]


@dataclass
class ArrayType:
    """Supported array type."""

    mod: str
    name: str
    inner: ArrayType | None

    @classmethod
    @cache
    def from_qualname(cls, qualname: str, inner: str | None = None) -> ArrayType:
        """Get a supported array type by qualname."""
        mod, name = qualname.rsplit(".", 1)
        return cls(mod, name, ArrayType.from_qualname(inner) if inner else None)

    def __str__(self) -> str:  # noqa: D105
        rv = f"{self.mod}.{self.name}"
        return f"{rv}[{self.inner}]" if self.inner else rv

    @cached_property
    def cls(self) -> type[Array]:  # noqa: PLR0911
        """Get a supported array class by qualname."""
        match self.mod, self.name, self.inner:
            case "numpy", "ndarray", None:
                return np.ndarray
            case "scipy.sparse", (
                "csr_array" | "csc_array" | "csr_matrix" | "csc_matrix"
            ) as cls_name, None:
                import scipy.sparse

                return getattr(scipy.sparse, cls_name)  # type: ignore[no-any-return]
            case "cupy", "ndarray", None:
                import cupy as cp

                return cp.ndarray  # type: ignore[no-any-return]
            case "cupyx.scipy.sparse", ("csr_matrix" | "csc_matrix") as cls_name, None:
                import cupyx.scipy.sparse as cu_sparse

                return getattr(cu_sparse, cls_name)  # type: ignore[no-any-return]
            case "dask.array", cls_name, _:
                if TYPE_CHECKING:
                    from dask.array.core import Array as DaskArray
                else:
                    from dask.array import Array as DaskArray

                return DaskArray
            case "h5py", "Dataset", _:
                import h5py

                return h5py.Dataset  # type: ignore[no-any-return]
            case "zarr", "Array", _:
                import zarr

                return zarr.Array
            case _:
                msg = f"Unknown array class: {self}"
                raise ValueError(msg)

    def random(
        self,
        shape: tuple[int, int],
        *,
        dtype: _DTypeLikeFloat32 | _DTypeLikeFloat64 | None,
        gen: np.random.Generator | None = None,
    ) -> Array:
        """Create a random array."""
        gen = np.random.default_rng(gen)

        match self.mod, self.name, self.inner:
            case "numpy", "ndarray", None:
                return gen.random(shape, dtype=dtype or np.float64)
            case "scipy.sparse", (
                "csr_array" | "csc_array" | "csr_matrix" | "csc_matrix"
            ) as cls_name, None:
                fmt, container = cls_name.split("_")
                return random_mat(shape, format=fmt, container=container, dtype=dtype)  # type: ignore[arg-type]
            case "cupy", "ndarray", None:
                raise NotImplementedError
            case "cupyx.scipy.sparse", ("csr_matrix" | "csc_matrix") as cls_name, None:
                raise NotImplementedError
            case "dask.array", cls_name, _:
                raise NotImplementedError
            case "h5py", "Dataset", _:
                raise NotImplementedError
            case "zarr", "Array", _:
                raise NotImplementedError
            case _:
                msg = f"Unknown array class: {self}"
                raise ValueError(msg)


_SUPPORTED_TYPE_NAMES_IN_DASK = [
    "numpy.ndarray",
    "scipy.sparse.csr_array",
    "scipy.sparse.csc_array",
    "scipy.sparse.csr_matrix",
    "scipy.sparse.csc_matrix",
]
_SUPPORTED_TYPE_NAMES_OTHER = [
    "h5py.Dataset",
    "zarr.Array",
    "cupy.ndarray",
    "cupyx.scipy.sparse.csr_matrix",
    "cupyx.scipy.sparse.csc_matrix",
]
SUPPORTED_TYPES_IN_DASK = tuple(map(ArrayType.from_qualname, _SUPPORTED_TYPE_NAMES_IN_DASK))
SUPPORTED_TYPES_DASK = tuple(
    ArrayType.from_qualname("dask.array.Array", t) for t in _SUPPORTED_TYPE_NAMES_IN_DASK
)
SUPPORTED_TYPES_OTHER = tuple(map(ArrayType.from_qualname, _SUPPORTED_TYPE_NAMES_OTHER))
SUPPORTED_TYPES = (*SUPPORTED_TYPES_IN_DASK, *SUPPORTED_TYPES_DASK, *SUPPORTED_TYPES_OTHER)


def random_mat(
    shape: tuple[int, int],
    *,
    density: SupportsFloat = 0.01,
    format: Literal["csr", "csc"] = "csr",  # noqa: A002
    dtype: DTypeLike | None = None,
    container: Literal["array", "matrix"] = "array",
    gen: np.random.Generator | None = None,
) -> CSBase:
    """Create a random matrix."""
    from scipy.sparse import random as random_spmat
    from scipy.sparse import random_array as random_sparr

    m, n = shape
    return (
        random_spmat(m, n, density=density, format=format, dtype=dtype, random_state=gen)
        if container == "matrix"
        else random_sparr(shape, density=density, format=format, dtype=dtype, random_state=gen)
    )
